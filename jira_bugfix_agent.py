"""
Jira Bug-Fix Agent (LangGraph + Claude)
========================================
An autonomous coding agent that reads a Jira bug ticket, localizes the fault
in a target repository, generates and applies a patch, verifies it against
the test suite, and suggests a commit — with a human approval gate before
any commit is created.

Flow:
    fetch_ticket -> localize -> plan -> patch -> test
                                  ^              |
                                  +--- retry <---+   (max 3 attempts)
                                                 |
                                       suggest_commit -> [human approval] -> commit

Usage:
    python jira_bugfix_agent.py --ticket PROJ-123 --repo /path/to/repo

Env vars required (see .env.example):
    ANTHROPIC_API_KEY, JIRA_SERVER, JIRA_EMAIL, JIRA_API_TOKEN
"""

import argparse
import os
import re
import subprocess
from typing import Literal, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

MAX_RETRIES = 3
MODEL = "claude-sonnet-4-6"

llm = ChatAnthropic(model=MODEL, max_tokens=4096, temperature=0)


# ---------------------------------------------------------------------------
# 1. State — kept deliberately small. The repo lives on disk; tools read it
#    lazily. We carry the diff, not the codebase.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    ticket_key: str
    repo_path: str
    ticket_summary: str
    ticket_description: str
    candidate_files: list[str]
    fault_hypothesis: str
    fix_plan: str
    patch: str                      # unified diff of the current attempt
    test_output: str
    tests_passed: bool
    attempts: int
    commit_message: Optional[str]
    status: str


# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------
def jira_client():
    from jira import JIRA
    return JIRA(
        server=os.environ["JIRA_SERVER"],
        basic_auth=(os.environ["JIRA_EMAIL"], os.environ["JIRA_API_TOKEN"]),
    )


def grep_repo(repo_path: str, pattern: str, max_hits: int = 20) -> list[str]:
    """Return files containing the pattern (cheap fault localization)."""
    try:
        out = subprocess.run(
            ["grep", "-rIl", "--exclude-dir=.git", "--exclude-dir=node_modules",
             pattern, repo_path],
            capture_output=True, text=True, timeout=30,
        )
        files = [f for f in out.stdout.strip().splitlines() if f]
        return files[:max_hits]
    except subprocess.TimeoutExpired:
        return []


def read_file(path: str, max_chars: int = 12000) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read(max_chars)
    except OSError:
        return ""


def run_tests(repo_path: str) -> tuple[bool, str]:
    """Run the repo's test suite. Runner is chosen by repo markers, not by
    which binaries happen to be installed on the machine."""
    import glob
    runners: list[list[str]] = []
    has_py_tests = os.path.isdir(os.path.join(repo_path, "tests")) or bool(
        glob.glob(os.path.join(repo_path, "test_*.py")))
    if has_py_tests and _tool_available("pytest"):
        runners.append(["pytest", "-x", "-q"])
    if os.path.exists(os.path.join(repo_path, "package.json")) and _tool_available("npm"):
        runners.append(["npm", "test", "--", "--watch=false"])
    if (glob.glob(os.path.join(repo_path, "*.sln")) or glob.glob(
            os.path.join(repo_path, "**", "*.csproj"), recursive=True)) and _tool_available("dotnet"):
        runners.append(["dotnet", "test"])
    if has_py_tests:
        runners.append(["python3", "-m", "unittest", "discover", "-s", "tests", "-v"])

    for cmd in runners:
        proc = subprocess.run(
            cmd, cwd=repo_path, capture_output=True, text=True, timeout=600
        )
        output = (proc.stdout + proc.stderr)[-6000:]
        return proc.returncode == 0, output
    return False, "No test runner matched this repo (pytest/unittest, npm, dotnet)."


def _tool_available(name: str) -> bool:
    return subprocess.run(["which", name], capture_output=True).returncode == 0


def _git(repo_path: str, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo_path, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------
def fetch_ticket(state: AgentState) -> dict:
    if state.get("ticket_summary"):  # --local-ticket mode: ticket supplied directly
        return {"attempts": 0, "status": "ticket_fetched"}
    issue = jira_client().issue(state["ticket_key"])
    return {
        "ticket_summary": issue.fields.summary,
        "ticket_description": issue.fields.description or "",
        "attempts": 0,
        "status": "ticket_fetched",
    }


def localize(state: AgentState) -> dict:
    """Fault localization, cheapest strategy first:
    1. File paths / stack-trace frames named in the ticket
    2. grep for quoted error strings
    3. Ask the model to propose search terms, then grep those
    """
    text = f"{state['ticket_summary']}\n{state['ticket_description']}"
    candidates: list[str] = []

    # Strategy 1: explicit file references in the ticket
    for match in re.findall(r"[\w/\\.-]+\.(?:py|cs|ts|js|html)", text):
        full = os.path.join(state["repo_path"], match.lstrip("/\\"))
        if os.path.exists(full):
            candidates.append(full)

    # Strategy 2: grep quoted error strings from the ticket
    if not candidates:
        for quoted in re.findall(r"[\"'](.{8,80}?)[\"']", text):
            candidates.extend(grep_repo(state["repo_path"], quoted))
            if candidates:
                break

    # Strategy 3: model-proposed search terms
    if not candidates:
        resp = llm.invoke(
            "Given this bug ticket, list 3 short, distinctive code search strings "
            "(function names, error fragments, identifiers) most likely to appear "
            "in the faulty file. One per line, no commentary.\n\n" + text
        )
        for term in resp.content.strip().splitlines()[:3]:
            candidates.extend(grep_repo(state["repo_path"], term.strip()))

    candidates = list(dict.fromkeys(candidates))[:5]

    context = "\n\n".join(
        f"=== {f} ===\n{read_file(f)}" for f in candidates
    ) or "(no candidate files found)"

    resp = llm.invoke(
        f"Bug ticket:\n{text}\n\nCandidate source files:\n{context}\n\n"
        "State a one-paragraph hypothesis of the root cause and which file/function "
        "contains it. Be specific."
    )
    return {
        "candidate_files": candidates,
        "fault_hypothesis": resp.content,
        "status": "localized",
    }


def plan(state: AgentState) -> dict:
    prior_failure = (
        f"\n\nPrevious attempt failed tests with:\n{state['test_output']}"
        if state["attempts"] > 0 else ""
    )
    resp = llm.invoke(
        f"Bug: {state['ticket_summary']}\n"
        f"Hypothesis: {state['fault_hypothesis']}{prior_failure}\n\n"
        "Write a minimal, surgical fix plan: which file(s), which lines/functions, "
        "what changes. Prefer the smallest change that fixes the root cause. "
        "Do not refactor unrelated code."
    )
    return {"fix_plan": resp.content, "status": "planned"}


def patch(state: AgentState) -> dict:
    context = "\n\n".join(
        f"=== {os.path.relpath(f, state['repo_path'])} ===\n{read_file(f)}"
        for f in state["candidate_files"]
    )
    resp = llm.invoke(
        f"Fix plan:\n{state['fix_plan']}\n\nCurrent file contents (paths are relative "
        f"to the repo root):\n{context}\n\n"
        "Output ONLY a valid unified diff (git apply format, a/ b/ prefixes built "
        "from the repo-relative paths shown above, correct hunk headers) "
        "implementing the plan. No prose, no code fences."
    )
    diff = resp.content.strip()
    diff = re.sub(r"^```(?:diff)?\n?|\n?```$", "", diff)
    if not diff.endswith("\n"):
        diff += "\n"  # git apply rejects a patch with no trailing newline

    _git(state["repo_path"], "checkout", "--", ".")  # reset any previous failed attempt

    proc = subprocess.run(
        ["git", "apply", "--whitespace=fix", "--recount", "-"],
        cwd=state["repo_path"], input=diff, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {
            "patch": diff,
            "tests_passed": False,
            "test_output": f"git apply failed:\n{proc.stderr}",
            "attempts": state["attempts"] + 1,
            "status": "patch_failed",
        }
    return {"patch": diff, "attempts": state["attempts"] + 1, "status": "patched"}


def test(state: AgentState) -> dict:
    if state["status"] == "patch_failed":
        return {}  # carry the apply failure straight to the retry decision
    passed, output = run_tests(state["repo_path"])
    return {"tests_passed": passed, "test_output": output,
            "status": "tests_passed" if passed else "tests_failed"}


def suggest_commit(state: AgentState) -> dict:
    resp = llm.invoke(
        f"Ticket {state['ticket_key']}: {state['ticket_summary']}\n"
        f"Diff:\n{state['patch']}\n\n"
        "Write a conventional-commit message. First line <= 72 chars, "
        f"reference {state['ticket_key']}, then a short body explaining the why."
    )
    commit_message = resp.content.strip()

    # -------- HUMAN-IN-THE-LOOP GATE --------
    # The graph pauses here. A human reviews the diff + message and resumes
    # with approval. The agent never commits autonomously.
    decision = interrupt({
        "diff": state["patch"],
        "commit_message": commit_message,
        "test_output": state["test_output"][-2000:],
        "question": "Approve commit? (yes/no)",
    })

    if str(decision).strip().lower() in ("yes", "y", "approve"):
        _git(state["repo_path"], "add", "-A")
        _git(state["repo_path"], "commit", "-m", commit_message)
        return {"commit_message": commit_message, "status": "committed"}
    _git(state["repo_path"], "checkout", "--", ".")
    return {"commit_message": commit_message, "status": "rejected_by_human"}


def escalate(state: AgentState) -> dict:
    """Retry budget exhausted — hand off to a human with full context."""
    print(f"\n[ESCALATION] {state['ticket_key']} could not be fixed automatically "
          f"after {state['attempts']} attempts.")
    print(f"Last hypothesis: {state['fault_hypothesis'][:500]}")
    print(f"Last test output:\n{state['test_output'][-2000:]}")
    return {"status": "escalated"}


# ---------------------------------------------------------------------------
# 4. Graph wiring
# ---------------------------------------------------------------------------
def route_after_test(state: AgentState) -> Literal["suggest_commit", "plan", "escalate"]:
    if state["tests_passed"]:
        return "suggest_commit"
    if state["attempts"] >= MAX_RETRIES:
        return "escalate"
    return "plan"  # re-plan with the failure output in context


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("fetch_ticket", fetch_ticket)
    g.add_node("localize", localize)
    g.add_node("plan", plan)
    g.add_node("patch", patch)
    g.add_node("test", test)
    g.add_node("suggest_commit", suggest_commit)
    g.add_node("escalate", escalate)

    g.set_entry_point("fetch_ticket")
    g.add_edge("fetch_ticket", "localize")
    g.add_edge("localize", "plan")
    g.add_edge("plan", "patch")
    g.add_edge("patch", "test")
    g.add_conditional_edges("test", route_after_test)
    g.add_edge("suggest_commit", END)
    g.add_edge("escalate", END)

    return g.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 5. CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Jira bug-fix agent")
    parser.add_argument("--ticket", help="Jira ticket key, e.g. PROJ-123")
    parser.add_argument("--local-ticket", help="Path to a JSON file with key/summary/description (no Jira needed)")
    parser.add_argument("--repo", required=True, help="Path to target git repository")
    args = parser.parse_args()
    if not args.ticket and not args.local_ticket:
        parser.error("provide --ticket or --local-ticket")

    initial = {"repo_path": args.repo}
    if args.local_ticket:
        import json
        with open(args.local_ticket) as f:
            t = json.load(f)
        initial.update(ticket_key=t.get("key", "LOCAL-1"),
                       ticket_summary=t["summary"],
                       ticket_description=t.get("description", ""))
    else:
        initial["ticket_key"] = args.ticket

    app = build_graph()
    config = {"configurable": {"thread_id": initial["ticket_key"]}}

    result = app.invoke(initial, config=config)

    # If the graph paused at the human gate, surface the review payload
    state = app.get_state(config)
    if state.next:  # interrupted at suggest_commit
        payload = state.tasks[0].interrupts[0].value
        print("\n===== REVIEW REQUIRED =====")
        print(payload["commit_message"])
        print("\n--- diff ---\n" + payload["diff"])
        answer = input("\nApprove commit? (yes/no): ")
        result = app.invoke(Command(resume=answer), config=config)

    print(f"\nFinal status: {result.get('status')}")


if __name__ == "__main__":
    main()
