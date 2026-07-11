# Jira Bug-Fix Agent

**An autonomous coding agent built with LangGraph and Claude.** It reads a Jira bug ticket, localizes the fault in your repository, generates and applies a patch, verifies it against your test suite, and suggests a commit — pausing for **human approval before anything is committed**.

```
Jira ticket → localize fault → plan fix → apply patch → run tests
                                   ↑                        │
                                   └────── retry ←──────────┘  (max 3)
                                                            │
                                            suggest commit → 🧑 human gate → commit
```

## Why a state machine, not a free-running loop

This agent touches real code, so it's built as a **deterministic LangGraph state graph** with explicit checkpoints rather than an open-ended ReAct loop:

- **Human-in-the-loop commit gate.** The agent *suggests* the commit; it never pushes autonomously. LangGraph's `interrupt()` pauses the graph with the diff, commit message, and test output for review. Approval resumes the graph via `Command(resume=...)`.
- **Fault localization is the hard part, not the fix.** Cheapest strategy first:
  1. File paths / stack-trace frames named directly in the ticket
  2. `grep` for quoted error strings from the ticket
  3. Model-proposed search terms, then grep those
  Most real tickets resolve at step 1 or 2 — no embeddings needed.
- **Test-verified retry loop.** Patch and test form a cycle with a retry budget. Each retry re-plans *with the failure output in context*. After 3 failed attempts, the agent **escalates to a human** with the hypothesis, diff, and failure log instead of thrashing.
- **Small state, lazy tools.** The LangGraph state carries the ticket, candidate files, the unified diff, and test output — never the codebase. The repo lives on disk; tools read files lazily.

## Setup

```bash
git clone https://github.com/SumeethPrabhakar/jira-bugfix-agent
cd jira-bugfix-agent
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
```

Required environment variables:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API access |
| `JIRA_SERVER` | e.g. `https://yourorg.atlassian.net` |
| `JIRA_EMAIL` | Jira account email |
| `JIRA_API_TOKEN` | Jira API token (Atlassian account settings) |

## Usage

```bash
python jira_bugfix_agent.py --ticket PROJ-123 --repo /path/to/target/repo
```

The agent runs the full pipeline, then prints the proposed commit message, the diff, and the test results, and waits for your `yes/no`. On rejection, the working tree is reset — no trace left behind.

Test runner is auto-detected (`pytest`, `npm test`, or `dotnet test`); adapt `run_tests()` for other stacks.

## Design notes

- **Idempotent attempts.** Every patch attempt starts from a clean tree (`git checkout -- .`), so a malformed diff from attempt 2 can't stack on top of attempt 1.
- **Diff-only patching.** The model outputs a unified diff applied via `git apply --whitespace=fix`, rather than rewriting whole files — smaller blast radius, reviewable output.
- **Checkpointed graph.** `MemorySaver` checkpoints mean the human gate can be answered in a later process (swap in a persistent checkpointer like SQLite for production).

## Limitations / roadmap

- Single-repo, single-ticket scope; no cross-repo changes
- Test detection is convention-based — monorepos need an explicit test command
- Next: post the diff back to the Jira ticket as a comment, open an MR via the GitLab API instead of committing locally

---

Built by [Sumeeth Prabhakar](https://www.linkedin.com/in/sumeethprabhakar/) · Principal Software Engineer, Sydney · [infomatebot.com](https://www.infomatebot.com)
