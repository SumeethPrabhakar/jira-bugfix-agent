"""
Unit tests for the Jira bug-fix agent's deterministic logic.

Run:  python -m unittest discover -s tests -v

The LLM client is mocked throughout — no API key or network needed.
Includes regression tests for two bugs found during fault-injection testing
(see DEMO.md): trailing-newline patch corruption, and test-runner detection
by installed binaries instead of repo markers.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jira_bugfix_agent as agent


class FakeLLM:
    """Returns a queued response per invoke() call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        content = self.responses.pop(0) if self.responses else "(fake)"
        return type("R", (), {"content": content})()


def make_repo(tmpdir, filename="calculator.py", content=None):
    """Create a small git repo fixture."""
    content = content or (
        "def divide(a, b):\n"
        "    if b == 0:\n"
        "        raise ValueError(\"division by zero\")\n"
        "    return a // b\n"
    )
    path = os.path.join(tmpdir, filename)
    with open(path, "w") as f:
        f.write(content)
    for cmd in (["git", "init", "-q"],
                ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"],
                ["git", "add", "-A"],
                ["git", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=tmpdir, check=True, capture_output=True)
    return path


def make_diff(tmpdir, old, new, filename="calculator.py"):
    """Produce a real git diff string by mutating and reverting the fixture."""
    path = os.path.join(tmpdir, filename)
    src = open(path).read()
    open(path, "w").write(src.replace(old, new))
    out = subprocess.run(["git", "diff"], cwd=tmpdir,
                         capture_output=True, text=True, check=True)
    subprocess.run(["git", "checkout", "--", "."], cwd=tmpdir,
                   check=True, capture_output=True)
    return out.stdout


class TestRouting(unittest.TestCase):
    """The retry loop's conditional edge."""

    def test_tests_passed_goes_to_commit(self):
        state = {"tests_passed": True, "attempts": 1}
        self.assertEqual(agent.route_after_test(state), "suggest_commit")

    def test_failure_under_budget_replans(self):
        state = {"tests_passed": False, "attempts": 1}
        self.assertEqual(agent.route_after_test(state), "plan")

    def test_failure_at_budget_escalates(self):
        state = {"tests_passed": False, "attempts": agent.MAX_RETRIES}
        self.assertEqual(agent.route_after_test(state), "escalate")


class TestLocalization(unittest.TestCase):
    def test_tier1_file_named_in_ticket(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)
            fake = FakeLLM(["hypothesis: floor division in divide()"])
            with patch.object(agent, "llm", fake):
                out = agent.localize({
                    "repo_path": tmp,
                    "ticket_summary": "divide() wrong result",
                    "ticket_description": "Bug is in calculator.py, divide(7,2) returns 3",
                    "attempts": 0,
                })
            self.assertEqual(len(out["candidate_files"]), 1)
            self.assertTrue(out["candidate_files"][0].endswith("calculator.py"))

    def test_tier2_grep_for_quoted_error_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)
            fake = FakeLLM(["hypothesis"])
            with patch.object(agent, "llm", fake):
                out = agent.localize({
                    "repo_path": tmp,
                    "ticket_summary": "crash on zero",
                    "ticket_description": 'App raises "division by zero" unexpectedly',
                    "attempts": 0,
                })
            self.assertTrue(any(f.endswith("calculator.py")
                                for f in out["candidate_files"]))


class TestPatchNode(unittest.TestCase):
    def test_diff_without_trailing_newline_still_applies(self):
        """Regression: git apply rejects a patch missing its trailing newline."""
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)
            diff = make_diff(tmp, "return a // b", "return a / b")
            fake = FakeLLM([diff.rstrip("\n")])  # model output stripped
            with patch.object(agent, "llm", fake):
                out = agent.patch({
                    "repo_path": tmp,
                    "candidate_files": [os.path.join(tmp, "calculator.py")],
                    "fix_plan": "use true division",
                    "attempts": 0,
                })
            self.assertEqual(out["status"], "patched")
            self.assertIn("return a / b", open(os.path.join(tmp, "calculator.py")).read())

    def test_code_fenced_diff_is_cleaned(self):
        """Model wraps the diff in ```diff fences — patch node must strip them."""
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)
            diff = make_diff(tmp, "return a // b", "return a / b")
            fake = FakeLLM([f"```diff\n{diff}```"])
            with patch.object(agent, "llm", fake):
                out = agent.patch({
                    "repo_path": tmp,
                    "candidate_files": [os.path.join(tmp, "calculator.py")],
                    "fix_plan": "use true division",
                    "attempts": 0,
                })
            self.assertEqual(out["status"], "patched")

    def test_garbage_diff_reports_patch_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)
            fake = FakeLLM(["this is not a diff at all"])
            with patch.object(agent, "llm", fake):
                out = agent.patch({
                    "repo_path": tmp,
                    "candidate_files": [],
                    "fix_plan": "plan",
                    "attempts": 0,
                })
            self.assertEqual(out["status"], "patch_failed")
            self.assertFalse(out["tests_passed"])
            self.assertEqual(out["attempts"], 1)


class TestRunnerDetection(unittest.TestCase):
    def test_python_repo_does_not_pick_npm(self):
        """Regression: runner must be chosen by repo markers, not installed binaries."""
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)
            os.makedirs(os.path.join(tmp, "tests"))
            with open(os.path.join(tmp, "tests", "test_x.py"), "w") as f:
                f.write("import unittest\n"
                        "class T(unittest.TestCase):\n"
                        "    def test_ok(self):\n"
                        "        self.assertTrue(True)\n")
            passed, output = agent.run_tests(tmp)
            self.assertTrue(passed, output)
            self.assertNotIn("npm", output.lower())

    def test_repo_with_no_tests_fails_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_repo(tmp)
            passed, output = agent.run_tests(tmp)
            self.assertFalse(passed)
            self.assertIn("No test runner matched", output)


if __name__ == "__main__":
    unittest.main()
