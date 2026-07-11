# Demo run: DEMO-1 — divide() returns wrong result

A recorded end-to-end run against a small target repo (a calculator module with
unit tests) containing a planted bug: `divide()` used floor division (`a // b`),
so `divide(7, 2)` returned `3` instead of `3.5`.

> **Honesty note:** this run was executed in an offline sandbox, so the LLM calls
> were mocked with canned responses (including a deliberately wrong first patch to
> exercise the retry loop) and the graph ran on a minimal LangGraph-compatible
> runtime. What it verifies is all the deterministic machinery: fault localization,
> diff application, test execution, retry routing, the interrupt/resume gate, and
> the final commit. Run it against the real API for live model behaviour.

## The ticket (demo-ticket.json)

> **DEMO-1 — divide() returns wrong result for non-even division**
> Users report incorrect results from the calculator's divide function.
> divide(7, 2) returns 3 instead of 3.5. Likely in calculator.py.

## Execution trace

```
[graph] -> fetch_ticket        # --local-ticket mode, no Jira needed
[graph] -> localize            # tier 1 hit: calculator.py named in the ticket
[graph] -> plan
[graph] -> patch               # attempt 1: wrong fix (b // a)
[graph] -> test                # unittest: FAILED
[graph] -> plan                # retry — re-plans WITH the failure output in context
[graph] -> patch               # attempt 2: correct fix (a / b)
[graph] -> test                # unittest: 5 tests OK
[graph] -> suggest_commit      # graph pauses at interrupt()
```

## Human gate payload

Proposed commit message:

```
fix(calculator): use true division in divide()

divide() used floor division so non-even divisions returned a
truncated int (divide(7, 2) -> 3). Switch to true division so the
function returns 3.5 as the tests expect. Refs DEMO-1.
```

Proposed diff:

```diff
--- a/calculator.py
+++ b/calculator.py
@@ -16,4 +16,4 @@ def multiply(a, b):
 def divide(a, b):
     if b == 0:
         raise ValueError("division by zero")
-    return a // b
+    return a / b
```

Resumed with `Command(resume="yes")` → commit created, working tree clean,
all tests passing. Final state: `committed`, `attempts: 2`.

## Bugs this run caught in the agent itself

1. **Trailing-newline patch corruption.** `diff.strip()` removed the final
   newline and `git apply` reading from stdin rejects such a patch
   ("corrupt patch at line N") — every attempt failed silently at apply.
   Fixed by re-appending the newline after cleanup.
2. **Test-runner detection by installed binaries.** The agent picked `npm test`
   on a Python repo simply because npm existed on the machine. Detection is now
   driven by repo markers (`tests/`, `package.json`, `*.csproj`/`*.sln`).

Both are exactly the class of failure you only find by running the pipeline,
which is why this demo exists.
