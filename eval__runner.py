"""
eval/runner.py — Layer 5: Eval runner.

Run with:  python eval/runner.py

Runs every case in DATASET against the live agent and reports:
  - Which cases passed / failed
  - Which assertions failed and why
  - Latency per case
  - CI exit code (0 = all pass, 1 = any failure)

Wire this into CI so every PR runs it. Gate is: no regression vs main.
"""
import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Add parent to path so we can import from the project root
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import root_agent
from config import APP_NAME
from eval.dataset import DATASET, EvalCase


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str]
    final_output: str
    responding_agent: str
    latency_ms: int


def run_case(runner: Runner, case: EvalCase, user_id: str) -> CaseResult:
    """Run a single eval case and return the result."""
    session_id = str(uuid.uuid4())[:8]
    session_service = runner.session_service
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={"session_id": session_id},
    )

    content = types.Content(role="user", parts=[types.Part(text=case.user_input)])
    failures = []
    final_output = ""
    responding_agent = "unknown"

    start = time.monotonic()
    try:
        events = runner.run(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
        )
        for event in events:
            if event.is_final_response():
                final_output = event.content.parts[0].text
                responding_agent = getattr(event, "author", "unknown")
                break
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id,
            passed=False,
            failures=[f"Agent raised exception: {exc}"],
            final_output="",
            responding_agent="error",
            latency_ms=0,
        )
    latency_ms = int((time.monotonic() - start) * 1000)

    # ── Programmatic assertions ───────────────────────────────────────────────
    out_lower = final_output.lower()

    for required in case.must_contain:
        if required.lower() not in out_lower:
            failures.append(f"missing required text: {required!r}")

    for forbidden in case.must_not_contain:
        if forbidden.lower() in out_lower:
            failures.append(f"contains forbidden text: {forbidden!r}")

    if case.expected_agent and responding_agent != case.expected_agent:
        failures.append(
            f"wrong agent: expected={case.expected_agent}, got={responding_agent}"
        )

    return CaseResult(
        case_id=case.case_id,
        passed=len(failures) == 0,
        failures=failures,
        final_output=final_output,
        responding_agent=responding_agent,
        latency_ms=latency_ms,
    )


def run_all() -> list[CaseResult]:
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    user_id = "eval_runner"
    results = []
    for case in DATASET.cases:
        print(f"  running {case.case_id}...", end="", flush=True)
        result = run_case(runner, case, user_id)
        status = "✓" if result.passed else "✗"
        print(f" {status}  ({result.latency_ms}ms)")
        results.append(result)
    return results


def print_report(results: list[CaseResult]) -> int:
    """Print summary and return exit code."""
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"\n{'═' * 68}")
    print(f"  Eval: {passed}/{len(results)} passed  ·  dataset={DATASET.version}")
    print(f"{'═' * 68}")

    for r in results:
        if not r.passed:
            print(f"\n  ✗ {r.case_id}")
            for f in r.failures:
                print(f"      {f}")
            print(f"      agent={r.responding_agent}  latency={r.latency_ms}ms")
            print(f"      output: {r.final_output[:200]!r}")

    if failed == 0:
        print(f"\n  All {passed} cases passed.\n")
    else:
        print(f"\n  {failed} case(s) failed. See above.\n")

    # Write machine-readable report for CI
    report = {
        "dataset_version": DATASET.version,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "cases": [
            {
                "case_id": r.case_id,
                "passed": r.passed,
                "failures": r.failures,
                "latency_ms": r.latency_ms,
                "agent": r.responding_agent,
            }
            for r in results
        ],
    }
    with open("eval/last_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report written to eval/last_report.json")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    print(f"\n  Running eval suite (dataset version={DATASET.version})\n")
    results = run_all()
    exit_code = print_report(results)
    sys.exit(exit_code)
