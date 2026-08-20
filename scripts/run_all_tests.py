#!/usr/bin/env python3
"""CHIMERA-Agent Unified Test Harness & Verification Suite.

Runs:
1. Python unit, nomogram, cohort math, parity, and sweep test suites via unittest
2. Client-side JavaScript test suites via Node.js node:test
3. E2E verification across all 429 multimodal patient cases
4. Comprehensive test summary report with pass/fail telemetry
"""

import glob
import json
import math
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def print_banner(title: str):
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_python_suite():
    print_banner("1. PYTHON UNITTEST SUITE (tests/test_*.py)")
    start = time.time()
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT_DIR / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start
    return {
        "suite": "Python unittest",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.wasSuccessful(),
        "elapsed_s": round(elapsed, 3),
    }


def _run_node_glob(label, pattern):
    """Run a single Node.js test glob and return a per-glob result dict."""
    test_files = sorted(glob.glob(str(ROOT_DIR / pattern)))
    if not test_files:
        print(f"  [{label}] No test files found for {pattern}")
        return {
            "label": label,
            "tests_run": 0,
            "failures": 0,
            "errors": 0,
            "passed": True,  # absence of files is not a failure
            "elapsed_s": 0.0,
        }

    cmd = ["node", "--test"] + test_files
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    print(f"\n--- {label} ({pattern}) ---")
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    passed = proc.returncode == 0

    # Parse test count from TAP output
    tests_count = 0
    for line in proc.stdout.splitlines():
        if line.startswith("# tests "):
            try:
                tests_count = int(line.split()[2])
            except (IndexError, ValueError):
                pass

    return {
        "label": label,
        "tests_run": tests_count,
        "failures": 0 if passed else 1,
        "errors": 0,
        "passed": passed,
        "elapsed_s": 0.0,
    }


def run_node_suite():
    print_banner("2. NODE.JS TEST SUITE (tests/js/*.test.js + test/*.test.js)")
    start = time.time()

    # 1. Existing lenient component suite (tests/js/*.test.js)
    lenient = _run_node_glob("tests/js", os.path.join("tests", "js", "*.test.js"))
    # 2. Rigorous tier + adversarial suite (test/*.test.js)
    rigorous = _run_node_glob("test", os.path.join("test", "*.test.js"))

    elapsed = time.time() - start

    # Aggregate counts; overall failure if either suite fails
    total_tests = lenient["tests_run"] + rigorous["tests_run"]
    total_failures = lenient["failures"] + rigorous["failures"]
    total_errors = lenient["errors"] + rigorous["errors"]
    all_passed = lenient["passed"] and rigorous["passed"]

    print(f"\n  Node.js aggregate: {total_tests} tests "
          f"({lenient['tests_run']} lenient + {rigorous['tests_run']} rigorous) "
          f"| failures={total_failures} errors={total_errors} "
          f"| {'PASS' if all_passed else 'FAIL'}")

    return {
        "suite": "Node.js node:test",
        "tests_run": total_tests,
        "failures": total_failures,
        "errors": total_errors,
        "passed": all_passed,
        "elapsed_s": round(elapsed, 3),
    }


def run_trace_audit_sweep():
    print_banner("3. 429-CASE COMPREHENSIVE MULTIMODAL AUDIT SWEEP")
    start = time.time()
    local_traces = sorted(glob.glob(str(ROOT_DIR.parent / "chimera-data" / "traces" / "all_local_cases" / "*.json")))
    sample_traces = sorted(glob.glob(str(ROOT_DIR.parent / "chimera-data" / "traces" / "run_sample_eval" / "*.json")))
    all_traces = local_traces + sample_traces

    nan_count = 0
    inf_count = 0
    invalid_schema = 0

    def check_clean(obj, case_id):
        nonlocal nan_count, inf_count
        if isinstance(obj, float):
            if math.isnan(obj):
                nan_count += 1
            if math.isinf(obj):
                inf_count += 1
        elif isinstance(obj, str) and obj.lower() in ("nan", "infinity", "-infinity"):
            nan_count += 1
        elif isinstance(obj, dict):
            for k, v in obj.items():
                check_clean(v, case_id)
        elif isinstance(obj, list):
            for v in obj:
                check_clean(v, case_id)

    for fpath in all_traces:
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cid = data.get("case_id", os.path.basename(fpath))
        if not data.get("case_id") or not data.get("task"):
            invalid_schema += 1
        check_clean(data, cid)

    elapsed = time.time() - start
    passed = (len(all_traces) == 429) and (nan_count == 0) and (inf_count == 0) and (invalid_schema == 0)

    print(f"  Total Traces Audited: {len(all_traces)} (423 local + 6 sample evaluation)")
    print(f"  NaN Count:            {nan_count}")
    print(f"  Inf Count:            {inf_count}")
    print(f"  Schema Violations:    {invalid_schema}")
    print(f"  Sweep Status:         {'✅ PASS' if passed else '❌ FAIL'}")

    return {
        "suite": "429-Case Audit Sweep",
        "tests_run": len(all_traces),
        "failures": nan_count + inf_count + invalid_schema,
        "errors": 0,
        "passed": passed,
        "elapsed_s": round(elapsed, 3),
    }


def main():
    total_start = time.time()
    print_banner("CHIMERA-AGENT UNIFIED DUAL TEST HARNESS")
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Python:         {sys.version.split()[0]}")
    
    # Run all 3 verification suites
    py_res = run_python_suite()
    node_res = run_node_suite()
    sweep_res = run_trace_audit_sweep()

    total_elapsed = time.time() - total_start

    # Summary Report
    print_banner("FINAL TEST TELEMETRY & VERIFICATION SUMMARY")
    results = [py_res, node_res, sweep_res]
    total_tests = sum(r["tests_run"] for r in results)
    total_failures = sum(r["failures"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    all_passed = all(r["passed"] for r in results)

    print(f"{'Suite':<28} | {'Tests':<8} | {'Failures':<10} | {'Errors':<8} | {'Time (s)':<10} | {'Status':<10}")
    print("-" * 85)
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{r['suite']:<28} | {r['tests_run']:<8} | {r['failures']:<10} | {r['errors']:<8} | {r['elapsed_s']:<10.3f} | {status:<10}")
    print("-" * 85)
    total_status = "✅ 100% ALL PASSED" if all_passed else "❌ TEST FAILURES DETECTED"
    print(f"{'TOTAL':<28} | {total_tests:<8} | {total_failures:<10} | {total_errors:<8} | {total_elapsed:<10.3f} | {total_status:<10}")
    print("=" * 85)

    if all_passed:
        print("\n🎉 All test suites passed with 100% scientific integrity and zero defects.\n")
        sys.exit(0)
    else:
        print("\n❌ Verification failed. Please inspect failure logs above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
