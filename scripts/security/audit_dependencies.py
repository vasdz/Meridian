"""Audit dependencies for security vulnerabilities."""

import json
import subprocess
import sys
from datetime import datetime


def run_safety_check():
    """Run safety scan on dependencies."""
    print("Running safety scan...")
    result = subprocess.run(
        ["safety", "scan", "--policy-file", ".safety-policy.yml", "--json"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("✓ No vulnerabilities found")
        return []

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Warning: Could not parse safety output")
        return []


def run_pip_audit():
    """Run pip-audit on dependencies."""
    print("Running pip-audit...")
    result = subprocess.run(
        ["pip-audit", "--format", "json"],
        capture_output=True,
        text=True,
    )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def run_bandit():
    """Run bandit security linter."""
    print("Running bandit...")
    result = subprocess.run(
        ["bandit", "-r", "src", "-f", "json", "-ll"],
        capture_output=True,
        text=True,
    )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def generate_report(safety_results, pip_audit_results, bandit_results):
    """Generate security audit report."""
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "dependency_vulnerabilities": {
            "safety": safety_results,
            "pip_audit": pip_audit_results,
        },
        "code_issues": bandit_results,
        "summary": {
            "total_dependency_issues": len(safety_results) + len(pip_audit_results),
            "total_code_issues": bandit_results.get("metrics", {}).get("_totals", {}).get("SEVERITY.HIGH", 0),
        },
    }

    return report


def main():
    """Run security audit."""
    print("=" * 60)
    print("Security Audit Report")
    print("=" * 60)
    print()

    safety_results = run_safety_check()
    pip_audit_results = run_pip_audit()
    bandit_results = run_bandit()

    report = generate_report(safety_results, pip_audit_results, bandit_results)

    print()
    print("Summary:")
    print(f"  Dependency vulnerabilities: {report['summary']['total_dependency_issues']}")
    print(f"  Code security issues (HIGH): {report['summary']['total_code_issues']}")

    # Save report
    with open("security_audit_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("Report saved to security_audit_report.json")

    # Exit with error if issues found
    if report['summary']['total_dependency_issues'] > 0 or report['summary']['total_code_issues'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
