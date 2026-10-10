"""Run every check in testing/checks with one command.

    python testing/run_checks.py            # run all
    python testing/run_checks.py source     # only scripts with 'source' in the name
    python testing/run_checks.py -v         # print output of passing scripts too

Each script is a standalone program that prints its own result and exits non-zero
on failure. They were written one at a time as bugs were fixed, so each is tied
to a defect that actually happened rather than to a unit of code — which is why
they live here instead of inside a conventional test framework.

A script needing data that is not in the repo (logs/ and testing/samples/ are both
gitignored because they hold customer files) is reported as SKIPPED, not failed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKS = Path(__file__).resolve().parent / "checks"

# Missing input data is not a failure — but it has to be visible, or a fresh
# machine reports "all passed" having actually run nothing.
MISSING_DATA_MARKERS = (
    "FileNotFoundError",
    "No such file or directory",
)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    pattern = args[0] if args else ""

    scripts = sorted(p for p in CHECKS.glob("verify_*.py") if pattern in p.name)
    if not scripts:
        print(f"No script matching {pattern!r} in {CHECKS}")
        return 1

    passed, failed, skipped = [], [], []
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        name = script.stem
        if result.returncode == 0:
            passed.append(name)
            print(f"✅ {name}")
            if verbose:
                print("\n".join("     " + line for line in output.splitlines()))
        elif any(marker in output for marker in MISSING_DATA_MARKERS):
            skipped.append(name)
            print(f"⏭️  {name} — thiếu dữ liệu đầu vào (không tính là hỏng)")
        else:
            failed.append(name)
            print(f"❌ {name}")
            print("\n".join("     " + line for line in output.splitlines()[-25:]))

    print()
    print(f"đạt {len(passed)} | hỏng {len(failed)} | bỏ qua {len(skipped)}")
    if failed:
        print("hỏng:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
