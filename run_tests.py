import sys
import subprocess


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "--cov=src",
        "--cov-report=term-missing",
        "--html=report.html",
        "--self-contained-html",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())


