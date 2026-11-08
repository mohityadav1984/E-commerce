"""Runs every test and demo under MOCK_LLM and saves transcripts/. No API keys needed.

Usage (from the project folder, with your venv active):
    python run_all.py
"""
from __future__ import annotations

import codecs
import datetime
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRANSCRIPTS = ROOT / "transcripts"
LOGS = ROOT / "logs"

ENV = {
    **os.environ,
    "MOCK_LLM": "true",
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "PYTHONUNBUFFERED": "1",        # stream output live, don't buffer it
    "PYTHONIOENCODING": "utf-8",    # avoid cp1252 errors on Windows consoles
}

STEPS = [
    "01_dataset",
    "02_calibrate",        # writes calibration.json (needed by everything below)
    "03_chunking_eval",    # writes transcripts/chunking_eval.json
    "04_grounded_demo",
    "05_crew_demo",
    "06_memory_demo",
    "07_guardrails_demo",
    "08_api_demo",
    "09_eval",
    "10_review_demo",
    "11_governance_demo",
    "12_cache_demo",
]


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def run_step(name: str, argv: list[str], transcript: Path) -> bool:
    print(f"\n=== {now()} >>> {name}", flush=True)
    if name == "02_calibrate":
        print("    (first run downloads the embedding model and loads PyTorch: this can take several minutes)",
              flush=True)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    proc = subprocess.Popen(argv, cwd=ROOT, env=ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    try:
        with open(transcript, "w", encoding="utf-8") as out:
            # Read raw chunks, not lines: progress bars use \r and never print a newline.
            while True:
                chunk = os.read(proc.stdout.fileno(), 1024)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                sys.stdout.write(text)
                sys.stdout.flush()
                out.write(text)
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        print(f"\n!!! interrupted during {name}", flush=True)
        raise
    if proc.returncode != 0:
        print(f"!!! {name} FAILED (exit {proc.returncode}) - see {transcript.relative_to(ROOT)}",
              flush=True)
        return False
    return True


def main() -> int:
    TRANSCRIPTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    (LOGS / "requests.jsonl").write_text("", encoding="utf-8")
    print(f"[preflight] python: {sys.executable}", flush=True)

    failed = []
    if not run_step("tests", [sys.executable, "-u", "-m", "pytest", "-q", "tests"],
                    TRANSCRIPTS / "00_tests.txt"):
        failed.append("tests")
    for step in STEPS:
        if not run_step(step, [sys.executable, "-u", str(ROOT / "scripts" / f"{step}.py")],
                        TRANSCRIPTS / f"{step}.txt"):
            failed.append(step)

    print(f"\nDone. Failed steps: {', '.join(failed) if failed else 'none'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
