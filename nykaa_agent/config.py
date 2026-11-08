"""Central config. IMPORT THIS FIRST (before crewai / chromadb) so the env vars
below are set before those libraries read them."""
import json
import os
from pathlib import Path

# --- zero-network / zero-key defaults -------------------------------------
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")   # required by brief
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")      # chromadb telemetry
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MOCK_LLM", "true")

MOCK_LLM = os.environ["MOCK_LLM"].strip().lower() != "false"

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "kb"
CHROMA_DIR = ROOT / ".chroma"
LOG_DIR = ROOT / "logs"
TRANSCRIPT_DIR = ROOT / "transcripts"
REQUEST_LOG = LOG_DIR / "requests.jsonl"
CALIBRATION_FILE = ROOT / "calibration.json"
CHUNKING_RESULT_FILE = TRANSCRIPT_DIR / "chunking_eval.json"

EMBED_MODEL = "all-MiniLM-L6-v2"   # free, local SentenceTransformers model
FIXED_SIZE = 300                   # characters per fixed chunk
FIXED_OVERLAP = 60                 # characters of overlap
TOP_K = 3
COLLECTIONS = {"fixed": "nykaa_kb_fixed", "sentence": "nykaa_kb_sentence"}

MAX_REQUEST_TOKENS = int(os.environ.get("NYKAA_MAX_REQUEST_TOKENS", "2000"))


def active_strategy() -> str:
    """Strategy used by the agent: env override > chunking_eval result > 'sentence'."""
    env = os.environ.get("NYKAA_STRATEGY")
    if env in COLLECTIONS:
        return env
    if CHUNKING_RESULT_FILE.exists():
        try:
            rec = json.loads(CHUNKING_RESULT_FILE.read_text()).get("recommended")
            if rec in COLLECTIONS:
                return rec
        except (OSError, ValueError):
            pass
    return "sentence"


def load_threshold(strategy: str) -> float:
    """Empirically calibrated 'I don't know' threshold (see scripts/02_calibrate.py)."""
    if not CALIBRATION_FILE.exists():
        raise RuntimeError(
            "calibration.json missing. Run `python scripts/02_calibrate.py` first - "
            "the brief forbids an untested preset threshold."
        )
    return float(json.loads(CALIBRATION_FILE.read_text())[strategy]["threshold"])
