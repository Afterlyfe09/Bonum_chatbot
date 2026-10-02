"""Central configuration: paths, URLs and tunable settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# --- Website -----------------------------------------------------------------
BASE_URL = "https://bonum.in"
HOME_URL = f"{BASE_URL}/index.php"
CONTACT_URL = f"{BASE_URL}/contact.php"
LOGO_URL = f"{BASE_URL}/images/icons/bonumlogopurple.png"

# Pages linked from the site navigation/footer. The crawler also follows internal
# links found on these pages (same domain only, .php pages only).
SEED_PATHS = [
    "index.php", "about.php", "services.php", "webdev.php", "appdev.php",
    "software.php", "data.php", "digital.php", "blogs.php", "faq.php",
    "contact.php", "refund.php", "terms.php", "privacy.php", "sitemap.php",
]
MAX_PAGES = 60
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 20

# --- Storage -----------------------------------------------------------------
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
VECTOR_DIR = ROOT / "vectorstore"
INDEX_PATH = VECTOR_DIR / "index.faiss"
META_PATH = VECTOR_DIR / "metadata.json"

# --- Models ------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
LLM_TIMEOUT_SECONDS = 30

# --- Retrieval / chat --------------------------------------------------------
TOP_K = _int_env("TOP_K", 6)
RELEVANCE_THRESHOLD = _float_env("RELEVANCE_THRESHOLD", 0.30)
MAX_CONTEXT_CHARS = 10000
HISTORY_MESSAGES = 8
MAX_QUESTION_CHARS = 1000


def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()