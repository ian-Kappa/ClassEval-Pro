"""Central configuration: paths, thresholds, ratios, API/model settings.

Secrets are read from environment variables, never hardcoded:
  GITHUB_TOKEN, AZURE_API_KEY, AZURE_ENDPOINT
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

# --- Per-stage artifact files (each stage reads prev, writes its own) ---
REPOS_FILE = ARTIFACTS / "01_repos.json"          # stage1 -> repo list
STRUCTURED_FILE = ARTIFACTS / "02_structured.json"  # stage2 -> structured classes
CLASSES_FILE = ARTIFACTS / "03_classes.json"      # stage3 -> deduped + ids
PAIRS_FILE = ARTIFACTS / "04_pairs.json"          # stage4 -> skeleton1/2 pairs
DATASET_FILE = ARTIFACTS / "dataset.json"         # stage5 -> final benchmark

# ============================================================================
# Stage 1: GitHub mining (paper Sec. 3.2)
# ============================================================================
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
START_DATE = "2025-01-01"      # post-cutoff repos only (contamination control)

# 11-domain keyword matrix
DOMAIN_KEYWORDS = {
    "1_Algo_DataStruct": ["algorithm", "data-structures", "leetcode",
        "competitive-programming", "sorting", "tree", "graph-theory",
        "dynamic-programming", "backtracking"],
    "2_Math_Stats": ["mathematics", "statistics", "numerical-analysis",
        "linear-algebra", "calculus", "probability", "scipy", "numpy",
        "sympy", "statsmodels"],
    "3_Text_NLP": ["nlp", "natural-language-processing", "text-mining",
        "sentiment-analysis", "tokenization", "text-processing", "spacy",
        "nltk", "transformers", "bert"],
    "4_File_Data_Docs": ["json", "csv", "xml", "yaml", "pdf", "excel",
        "data-processing", "file-parsing", "etl", "data-pipeline",
        "pandas", "openpyxl"],
    "5_Network_Web": ["web-scraping", "crawler", "requests", "beautifulsoup",
        "selenium", "http-client", "websocket", "api-client", "proxy",
        "asyncio", "aiohttp"],
    "6_DB_SQL": ["database", "sql", "orm", "sqlalchemy", "postgresql",
        "mysql", "mongodb", "redis", "sqlite", "nosql", "query-builder"],
    "7_Business_App": ["business-logic", "workflow", "task-management",
        "scheduling", "booking-system", "inventory", "crm", "erp",
        "project-management"],
    "8_Finance_Ecommerce": ["finance", "trading", "stock-market",
        "cryptocurrency", "blockchain", "payment", "e-commerce",
        "shopping-cart", "fintech", "quantitative"],
    "9_Game_Sim": ["game-development", "pygame", "simulation", "2d-game",
        "game-engine", "physics-engine", "rendering", "ai-game", "roguelike"],
    "10_Security_Crypto": ["cryptography", "encryption", "security",
        "authentication", "hashing", "ssl", "jwt", "oauth", "password",
        "aes", "rsa"],
    "11_Utils_Tools": ["utilities", "cli", "command-line", "logging",
        "config", "file-utils", "string-utils", "datetime", "parser",
        "validator"],
}

# Stratified star-tier sampling (paper: tiers akin to The Stack)
SEARCH_STRATEGIES = [
    {"stars_range": "100..500", "per_page": 10},
    {"stars_range": "500..1000", "per_page": 8},
    {"stars_range": ">1000", "per_page": 5},
]
KEYWORD_CHUNK = 3  # keywords per OR-query (avoids GitHub query complexity 422)

# ============================================================================
# Stage 2: extract / validate / structure (paper Sec. 3.2)
# ============================================================================
MIN_METHODS = 5      # structural constraint: >= 5 methods
MIN_LOC = 40         # 40..800 lines of code
MAX_LOC = 800
CLONE_DEPTH = 1      # shallow clone
SKIP_TEST_FILES = True
CLONE_TIMEOUT = 60   # seconds per repo

# ============================================================================
# Stage 3: The Stack-style content deduplication (paper Sec. 3.2)
# ============================================================================
DEDUP_NEAR_JACCARD = 0.85   # drop near-duplicates above this MinHash-Jaccard
MINHASH_PERM = 64           # number of MinHash permutations
SHINGLE_K = 5               # token n-gram size for shingling

# ============================================================================
# Stage 4: intra/cross-domain composition (paper Sec. 3.2, Table 2)
# ============================================================================
TOTAL_TASKS = 300
INTRA_DOMAIN_TASKS = 67          # same-domain; remaining 233 are cross-domain
MAX_CLASS_REUSE = 8              # cap reuse of a source class across pairs
SEED = 42                        # deterministic pairing
# Expert-curated composable-domain matrix hook (paper: "experts first
# categorize composable domains"). None = allow any distinct-domain pair.
# Else a set of unordered domain pairs, e.g. {("6_DB_SQL", "9_Game_Sim")}.
COMPOSABLE_DOMAIN_PAIRS = None

# ============================================================================
# Stage 5: combine + test + judge + solution + coverage (paper Sec. 3.3-3.4)
# ============================================================================
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "YOUR_API_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "YOUR_API_ENDPOINT")
AZURE_API_VERSION = "2024-03-01-preview"

# Model identities are not hardcoded. Provide deployment names via env vars
# (paper uses a strong frontier model for fusion/test/solution and three
# independent judges; substitute your own deployments here).
SKELETON_MODEL = os.environ.get("SKELETON_MODEL", "YOUR_SKELETON_MODEL")
TESTCASE_MODEL = os.environ.get("TESTCASE_MODEL", "YOUR_TESTCASE_MODEL")
SOLUTION_MODEL = os.environ.get("SOLUTION_MODEL", "YOUR_SOLUTION_MODEL")
JUDGE_MODELS = [                                # 3 independent LLM judges
    os.environ.get("JUDGE_MODEL_1", "YOUR_JUDGE_MODEL_1"),
    os.environ.get("JUDGE_MODEL_2", "YOUR_JUDGE_MODEL_2"),
    os.environ.get("JUDGE_MODEL_3", "YOUR_JUDGE_MODEL_3"),
]
JUDGE_PASS_MIN = 2          # advance if >= 2 of 3 judges give full marks (10)
COVERAGE_MIN = 90.0         # keep tasks whose reference solution covers > 90%
GEN_MAX_WORKERS = 10
GEN_MAX_TOKENS = 16384
GEN_TIMEOUT = 60            # seconds per coverage run
GEN_RETRIES = 1
