"""Stage 5 - skeleton fusion, test generation, judging, solution & coverage.

For each (skeleton1, skeleton2) pair (paper Sec. 3.3-3.4):
  1. fuse the two skeletons into one merged class spec (SKELETON_MODEL)
  2. generate a unittest suite (TESTCASE_MODEL)
  3. score skeleton/test compatibility with 3 LLM judges; keep pairs
     scoring full marks (10) from >= JUDGE_PASS_MIN judges
  4. generate a reference solution (SOLUTION_MODEL)
  5. measure pytest line coverage; keep tasks with coverage > COVERAGE_MIN
Output: config.DATASET_FILE.

NOTE: the four prompt blocks below are the original full prompts from the
paper's construction pipeline and are reproduced verbatim. Only the *_USER
templates are ever passed through str.format(); the *_SYS system prompts are
sent raw (they may contain literal braces) -- do not .format() them.
"""

import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

from . import config

# ---------------------------------------------------------------------------
# Prompts (verbatim from the paper's construction pipeline; do not alter)
# ---------------------------------------------------------------------------
SKELETON_SYS = '''
# Role:
You are an expert **Cross-Domain Systems Architect** specializing in creating
**tightly-coupled hybrid systems** with rich dependency relationships within a
**single unified class**.

# Goal
Given **TWO** input skeletons from **different domains**, output **ONE merged skeleton**
containing **EXACTLY ONE class** that:
- Establishes **deep functional dependencies** where methods rely on each other's
  state and behavior
- Creates a **unified abstraction** with explicit internal coupling mechanisms
- Implements **bidirectional interactions** and **dependency chains** between domain
  capabilities
- Contains fields representing state from **both domains** (never empty fields)
- Remains executable as a precise implementation specification

# Critical Constraint:
**OUTPUT EXACTLY ONE CLASS DEFINITION ONLY**
- NO separate helper classes, interfaces, or auxiliary objects
- All functionality must be self-contained within the single hybrid class
- Dependencies expressed through:
  - Internal method calls between methods of the same class
  - Field-level state sharing within the same class
  - Method parameter passing within the same class

# Workflow:

## 1. Analyze Dependencies
- Identify potential state dependencies between domain capabilities
- Map data flow and control flow intersections within a unified context
- Determine internal coupling points (shared fields, method chains, state transitions)

## 2. Synthesize Single Hybrid Class
- **Name**: Create composite class name reflecting integration
  - Example: "MusicPlayer" + "StockMarket" → "SonifiedMarketStream"
- **Fields**: Compose fields representing BOTH domains
  - Shared state modified/read by multiple methods
  - Domain-specific state that triggers cross-domain behavior
  - Configuration flags controlling inter-domain coupling
- **Methods**: Design methods showcasing internal dependencies
  - Methods requiring state from BOTH domain aspects
  - Methods that call other methods in dependency chains
  - Transformation methods where one domain's output feeds another's input
  - Coordination methods managing internal state consistency

## 3. Dependency Patterns (within single class)
- **State Coupling**: Methods reading/writing shared fields
- **Method Chaining**: Method A calls Method B which calls Method C
- **Event-like Patterns**: Methods triggering state changes that other methods react to
- **Lifecycle Dependencies**: Methods that must be called in specific order

## 4. Complexity Upgrades (document in Implementation Notes)
- **Method dependency graph**: Which methods call which others
- **State transition flows**: How fields change across method calls
- **Initialization order**: Constructor logic and field setup sequences
- **Coupling mechanisms**: How the two domains interact internally
- **Error propagation**: How failures in one domain affect the other

# Output Format:
Please format your response exactly as follows:
# Class Name
`ClassName`

# Purpose
[1-2 sentences describing what this class does and why it exists]

# Fields
- `field_name` (type): [brief description]
- [If no fields: "No instance fields"]

# Methods

## `method_name(param1, param2)`
- **Purpose**: [what this method does]
- **Parameters**: `param1` (type) - [description]; `param2` (type) - [description]
- **Returns**: [type and description]
- **Example**:
```python
obj = ClassName()
result = obj.method_name(arg1, arg2)  # Returns: expected_output
```

## `method_name_2(...)`
[Repeat structure]

# Implementation Notes
[Optional: Key algorithms, design patterns, or important technical details]
'''

SKELETON_USER = '''
#Input skeleton1
{skeleton1}

#Input skeleton2
{skeleton2}
'''

TESTCASE_SYS = '''
## Role
You are a Senior Automation Architect specializing in implementation-faithful test design.
Your responsibility is to generate a **production-grade unittest suite that is strictly and provably aligned with the concrete implementation implied by the provided skeleton**.

## Core Principle (Non-Negotiable)
⚠️ All test cases MUST be derived from and executable against a **literal, minimal, and conservative implementation** of the given skeleton.

You are explicitly FORBIDDEN from:
- Testing behavior not directly implied by the skeleton
- Assuming "reasonable" future logic, best practices, or ideal designs
- Introducing expectations that cannot be satisfied by a straightforward implementation of the skeleton

If a behavior is not explicitly inferable from:
- method signatures
- docstrings
- parameter names
- control flow hints
- state variables
then it MUST NOT be tested.

## Goal
Analyze the skeleton at the **implementation level**, not the conceptual level.

Your objective is to:
- Extract all **observable behaviors** implied by the skeleton
- Identify **every executable branch, state transition, and failure mode**
- Design a unittest suite that achieves **maximum feasible coverage of those behaviors**
- Ensure that a faithful implementation of the skeleton can pass **all tests without hacks or test-oriented code**

If a test would force an implementation to be written "just to satisfy the test", that test MUST be removed.

## Test Design Constraints
1. **Implementation-Exact Matching**
- Every assertion must map to a specific, traceable behavior in the skeleton
- Prefer testing:
- return values
- raised exceptions
- state mutations
- idempotency
- default initialization effects
- Avoid testing abstract intent or high-level semantics unless explicitly stated

2. **Conservative Interpretation Rule**
- When the skeleton is ambiguous, choose the **weakest valid interpretation**
- Never "strengthen" behavior via tests
- Absence of specification = absence of test

3. **Coverage Strategy**
- Coverage must come from:
- control-flow paths
- parameter variations
- boundary values
- stateful interactions
- NOT from speculative feature completeness

4. **Dynamic Test Class Selection**
- Treat the Reference Template as a strict capability menu
- Include a test class ONLY IF the skeleton contains:
- explicit signals (methods, fields, docstrings) requiring it
- If a class is included, it must test **all relevant branches** of that concern

Examples:
- No numerical computation → do NOT include Test07_AnalyticalComputation
- No error semantics defined → Test06 must only verify absence or minimal raising
- No special tokens mentioned → skip Test05 entirely

5. **Docstring Fidelity**
- Test expectations must not exceed what is stated in skeleton docstrings
- If docstrings are vague, tests must be minimal and permissive

## Implementation Standards
- **Self-Contained**: No external dependencies; mock only if the skeleton implies them
- **Deterministic**: No randomness unless explicitly present in the skeleton
- **Traceable**: Each test must clearly correspond to a skeleton element

Each test method MUST include a docstring in the following exact format:
"""Target: <method_or_state> | Input: <concrete_input> | Expected: <explicit_observable_result>"""

## Output Rules (Strict)
- Output ONLY a single valid Python code block
- No explanations, no markdown, no comments outside code
- The test suite must be executable as-is

## Required Structural Template
You MUST follow this structure exactly:
```python
import unittest

class Test01_BasicFunctionality(unittest.TestCase):
"Core logic, standard inputs, deterministic outputs."

class Test02_Initialization(unittest.TestCase):
"Constructor behavior, default values, post-init observable state."

class Test03_ConfigurationAndState(unittest.TestCase):
"State transitions and configuration changes explicitly implied."

class Test04_InputValidation(unittest.TestCase):
"Type checks and invalid inputs only if enforced by the skeleton."

class Test05_SpecialTokenHandling(unittest.TestCase):
"Reserved tokens or control markers ONLY if defined."

class Test06_ErrorHandling(unittest.TestCase):
"Exceptions ONLY if explicitly raised or documented."

class Test07_AnalyticalComputation(unittest.TestCase):
"Numerical logic ONLY if present."

class Test08_BoundaryAndEdgeCases(unittest.TestCase):
"Edge cases derivable from data structures or loops."

class Test09_PerformanceBehavior(unittest.TestCase):
"Caching, batching, or branching ONLY if observable."

class Test10_IntegrationBehavior(unittest.TestCase):
"Cross-method interactions ONLY if methods are composable."

if __name__ == '__main__':
unittest.main()
```
'''

TESTCASE_USER = '''
# Input Data
{skeleton}
'''

JUDGE_SYS = '''
# Role
You are a senior QA Engineer.

# Task
Evaluate the actual compatibility between a Class Skeleton (class definition/implementation) and its corresponding Test Cases.

# Core Objective
Determine whether the Test Cases' assertions can realistically pass if the Skeleton is filled with minimal logic.

# Evaluation Criteria (Strictly Enforced)
* **Starting Score:** 10 points
* **Deduct 1 point for each of the following defects:**
  * **API Incompatibility:** Method name, parameter count, parameter type, or return type called in Test Cases does not match the Skeleton.
  * **Missing Internal State:** Side effects expected by Test Cases (e.g., property modification, counter increment) lack corresponding attribute support in the Skeleton.
  * **Constructor Mismatch:** Class initialization approach (parameters) in Test Cases is inconsistent with Skeleton definition.
  * **Logic Assertion Failure:** Even with matching signatures, the Skeleton structure cannot support the assertion logic in tests (e.g., test expects a List return, but Skeleton defines String return).
  * **Missing Exception Handling:** Test Cases expect specific exceptions to be thrown, but the Skeleton signature does not reflect this or the structure does not support it.

# Review Process (Mental Execution)
1. **Scan Test Cases line by line:** Extract all calls to the target class.
2. **Symbol Alignment:** Locate corresponding definitions in the Skeleton.
3. **Causality Verification:** Verify whether the `assert` logic in Test Cases can hold under the Skeleton structure.

# Output Format
Must return the following JSON format:
```json
{
  "score": <number between 0-10>
}
```
'''

JUDGE_USER = '''
**Class Skeleton:**
```
{skeleton}
```

**Test Cases:**
```
{test}
```
Respond only with the JSON object, no additional text.
'''

SOLUTION_SYS = '''
# Role
You are a code expert, Given the following code skeleton and test cases, generate a complete, working solution code in Python.

# Goal
Requirements:
- Implement all missing function bodies
- Ensure all test cases pass
- Write clean, efficient code
- Include necessary imports

# Output Format
Only output the full solution code wrapped in ```python ``` markers, dont't output test case, just solution code.
'''

SOLUTION_USER = '''
# Skeleton
{skeleton}

#Testcase
{test}
'''


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
_client = openai.AzureOpenAI(
    api_key=config.AZURE_API_KEY,
    azure_endpoint=config.AZURE_ENDPOINT,
    api_version=config.AZURE_API_VERSION,
)


def _ask(system: str, user: str, model: str, temperature: float) -> str:
    for attempt in range(config.GEN_RETRIES + 1):
        try:
            r = _client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=config.GEN_MAX_TOKENS, temperature=temperature)
            return (r.choices[0].message.content or "").strip()
        except Exception:
            if attempt == config.GEN_RETRIES:
                raise
    return ""


def _code_block(text: str) -> str:
    m = re.search(r"```python\s*([\s\S]*?)\s*```", text)
    return m.group(1).strip() if m else text.strip()


def _judge_score(text: str):
    t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(t.strip()).get("score")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Coverage measurement
# ---------------------------------------------------------------------------
def _line_coverage(solution: str, test: str) -> float | None:
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        (ws / "solution.py").write_text(_code_block(solution), encoding="utf-8")
        tests = ws / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        (tests / "test_solution.py").write_text(
            "import sys, os\n"
            "sys.path.insert(0, os.path.dirname(os.path.dirname("
            "os.path.abspath(__file__))))\nfrom solution import *\n\n"
            + _code_block(test), encoding="utf-8")
        cov = ws / "coverage.json"
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", str(tests),
                 f"--cov={ws}", "--cov-report", f"json:{cov}", "-q"],
                capture_output=True, timeout=config.GEN_TIMEOUT, cwd=ws)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or not cov.exists():   # not 100% pass -> reject
            return None
        data = json.loads(cov.read_text())
        for fp, fd in data.get("files", {}).items():
            if "solution.py" in fp and "test" not in fp.lower():
                return round(fd["summary"]["percent_covered"], 2)
        return data.get("totals", {}).get("percent_covered")


# ---------------------------------------------------------------------------
# Per-task pipeline
# ---------------------------------------------------------------------------
def _process(item: dict) -> dict | None:
    # 1. fuse skeletons
    skeleton = _ask(SKELETON_SYS,
                    SKELETON_USER.format(skeleton1=item["skeleton1"],
                                         skeleton2=item["skeleton2"]),
                    config.SKELETON_MODEL, 0.8)
    m = re.search(r"`([^`]+)`", skeleton)
    class_name = m.group(1) if m else "UnknownClass"

    # 2. generate tests
    test = _ask(TESTCASE_SYS, TESTCASE_USER.format(skeleton=skeleton),
                config.TESTCASE_MODEL, 0.2)

    # 3. three-judge filter
    scores = []
    for jm in config.JUDGE_MODELS:
        try:
            scores.append(_judge_score(_ask(
                JUDGE_SYS, JUDGE_USER.format(skeleton=skeleton, test=test),
                jm, 0.0)))
        except Exception:
            scores.append(None)
    if sum(s == 10 for s in scores) < config.JUDGE_PASS_MIN:
        return None

    # 4. reference solution
    solution = _ask(SOLUTION_SYS,
                    SOLUTION_USER.format(skeleton=skeleton, test=test),
                    config.SOLUTION_MODEL, 0.2)

    # 5. coverage filter
    coverage = _line_coverage(solution, test)
    if coverage is None or coverage <= config.COVERAGE_MIN:
        return None

    return {
        "task_id": item["task_id"],
        "class_name": class_name,
        "composition_type": item["composition_type"],
        "domain1": item["domain1"], "domain2": item["domain2"],
        "skeleton": skeleton,
        "test": test,
        "solution_code": solution,
        "coverage": coverage,
        "judge_scores": scores,
    }


def run():
    pairs = json.loads(config.PAIRS_FILE.read_text(encoding="utf-8"))
    results = []
    with ThreadPoolExecutor(max_workers=config.GEN_MAX_WORKERS) as ex:
        futs = {ex.submit(_process, p): p for p in pairs}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                results.append(r)
            print(f"[stage5] {k}/{len(pairs)} kept={len(results)}", end="\r")

    config.DATASET_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[stage5] {len(results)} tasks passed all filters "
          f"-> {config.DATASET_FILE.name}")
    return len(results)


if __name__ == "__main__":
    run()
