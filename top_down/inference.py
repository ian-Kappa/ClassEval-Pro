import json
import re
from loguru import logger
from utils import ModelClient, extract_code
from typing import Dict, List


class HierarchicalCodeGenPipeline:
    """
    Hierarchical code generation pipeline:
    1. Analyze method dependencies
    2. Generate code top-down based on dependency tree
    """

    def __init__(self, model_client):
        self.model_client = model_client

    def dependency_analysis(self, skeleton, class_name, greedy=False):
        """
        Stage 1: Analyze method dependencies within the class
        """
        prompt = f"""
# Role
You are a Python static analyzer specializing in method dependency extraction.

# Goal
Analyze `{class_name}` and output a JSON dependency tree indicating how methods depend on each other within this class.

# Input
```python
{skeleton}
```

# Dependency Rules
Method A depends on Method B if ANY of these conditions hold:
1. **Direct call**: A explicitly invokes B (e.g., `self.B()` or `B()`)
2. **State dependency**: A reads/uses instance attributes that B modifies
3. **Execution order**: A's correctness requires B to run first

## Dependency Level Assignment
- **Level 0**: Independent methods (no dependencies on other class methods)
- **Level N**: Methods depending only on levels 0 to N-1

# Workflow
1. **Parse skeleton**: Extract all method signatures and their bodies
2. **Identify calls**: For each method, find all `self.method_name()` invocations
3. **Track state**: Detect which methods read/write shared instance attributes
4. **Compute levels**: Assign minimum level where all dependencies are at lower levels
5. **Validate completeness**: Ensure ALL methods from the skeleton are included

# Output Format
Return ONLY a valid JSON object with this exact structure:

```json
{{
  "method_name_1": {{
    "level": 0,
    "depends_on": []
  }},
  "method_name_2": {{
    "level": 1,
    "depends_on": ["method_name_1"]
  }}
}}
```

# Constraints
- Include every method from `{class_name}` (including `__init__`)
- Use exact method names as they appear in the skeleton
- Order methods by level (descending: highest level first for top-down implementation)
- Return pure JSON—no markdown fences, no explanations, no comments
- If uncertain about a dependency, include it (false positives are safer than missing dependencies)
"""
        return self.model_client.generate(prompt, greedy=greedy)

    def extract_dependency_tree(self, response):
        """
        Extract dependency tree from model response with robust parsing
        """
        try:
            # Try to extract from markdown code block first
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, response, re.DOTALL)

            if json_match:
                tree_str = json_match.group(1)
            else:
                # Try to find raw JSON object
                json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_match = re.search(json_obj_pattern, response, re.DOTALL)
                if json_match:
                    tree_str = json_match.group(0)
                else:
                    logger.warning("No JSON structure found in response")
                    return {}

            # Clean and parse JSON
            cleaned_tree_str = re.sub(r'//.*?[\n\r]', '\n', tree_str)
            cleaned_tree_str = re.sub(r',\s*}', '}', cleaned_tree_str)  # Remove trailing commas
            json_tree = json.loads(cleaned_tree_str)

            if not isinstance(json_tree, dict):
                logger.warning("Extracted structure is not a valid dictionary")
                return {}

            logger.info("✅ Successfully extracted dependency tree:")
            logger.info(json.dumps(json_tree, indent=2, ensure_ascii=False))
            return json_tree

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Problematic string: {tree_str[:200]}...")
            return {}
        except Exception as e:
            logger.error(f"Dependency extraction failed: {e}")
            return {}

    def construct_dependency_based_prompt(self, skeleton, dependency_tree, class_name, greedy=False):
        """
        Stage 2: Generate code using dependency-aware top-down approach
        """
        # Format dependency tree for better readability
        dependency_tree_formatted = json.dumps(dependency_tree, indent=2, ensure_ascii=False)

        # Extract method order from dependency tree (top-down: highest level first)
        sorted_methods = sorted(dependency_tree.items(), key=lambda x: x[1]['level'], reverse=True)
        method_order = [method_name for method_name, _ in sorted_methods]
        method_order_str = " → ".join(method_order)

        prompt = f"""
# Role
You are an expert Python developer implementing class methods with strict dependency awareness.

# Goal
Implement ALL methods of `{class_name}` following the top-down dependency order to ensure correctness.

# Input Context

## Class Skeleton
```python
{skeleton}
```

## Dependency Structure (Critical)
The methods MUST be implemented in this order:
```json
{dependency_tree_formatted}
```

**Implementation Order**: {method_order_str}

# Constraints

## Dependency Requirements
1. **Top-down implementation**: Implement higher-level methods first (those that depend on others), then their dependencies
2. **Dependency anticipation**: When implementing a method at Level N, assume its dependencies (from `depends_on` list) will be implemented later
3. **Forward references allowed**: You can call methods before they're implemented in this top-down approach

## Code Quality Requirements
1. **Skeleton compliance**: Preserve all signatures, parameter names, and return types from the skeleton
2. **Docstring adherence**: If docstrings contain usage examples, your implementation MUST pass those examples
3. **Type flexibility**: Handle various input types gracefully (e.g., accept both `int` and `str` where sensible)
4. **Edge cases**: Handle empty inputs, None values, out-of-range indices, and other boundary conditions
5. **Dynamic solutions**: Prefer algorithmic approaches over hardcoded logic
6. **No external dependencies**: Use only Python standard library (no pip packages)

## Forbidden Patterns
- DO NOT use placeholder code like `pass` or `raise NotImplementedError`
- DO NOT add methods not present in the skeleton
- DO NOT change method signatures
- DO NOT ignore dependency order

# Workflow
1. **Start with highest level**: Implement methods that depend on others first
2. **Progress downward**: Move to lower levels (dependencies) after high-level methods are complete
3. **Assume dependencies exist**: When implementing a method, assume its dependencies will be implemented later
4. **Test mentally**: Trace through docstring examples to verify correctness

# Output Format
Return ONLY the complete, runnable Python code:

```python
# Complete implementation of {class_name}
# All methods implemented following top-down dependency order: {method_order_str}

<your complete class implementation here>
```

**Critical**: No explanations, no markdown outside the code block, just the working code.
"""
        return self.model_client.generate(prompt, greedy=greedy)

    def process_single_item(self, data_item, sample_nums=1, greedy=False):
        """
        Process a single data item using dependency-aware pipeline
        """
        pred = []
        class_name = data_item['class_name']
        task_id = data_item.get('task_id', 'unknown')

        for i in range(sample_nums):
            logger.info(f"{'='*60}")
            logger.info(f"🚀 Pipeline Run {i + 1}/{sample_nums}")
            logger.info(f"📋 Task: {task_id} | Class: {class_name}")
            logger.info(f"{'='*60}")

            skeleton = data_item['skeleton']

            # Stage 1: Dependency Analysis
            logger.info("📊 [Stage 1/2] Analyzing method dependencies...")
            dependency_response = self.dependency_analysis(skeleton, class_name)
            dependency_tree = self.extract_dependency_tree(dependency_response)

            if not dependency_tree:
                logger.warning("⚠️  Dependency analysis failed, using fallback generation")
                # Fallback: simple generation without dependency awareness
                fallback_prompt = f"""Implement the following Python class completely:

```python
{skeleton}
```

Return only the complete implementation in a Python code block."""
                response = self.model_client.generate(fallback_prompt, greedy=greedy)
                current_code = extract_code(response)
            else:
                logger.info(f"✅ Extracted {len(dependency_tree)} methods with dependency structure")

                # Stage 2: Top-down Code Generation
                logger.info("🏗️  [Stage 2/2] Generating code with dependency-aware approach...")
                response = self.construct_dependency_based_prompt(skeleton, dependency_tree, class_name)
                current_code = extract_code(response)

            logger.info("✅ Code generation completed")
            pred.append(current_code)

        return pred