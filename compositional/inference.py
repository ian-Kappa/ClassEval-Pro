import json
import re
from loguru import logger
from utils import ModelClient

MAX_RETRIES = 3


class CompositionalCodeGenPipeline:
    """
    Compositional code generation pipeline:
    1. Extract method list from skeleton
    2. Generate import statements and class definition
    3. Generate each method independently (based on original skeleton)
    4. Assemble complete class code
    """

    def __init__(self, model_client):
        self.model_client = model_client

    # ========== Parsing Utilities ==========

    def clean_and_parse_list(self, llm_output):
        """Try multiple approaches to parse list from LLM output"""
        if not llm_output or not llm_output.strip():
            return []

        try:
            result = json.loads(llm_output)
            if isinstance(result, list):
                return result
        except Exception:
            pass

        try:
            match = re.search(r'\[.*?\]', llm_output, re.DOTALL)
            if match:
                list_str = match.group(0)
                try:
                    import ast
                    result = ast.literal_eval(list_str)
                    if isinstance(result, list):
                        return result
                except Exception:
                    result = json.loads(list_str.replace("'", '"'))
                    if isinstance(result, list):
                        return result
        except Exception as e:
            logger.error(f"[Parse Error] Could not parse list: {e}")

        return []

    def extract_methods_from_skeleton(self, skeleton):
        """
        Extract method names from Markdown skeleton using regex as fallback.
        Matches ## `method_name(...)` format headers.
        """
        pattern = r'##\s+`(\w+)\s*\('
        methods = re.findall(pattern, skeleton)
        if methods:
            logger.info(f"[Regex Fallback] Extracted methods from skeleton: {methods}")
        return methods

    def extract_method_signature_from_skeleton(self, method_name, skeleton):
        """
        Extract method signature from Markdown skeleton.
        Matches ## `method_name(param1, param2=default, ...)` format.
        Returns complete signature string like "self, param1, param2=None"
        """
        pattern = rf'##\s+`{re.escape(method_name)}\(([^)]*)\)`'
        match = re.search(pattern, skeleton)
        if match:
            params_str = match.group(1).strip()
            if params_str:
                # Add self to parameter list
                return f"self, {params_str}"
            else:
                return "self"
        return "self"

    def extract_code_block(self, text):
        """Extract code block from LLM output"""
        if not text or not text.strip():
            return ""

        # Prefer matching python code blocks
        match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code

        # Match any code block
        match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code

        # If no markdown markers but looks like pure code (lines starting with def/class/import), return directly
        lines = text.strip().split('\n')
        code_lines = [l for l in lines if l.strip()]
        if code_lines and any(code_lines[0].strip().startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ']):
            return text.strip()

        return ""

    def is_valid_code(self, code):
        """Check if extracted code is valid (exclude Markdown content)"""
        if not code or len(code.strip()) < 10:
            return False
        # Contains Markdown headers, not code
        if re.search(r'^#{1,3}\s+', code, re.MULTILINE):
            return False
        # Contains Markdown bold format
        if '**Purpose**' in code or '**Parameters**' in code:
            return False
        return any(kw in code for kw in ['class ', 'def ', 'import '])

    # ========== Prompt Construction ==========

    def build_extraction_prompt(self, class_name, skeleton):
        """Build method extraction prompt"""
        instruction = (
            f"Analyze the skeleton of class '{class_name}'. "
            f"Output a single Python list containing the names of all methods (include '__init__') that need to be implemented. "
            f"Example format: ['method_a', 'method_b'].\n"
            f"Strictly output the list ONLY, without any markdown formatting or extra text."
        )
        return instruction + '\n' + skeleton

    def build_import_class_prompt(self, class_name, skeleton):
        """Build prompt for generating import info and class name"""
        instruction = (
            f"Task: Generate the complete import information, class definition and constructor (without any other methods) for class '{class_name}' based on the skeleton.\n\n"
            f"Strict Constraints:\n"
            f"1. Output ONLY the import statements, the class definition line like 'class ClassName(...): and constructor this class needs'\n"
            f"2. Do NOT include any method definitions or stubs.\n"
            f"3. Output pure Python code enclosed in ```python ... ```.\n"
            f"4. The output should be the minimal and complete code that can be used as the beginning of the class file.\n"
        )
        return instruction + '\n' + skeleton

    def build_method_generation_prompt(self, class_name, method_name, skeleton):
        """Build prompt for single method generation"""
        instruction = (
            f"Task: Implement ONLY the method `{method_name}` in class `{class_name}`.\n\n"
            f"Strict Constraints:\n"
            f"1. Output ONLY the method implementation (the `def {method_name}(...)` block) with its body.\n"
            f"2. Do NOT include imports, class definition, or other methods.\n"
            f"3. Output pure Python code enclosed in ```python ... ```.\n"
            f"4. The method should be complete and ready to be inserted into the class.\n"
        )
        return instruction + '\n' + skeleton

    # ========== Generation with Retry ==========

    def generate_with_retry(self, prompt, greedy=False, max_retries=MAX_RETRIES):
        """LLM call with retry, returns non-empty result or empty string after retries exhausted"""
        for attempt in range(1, max_retries + 1):
            output = self.model_client.generate(prompt, greedy=greedy)
            if output and output.strip():
                return output
            logger.warning(f"[Retry] Attempt {attempt}/{max_retries} returned empty, retrying...")
        return ""

    # ========== Assembly ==========

    def assemble_complete_class(self, import_class_part, method_codes):
        """
        Merge generated import info, class name, and all method code into a complete Python file string.
        Handles indentation automatically.
        """
        if not import_class_part.strip():
            return ""

        lines = import_class_part.split('\n')

        # 1. Determine method indentation level (typically class indent + 4 spaces)
        class_indent_len = 0
        for line in lines:
            if line.strip().startswith('class '):
                class_indent_len = len(line) - len(line.lstrip())
                break

        # Build indent prefix
        indent_str = ' ' * (class_indent_len + 4)

        # Initialize full code with import and class parts
        full_code = import_class_part.rstrip() + "\n\n"

        for code in method_codes:
            if not code.strip():
                continue

            # 2. First remove common indentation from method code, then add class-level indentation
            code_lines = code.split('\n')
            non_empty_lines = [l for l in code_lines if l.strip()]
            if non_empty_lines:
                min_indent = min(len(l) - len(l.lstrip()) for l in non_empty_lines)
            else:
                min_indent = 0

            indented_method_lines = []
            for line in code_lines:
                if line.strip():
                    indented_method_lines.append(indent_str + line[min_indent:])
                else:
                    indented_method_lines.append("")

            full_code += "\n".join(indented_method_lines) + "\n\n"

        return full_code.strip()

    # ========== Main Pipeline ==========

    def process_single_item(self, data_item, sample_nums=1, greedy=False):
        """
        Process a single data item using compositional pipeline

        Args:
            data_item: Data item containing class_name, skeleton, etc.
            sample_nums: Number of samples
            greedy: Whether to use greedy decoding

        Returns:
            pred: List of generated code (each element is a complete class code)
            raw_outputs_all: List of raw outputs for each sample
        """
        pred = []
        raw_outputs_all = []
        class_name = data_item['class_name']
        task_id = data_item.get('task_id', 'unknown')
        skeleton = data_item.get('skeleton', '')

        for i in range(sample_nums):
            logger.info(f"{'=' * 60}")
            logger.info(f"Pipeline Run {i + 1}/{sample_nums}")
            logger.info(f"Task: {task_id} | Class: {class_name}")
            logger.info(f"{'=' * 60}")

            raw_outputs = []

            # ===== Phase 1: Extract method list (with retry + regex fallback) =====
            extract_prompt = self.build_extraction_prompt(class_name, skeleton)
            extracted_methods = []

            for attempt in range(1, MAX_RETRIES + 1):
                method_extraction_result = self.model_client.generate(extract_prompt, greedy=greedy)
                raw_outputs.append(method_extraction_result)
                extracted_methods = self.clean_and_parse_list(method_extraction_result)
                if extracted_methods:
                    break
                logger.warning(f"[Phase 1] Attempt {attempt}/{MAX_RETRIES} failed to parse method list")

            # All LLM attempts failed -> regex extraction from Markdown skeleton
            if not extracted_methods:
                logger.warning(f"[Phase 1] LLM extraction failed, falling back to regex extraction from skeleton")
                extracted_methods = self.extract_methods_from_skeleton(skeleton)

            if not extracted_methods:
                logger.error(f"[Phase 1] Task: {task_id} | Cannot extract any methods, skipping this sample")
                pred.append("")
                raw_outputs_all.append(raw_outputs)
                continue

            logger.info(f"[Phase 1] Task: {task_id} | Extracted methods: {extracted_methods}")

            # ===== Phase 2: Generate import + class definition + constructor (with retry) =====
            import_class_prompt = self.build_import_class_prompt(class_name, skeleton)
            import_class_part = ""

            for attempt in range(1, MAX_RETRIES + 1):
                import_class_output = self.model_client.generate(import_class_prompt, greedy=greedy)
                raw_outputs.append(import_class_output)
                import_class_part = self.extract_code_block(import_class_output)
                if import_class_part and self.is_valid_code(import_class_part):
                    break
                logger.warning(f"[Phase 2] Attempt {attempt}/{MAX_RETRIES} failed to generate class header")
                import_class_part = ""

            # Still failed -> construct minimal valid class header
            if not import_class_part:
                logger.warning(f"[Phase 2] Task: {task_id} | All retries failed, constructing minimal class header")
                init_sig = self.extract_method_signature_from_skeleton('__init__', skeleton)
                import_class_part = (
                    f"class {class_name}:\n"
                    f"    def __init__({init_sig}):\n"
                    f"        pass"
                )

            logger.info(f"[Phase 2] Task: {task_id} | Class header length: {len(import_class_part)}")

            # ===== Phase 3: Generate methods one by one (with retry + signature-aware placeholders) =====
            method_implementations = []

            for method_name in extracted_methods:
                # Phase 2 already generated __init__, skip to avoid duplication
                if method_name == '__init__':
                    continue

                prompt = self.build_method_generation_prompt(class_name, method_name, skeleton)
                method_code = ""

                for attempt in range(1, MAX_RETRIES + 1):
                    output = self.model_client.generate(prompt, greedy=greedy)
                    raw_outputs.append(output)
                    method_code = self.extract_code_block(output)
                    if method_code:
                        break
                    logger.warning(f"[Phase 3] Method: {method_name} | Attempt {attempt}/{MAX_RETRIES} failed")

                if method_code:
                    logger.info(f"[Phase 3] Task: {task_id} | Method: {method_name} | Code length: {len(method_code)}")
                    method_implementations.append(method_code)
                else:
                    # Parse correct signature from skeleton as placeholder
                    sig = self.extract_method_signature_from_skeleton(method_name, skeleton)
                    placeholder = f"def {method_name}({sig}):\n    pass"
                    logger.warning(f"[Phase 3] Task: {task_id} | Method: {method_name} | Using placeholder: {placeholder.split(chr(10))[0]}")
                    method_implementations.append(placeholder)

            # ===== Phase 4: Assemble final code =====
            final_code = self.assemble_complete_class(import_class_part, method_implementations)

            if final_code and self.is_valid_code(final_code):
                logger.info(f"[Phase 4] Task: {task_id} | Code generation completed, length: {len(final_code)}")
                pred.append(final_code)
            else:
                logger.error(f"[Phase 4] Task: {task_id} | Assembly produced invalid code, marking as empty")
                pred.append("")

            raw_outputs_all.append(raw_outputs)

        return pred, raw_outputs_all
