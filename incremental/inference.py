import json
import re
from loguru import logger
from utils import ModelClient


class HierarchicalCodeGenPipeline:
    """
    Incremental code generation pipeline:
    1. Extract method list from skeleton
    2. Generate code incrementally (method by method)
    """

    def __init__(self, model_client):
        self.model_client = model_client

    def clean_and_parse_list(self, llm_output):
        """Try multiple approaches to parse list from LLM output"""
        try:
            return json.loads(llm_output)
        except:
            pass
        
        try:
            match = re.search(r'\[.*?\]', llm_output, re.DOTALL)
            if match:
                list_str = match.group(0)
                try:
                    import ast
                    return ast.literal_eval(list_str)
                except:
                    return json.loads(list_str.replace("'", '"'))
        except Exception as e:
            logger.error(f"[Parse Error] Could not parse list: {e}")
        
        return []

    def extract_code_block(self, text):
        """Extract code block from LLM output - enhanced version"""
        if not text or not text.strip():
            return ""
        
        # Prefer matching ```python code blocks (\s* matches possible whitespace)
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
        
        # If no code block markers but looks like code
        if 'class ' in text or 'def ' in text:
            return text.strip()
        
        logger.warning(f"[WARN] Could not extract code block from output")
        return ""

    def is_valid_code(self, code):
        """Check if extracted code is valid"""
        if not code or len(code.strip()) < 10:
            return False
        return any(kw in code for kw in ['class ', 'def ', 'import '])

    def build_extraction_prompt(self, class_name, skeleton):
        """Build method extraction prompt"""
        instruction = (
            f"Analyze the skeleton of class '{class_name}'. "
            f"Output a single Python list containing the names of all methods (include '__init__') that need to be implemented. "
            f"Example format: ['method_a', 'method_b'].\n"
            f"Strictly output the list ONLY, without any markdown formatting or extra text."
        )
        return instruction + '\n' + skeleton

    def build_generation_prompt(self, class_name, method_name, skeleton, current_context, is_first):
        """Build code generation prompt"""
        if is_first:
            instruction = (
                f"Task: Start implementing the class `{class_name}` based on the skeleton provided below.\n"
                f"Current Target: Create the file structure (imports, class definition) and implement method `{method_name}`.\n\n"
                f"Strict Constraints:\n"
                f"1. Output the **ENTIRE** Python file content.\n"
                f"2. MUST include necessary imports at the top.\n"
                f"3. MUST include `class {class_name}:` definition.\n"
                f"4. Implement `{method_name}` inside the class.\n"
                f"5. Output pure Python code enclosed in ```python ... ```.\n"
            )
            context = skeleton
        else:
            instruction = (
                f"Reference Documentation:\n{skeleton}\n\n"
                f"Current Code State:\n```python\n{current_context}\n```\n\n"
                f"Task: Continue implementing class `{class_name}`. Add the implementation for method `{method_name}`.\n\n"
                f"Strict Constraints:\n"
                f"1. Output the **ENTIRE** updated Python file.\n"
                f"2. **KEEP** all previous imports, class definitions, and methods exactly as they are.\n"
                f"3. **ADD** the implementation of `{method_name}`.\n"
                f"4. Output pure Python code enclosed in ```python ... ```.\n"
            )
            context = skeleton
        
        return instruction + '\n' + context

    def process_single_item(self, data_item, sample_nums=1, greedy=False):
        """
        Process a single data item using incremental pipeline
        """
        pred = []
        raw_outputs_all = []
        class_name = data_item['class_name']
        task_id = data_item.get('task_id', 'unknown')
        skeleton = data_item.get('skeleton', '')

        for i in range(sample_nums):
            logger.info(f"{'='*60}")
            logger.info(f"🚀 Pipeline Run {i + 1}/{sample_nums}")
            logger.info(f"📋 Task: {task_id} | Class: {class_name}")
            logger.info(f"{'='*60}")

            # Phase 1: Extract method list
            extract_prompt = self.build_extraction_prompt(class_name, skeleton)
            method_extraction_result = self.model_client.generate(extract_prompt, greedy=greedy)
            extracted_methods = self.clean_and_parse_list(method_extraction_result)
            
            logger.info(f"[INFO] Task: {task_id} | Extracted: {extracted_methods}")
            
            # Return skeleton if extraction fails
            if not extracted_methods:
                logger.warning(f"[WARN] Task: {task_id} | Method extraction failed, using skeleton")
                pred.append(skeleton)
                raw_outputs_all.append([method_extraction_result])
                continue
            
            # Phase 2: Incremental generation
            current_context = ""
            sample_raw_outputs = []
            
            for idx, method_name in enumerate(extracted_methods):
                # Build prompt
                prompt = self.build_generation_prompt(
                    class_name, method_name, skeleton, 
                    current_context, is_first=(idx == 0)
                )
                
                # Call model
                output = self.model_client.generate(prompt, greedy=greedy)
                sample_raw_outputs.append(output)
                
                # Extract code
                generated_code = self.extract_code_block(output)
                
                # Validate and update
                if self.is_valid_code(generated_code):
                    logger.info(f"[INFO] Task: {task_id} | Method: {method_name} | Code length: {len(generated_code)}")
                    current_context = generated_code
                else:
                    logger.warning(f"[WARN] Task: {task_id} | Method: {method_name} | Invalid code, keeping previous state")
                    # If first round fails, keep skeleton as fallback
                    if idx == 0:
                        current_context = skeleton
            
            # Final fallback: return skeleton if no valid code was generated
            if not current_context or not self.is_valid_code(current_context):
                logger.error(f"[ERROR] Task: {task_id} | No valid code generated, using skeleton")
                pred.append(skeleton)
                raw_outputs_all.append(sample_raw_outputs)
            else:
                logger.info("✅ Code generation completed")
                pred.append(current_context)
                raw_outputs_all.append(sample_raw_outputs)

        return pred, raw_outputs_all
