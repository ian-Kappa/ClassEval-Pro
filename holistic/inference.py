import json
import re
from loguru import logger
from utils import ModelClient, extract_code
from typing import Dict, List


class HolisticCodeGenPipeline:
    """
    Holistic code generation pipeline:
    Directly generates the entire class implementation in a single pass (One-shot).
    """

    def __init__(self, model_client: ModelClient):
        self.model_client = model_client

    def construct_direct_prompt(self, skeleton, greedy=False):
        """
        Stage 1: Construct prompt and directly call the model generate method.
        """
        prompt = f"""
Please complete the class in following skeleton:

## Skeleton
{skeleton}

## Output Format
Output the complete code in a single ```python``` code block. DO NOT include any text outside the code block. NO comments, NO docstrings, NO doctests.
"""
   
        logger.debug(
                f"\n{'#'*30} [ FULL PROMPT START ] {'#'*30}\n"
                f"{prompt.strip()}\n"
                f"{'#'*31} [ FULL PROMPT END ] {'#'*31}"
            )
        # Execute generation here and return response
        return self.model_client.generate(prompt, greedy=greedy)

    def process_single_item(self, data_item: Dict, sample_nums = 1, greedy = False) :
        """
        Process a single data item using a direct, holistic generation approach
        """
        pred = []
        task_id = data_item.get('task_id', 'unknown')
        skeleton = data_item['skeleton']

        for i in range(sample_nums):
            logger.info(f"{'='*60}")
            logger.info(f"🚀 Holistic Pipeline Run {i + 1}/{sample_nums}")
            logger.info(f"📋 Task: {task_id}")
            logger.info(f"{'='*60}")

            try:
                # Step 1 & 2: Prompt Construction and Generation are now combined
                logger.info(f"🏗️  [Step 1/1] Generating code via construct_direct_prompt...")
                
                # Get model response directly
                response = self.construct_direct_prompt(skeleton, greedy=greedy)
                
                # Step 3: Extraction
                current_code = extract_code(response)
                
                if current_code:
                    logger.info("✅ Code generation and extraction completed successfully")
                else:
                    logger.warning("⚠️  Extraction failed: No Python code block found in response")
                
                pred.append(current_code)
                
            except Exception as e:
                logger.error(f"❌ Error during generation for {task_id}: {str(e)}")
                pred.append("")

        return pred