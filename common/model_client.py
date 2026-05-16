import json
import os
import openai
from openai import AzureOpenAI, OpenAI
import re
from loguru import logger
from pathlib import Path


class PathUtil:
    """Path utility class"""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.tmp_dir = self.base_dir / "tmp"
        self.output_dir = self.base_dir
        self.logs_dir = self.base_dir / "logs"

        # Create directories if they don't exist
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def get_tmp_dir(self):
        return str(self.tmp_dir)

    def get_output_dir(self):
        return str(self.output_dir)

    def get_logs_dir(self):
        return str(self.logs_dir)

    def get_log_file(self, filename):
        return str(self.logs_dir / f"{filename}.log")

    def get_output_file(self, filename, ext="json"):
        return str(self.output_dir / f"{filename}.{ext}")


class ModelClient:
    """Model client for generating responses from vLLM, OpenAI, or Azure OpenAI API"""

    def __init__(self, model_name, temperature=0.2, max_length=8192,
                 vllm_base_url=None, vllm_api_key=None,
                 openai_api_key=None, openai_base_url=None,
                 use_azure=False, azure_api_key=None, azure_endpoint=None,
                 azure_api_version="2024-03-01-preview"):
        """
        Initialize the model client

        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (default: 0.2)
            max_length: Maximum token length for generation (default: 8192)
            vllm_base_url: vLLM server base URL (optional)
            vllm_api_key: vLLM API key (optional)
            openai_api_key: OpenAI API key (optional)
            openai_base_url: OpenAI base URL (optional)
            use_azure: Whether to use Azure OpenAI (default: False)
            azure_api_key: Azure OpenAI API key (optional)
            azure_endpoint: Azure OpenAI endpoint URL (optional)
            azure_api_version: Azure API version (default: "2024-03-01-preview")
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_length = max_length

        # Determine which API to use
        if use_azure:
            # Use Azure OpenAI API
            if not azure_api_key or not azure_endpoint:
                raise ValueError("azure_api_key and azure_endpoint are required when use_azure=True")

            self.api_type = "azure"
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
            logger.info(f"Initialized Azure OpenAI client with endpoint: {azure_endpoint}")

        elif openai_api_key:
            # Use OpenAI API (default or custom base URL)
            self.api_type = "openai"
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_base_url  # Use OpenAI's default if None
            )
            logger.info(
                f"Initialized OpenAI client" + (f" with base URL: {openai_base_url}" if openai_base_url else ""))

        else:
            # Use vLLM API
            self.api_type = "vllm"
            self.client = OpenAI(
                base_url=vllm_base_url,
                api_key=vllm_api_key or "EMPTY"
            )
            logger.info(f"Initialized vLLM client with base URL: {vllm_base_url}")

    def generate(self, prompt, greedy=False, max_tokens=None):
        """Generate response using configured API (vLLM, OpenAI, or Azure OpenAI)"""
        try:
            logger.info(f"Using {self.api_type} API to query model {self.model_name}")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens if max_tokens is not None else self.max_length,
                temperature=0 if greedy else self.temperature,
            )

            # Show the number of tokens of input and output
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"Input tokens: {response.usage.prompt_tokens}")
                logger.info(f"Output tokens: {response.usage.completion_tokens}")

            return response.choices[0].message.content

        except Exception as e:
            logger.warning(f"Error querying model {self.model_name}: {e}")
            return ""


def get_leading_spaces(string):
    """Get the number of leading spaces in a string"""
    return len(string) - len(string.lstrip())


def extract_code(text, model_name=""):
    """
    Extract code from model response with support for multiple formats and models

    Args:
        text: Raw model response text
        model_name: Name of the model to handle specific formats

    Returns:
        Extracted code as string
    """
    text = text.rstrip()
    output_split_identifier_list = ["### Response:", "@@ Response:", "[/INST]"]
    for identifier in output_split_identifier_list:
        if identifier in text:
            text = text.split(identifier)[1]
            break

    # Handle incoder specific format
    if "incoder" in model_name:
        if "<|/ file |>" in text:
            text = text.split("<|/ file |>")[0]
        return text.strip()

    # Try to extract code blocks with standard patterns
    pattern_list = [r"```python(.*?)```", r"```(.*?)```", r"\[PYTHON\](.*?)\[/PYTHON\]"]
    for pattern in pattern_list:
        try:
            code = re.findall(pattern, text, re.DOTALL)
            if code:
                # Get the longest match which is most likely the complete code
                code = max(code, key=lambda x: len(x.strip()))
                return code.strip()
        except:
            continue

    # If no code block matches, return the raw text (simplified approach)
    logger.warning("No code block pattern matched, returning raw text")
    return text.strip()


def save_results(results, output_path):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Results saved to: {output_path}")


def save_config(config, output_dir):
    """Save experiment configuration to output directory"""
    config_path = os.path.join(output_dir, "experiment_config.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    logger.info(f"Configuration saved to: {config_path}")


def load_data(data_path, data_limit=None):
    """Load data from JSON file with optional limit"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if data_limit is not None:
        original_count = len(data)
        data = data[:data_limit]
        logger.info(f"Applied data limit: processing {len(data)} out of {original_count} tasks")

    return data
