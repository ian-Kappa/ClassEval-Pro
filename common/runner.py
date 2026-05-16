import argparse
from common.evaluate import run_evaluation
from common.model_client import ModelClient, load_data, save_results, save_config
from loguru import logger
from tqdm import tqdm
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set by main(); the strategy-specific pipeline class (e.g. HolisticCodeGenPipeline).
_PIPELINE_CLS = None


def args_init(default_max_length=16384, default_auto_test=0):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data.json",
        help="ClassEval data (default: repo-root data.json, run from a strategy dir)",
    )
    parser.add_argument(
        "--greedy",
        type=int,
        default=0,
        help="Whether to generate model results with greedy strategy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory path where results will be saved",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_NAME", "YOUR_MODEL_DEPLOYMENT_NAME"),
        help="Model name for Azure OpenAI service",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="temperature value in generation config",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=default_max_length,
        help="max length of model's generation result",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="",
        help="API base URL for vLLM server (deprecated, use azure_endpoint)",
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default="",
        help="API key for vLLM server (deprecated, use azure_api_key)",
    )
    parser.add_argument(
        "--azure_api_key",
        type=str,
        default="YOUR_API_KEY",
        help="API key for Azure OpenAI",
    )
    parser.add_argument(
        "--azure_endpoint",
        type=str,
        default="YOUR_API_ENDPOINT",
        help="Azure endpoint URL",
    )
    parser.add_argument(
        "--azure_api_version",
        type=str,
        default="2024-03-01-preview",
        help="Azure OpenAI API version",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="",
        help="API key for OpenAI API (deprecated, use azure_api_key)",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default="",
        help="Base URL for OpenAI API (deprecated, use azure_endpoint)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="The number of code samples that are randomly generated for each task.",
    )
    parser.add_argument(
        "--data_limit",
        type=int,
        default=None,
        help="Limit the number of tasks to process (e.g., only process first k tasks). If None, process all tasks.",
    )
    parser.add_argument(
        "--auto_test",
        type=int,
        default=default_auto_test,
        help="Whether to automatically run tests after inference (1=yes, 0=no)",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="Path to evaluation data for auto testing (optional, will auto-detect if not provided)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of parallel workers for processing tasks",
    )
    parser.add_argument(
        "--use_azure",
        type=int,
        default=1,
        help="Use Azure OpenAI (1=yes, 0=no for regular OpenAI)",
    )
    args = parser.parse_args()
    return args


def process_single_task(args, data_item, worker_id):
    """
    Process a single data item in a worker thread

    Args:
        args: Command line arguments
        data_item: Data item to process
        worker_id: Unique worker ID for this thread

    Returns:
        tuple: (success, result_data_item, error_message)
    """
    try:
        # Create a separate model client for this worker with Azure configuration
        model_client = ModelClient(
            model_name=args.model,
            use_azure=bool(args.use_azure),
            azure_api_key=args.azure_api_key,
            azure_endpoint=args.azure_endpoint,
            azure_api_version=args.azure_api_version,
            # Legacy parameters for backward compatibility
            vllm_base_url=args.vllm_base_url,
            vllm_api_key=args.vllm_api_key,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
            temperature=args.temperature,  # Use user-configured temperature
            max_length=args.max_length
        )

        # Create pipeline (strategy class injected via main())
        pipeline = _PIPELINE_CLS(model_client)

        # Process the task
        result = pipeline.process_single_item(
            data_item=data_item,
            sample_nums=args.sample,
            greedy=bool(args.greedy)
        )

        # Some strategies (Incremental/Compositional) also return raw outputs
        if isinstance(result, tuple):
            predictions, raw_outputs = result
            data_item['raw_output'] = raw_outputs
        else:
            predictions = result

        # Add predictions to data item
        data_item['predict'] = predictions

        return (True, data_item, None)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Worker {worker_id} failed processing task {data_item.get('task_id', 'unknown')}: {error_msg}")
        return (False, data_item, error_msg)


class ThreadSafeResultCollector:
    """Thread-safe result collector"""

    def __init__(self, output_path):
        self.results = []
        self.error_task_ids = []
        self.lock = threading.Lock()
        self.output_path = output_path

    def add_result(self, success, data_item, error_msg=None):
        with self.lock:
            if success:
                self.results.append(data_item)
            else:
                self.error_task_ids.append(data_item.get('task_id', 'unknown'))

            # Save intermediate results every 10 completed tasks
            if len(self.results) % 10 == 0:
                save_results(self.results, self.output_path)

    def get_results(self):
        with self.lock:
            return self.results.copy(), self.error_task_ids.copy()

    def save_final_results(self):
        with self.lock:
            save_results(self.results, self.output_path)


def main(pipeline_cls, default_max_length=16384, default_auto_test=0):
    global _PIPELINE_CLS
    _PIPELINE_CLS = pipeline_cls
    args = args_init(default_max_length, default_auto_test)

    logger.info("=" * 60)
    logger.info("Multi-threaded Code Generation and Testing Pipeline")
    logger.info("=" * 60)
    logger.info(f"API Type: {'Azure OpenAI' if args.use_azure else 'OpenAI'}")
    logger.info(f"Model: {args.model}")
    if args.use_azure:
        logger.info(f"Azure Endpoint: {args.azure_endpoint}")
        logger.info(f"Azure API Version: {args.azure_api_version}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max Length: {args.max_length}")
    logger.info(f"Sample Count: {args.sample}")
    logger.info(f"Data Limit: {args.data_limit if args.data_limit else 'None (process all)'}")
    logger.info(f"Greedy Mode: {'Enabled' if args.greedy else 'Disabled'}")
    logger.info(f"Input Data: {args.data_path}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Auto Test: {'Enabled' if args.auto_test else 'Disabled'}")
    logger.info(f"Number of Workers: {args.num_workers}")
    if args.eval_data_path:
        logger.info(f"Eval Data Path: {args.eval_data_path}")
    logger.info("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = load_data(args.data_path, args.data_limit)

    # Save configuration
    config = {
        'api_type': 'azure' if args.use_azure else 'openai',
        'model': args.model,
        'temperature': args.temperature,  # Use user-configured temperature
        'max_length': args.max_length,
        'greedy': args.greedy,
        'sample_nums': args.sample,  # Fixed: Only 1 output per data item
        'azure_endpoint': args.azure_endpoint if args.use_azure else None,
        'azure_api_version': args.azure_api_version if args.use_azure else None,
        'data_path': args.data_path,
        'data_limit': args.data_limit,
        'output_dir': args.output_dir,
        'auto_test': args.auto_test,
        'num_workers': args.num_workers
    }
    save_config(config, args.output_dir)

    # Initialize result collector
    output_path = os.path.join(args.output_dir, "results.json")
    result_collector = ThreadSafeResultCollector(output_path)

    # Process all data items with multi-threading
    logger.info(f"Starting inference pipeline with {args.num_workers} workers...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_task = {}
        for i, data_item in enumerate(data):
            worker_id = i % args.num_workers  # Distribute tasks among workers
            future = executor.submit(process_single_task, args, data_item, worker_id)
            future_to_task[future] = data_item

        # Process completed tasks with progress bar
        with tqdm(total=len(data), desc="Processing tasks") as pbar:
            for future in as_completed(future_to_task):
                data_item = future_to_task[future]
                try:
                    success, result_data_item, error_msg = future.result()
                    result_collector.add_result(success, result_data_item, error_msg)
                except Exception as e:
                    logger.error(f"Future execution failed for task {data_item.get('task_id', 'unknown')}: {e}")
                    result_collector.add_result(False, data_item, str(e))

                pbar.update(1)

    # Get final results
    results, error_task_ids = result_collector.get_results()
    result_collector.save_final_results()

    end_time = time.time()
    total_time = end_time - start_time

    logger.info("=" * 60)
    logger.info("Inference Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"Total tasks: {len(data)}")
    logger.info(f"Successful tasks: {len(results)}")
    logger.info(f"Failed tasks: {len(error_task_ids)}")
    logger.info(f"Success rate: {len(results) / len(data) * 100:.2f}%")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per task: {total_time / len(data):.2f} seconds")
    logger.info(f"Workers used: {args.num_workers}")

    if error_task_ids:
        logger.warning(f"Failed task IDs: {error_task_ids}")

    logger.info("Inference pipeline completed!")

    # Run testing if enabled
    if args.auto_test:
        logger.info("Starting testing pipeline...")

        # Determine eval data path
        eval_data_path = args.eval_data_path
        if eval_data_path is None:
            # Auto-detect from data_path
            eval_data_path = args.data_path
            logger.info(f"Auto-detected eval data path: {eval_data_path}")

        # Determine model result file path
        model_result_file = os.path.join(args.output_dir, "results.json")

        # Extract model name from model path
        model_name = args.model.split('/')[-1] if '/' in args.model else args.model

        # Run evaluation using the test.py functionality
        test_metrics = run_evaluation(
            model_result_file=model_result_file,
            eval_data_file=eval_data_path,
            model_name=model_name
        )

        if test_metrics:
            logger.info("Testing pipeline completed successfully!")
            logger.info("Test metrics:")
            for metric_type, results in test_metrics.items():
                logger.info(f"  {metric_type}:")
                for model, scores in results.items():
                    logger.info(f"    {model}:")
                    for score_name, score_value in scores.items():
                        logger.info(f"      {score_name}: {score_value:.4f}")
        else:
            logger.warning("Testing pipeline completed with issues.")

    else:
        logger.info("Testing skipped (auto_test disabled)")

    logger.info("\n" + "=" * 60)
    logger.info("Experiment Complete!")
    logger.info("=" * 60)
    logger.info(f"All results saved to: {args.output_dir}")
    logger.info("Output files:")
    logger.info(f"  - results.json (inference results)")
    if args.auto_test:
        logger.info(f"  - output/pass_at_k_result.json (evaluation metrics)")
        logger.info(f"  - output/detailed_result.json (detailed test results)")
        logger.info(f"  - logs/ (test execution logs)")
    logger.info("=" * 60)