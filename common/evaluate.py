import argparse
import json
import os
import shutil
import time
import importlib
import unittest
import re
from pathlib import Path
from func_timeout import func_set_timeout, FunctionTimedOut
from scipy.special import comb
from loguru import logger
import glob
from prettytable import PrettyTable
from common.model_client import PathUtil, extract_code, get_leading_spaces


class AutoTest:
    """Automatic testing class"""

    def __init__(self, eval_data_path, base_dir):
        self.path_util = PathUtil(base_dir)
        self.eval_data = self.get_eval_data(eval_data_path)
        self.tmp_dir = self.path_util.get_tmp_dir()

    def get_eval_data(self, eval_data_path):
        eval_data = {}
        logger.info(f"Loading evaluation data from: {eval_data_path}")
        with open(eval_data_path, encoding='utf-8') as file:
            data = json.load(file)
        for item in data:
            eval_data[item['task_id']] = item
        logger.info(f"Loaded {len(eval_data)} evaluation tasks")
        return eval_data

    def gen_py_file(self, test_code_name, code_list, test_code):
        cnt = 0
        for code_snippet in code_list:
            test_code_py = code_snippet + '\n' + test_code
            test_file_path = os.path.join(self.tmp_dir, f'{test_code_name}_{cnt}.py')
            with open(test_file_path, 'w+', encoding='utf-8') as f:
                f.write(test_code_py)
            cnt += 1

    def add_static_statement(self, code):
        filtered_code_list = []
        for line in code.split('\n'):
            if '@staticmethod' in line:
                continue
            filtered_code_list.append(line)
        code = '\n'.join(filtered_code_list)
        final_code_list = []
        for line in code.split('\n'):
            if line.strip().startswith('def ') and 'self' not in line and 'cls' not in line and get_leading_spaces(line) == 4:
                final_code_list.append('    @staticmethod')
            final_code_list.append(line)
        return '\n'.join(final_code_list)

    def gen_code_list(self, file_path, model_name=""):
        code_list = {}
        logger.info(f"Generating code list from: {file_path}")
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)

        # Check solution counts
        solution_counts = [len(item['predict']) for item in data]
        min_solutions = min(solution_counts)
        max_solutions = max(solution_counts)
        avg_solutions = sum(solution_counts) / len(solution_counts)

        logger.info(f"Solution count - Min: {min_solutions}, Max: {max_solutions}, Avg: {avg_solutions:.1f}")

        # Determine which pass@k can be calculated
        possible_k = []
        if min_solutions >= 1:
            possible_k.append(1)
        if min_solutions >= 3:
            possible_k.append(3)
        if min_solutions >= 5:
            possible_k.append(5)

        logger.info(f"Can calculate Pass@K for K = {possible_k}")

        for item in data:
            code_list[item['task_id']] = []
            for predict in item['predict']:
                predict = extract_code(predict, model_name)
                predict = self.add_static_statement(predict)
                code_list[item['task_id']].append(predict)
        logger.info(f"Generated code for {len(code_list)} tasks")
        return code_list, min_solutions

    @func_set_timeout(5)
    def run_unit_test(self, test_code, test_class, model_name):
        # Add temporary directory to Python path
        import sys
        if self.tmp_dir not in sys.path:
            sys.path.insert(0, self.tmp_dir)

        # Save current working directory
        original_cwd = os.getcwd()
        try:
            # Change to temporary directory for test execution
            os.chdir(self.tmp_dir)

            # Clean module cache if module was imported before
            if test_code in sys.modules:
                del sys.modules[test_code]

            # Import the test module
            module = importlib.import_module(test_code)

            # Load test suite
            test_suite = unittest.TestLoader().loadTestsFromTestCase(getattr(module, test_class))
            test_count = test_suite.countTestCases()

            # Log individual test cases
            logger.debug(f"            Found {test_count} test cases in {test_class}")

            # Run tests with custom result handler
            log_path = self.path_util.get_log_file(f"{model_name}_test")
            with open(log_path, 'a', encoding='utf-8') as f:
                # Write test execution header
                f.write(f"\n{'='*60}\n")
                f.write(f"Running {test_class} from {test_code}.py\n")
                f.write(f"{'='*60}\n")

                test_result = unittest.TextTestRunner(
                    stream=f,
                    verbosity=2,
                    descriptions=True,
                    failfast=False
                ).run(test_suite)

                # Write summary
                f.write(f"\nTest Summary for {test_class}:\n")
                f.write(f"  Tests run: {test_result.testsRun}\n")
                f.write(f"  Failures: {len(test_result.failures)}\n")
                f.write(f"  Errors: {len(test_result.errors)}\n")
                f.write(f"{'='*60}\n\n")

            return test_result
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            # Clean module cache
            if test_code in sys.modules:
                del sys.modules[test_code]

    def test(self, code_num, test_code_name, test_classes, model_name):
        result = {}
        for i in range(code_num):
            test_code = test_code_name + '_' + str(i)
            result[test_code] = {}
            logger.info(f"      Testing code sample {i + 1}/{code_num} (file: {test_code}.py)")

            for j, test_class in enumerate(test_classes, 1):
                logger.info(f"        Running test class {j}/{len(test_classes)}: {test_class}")
                res_item = {}
                try:
                    res = self.run_unit_test(test_code, test_class, model_name)
                    res_item['errors'] = len(res.errors)
                    res_item['failures'] = len(res.failures)
                    res_item['testsRun'] = res.testsRun
                    result[test_code][test_class] = res_item

                    # Log test results
                    if res_item['errors'] == 0 and res_item['failures'] == 0:
                        logger.info(f"          ✓ {test_class}: {res_item['testsRun']} tests passed")
                    else:
                        logger.warning(
                            f"          ✗ {test_class}: {res_item['testsRun']} tests, {res_item['errors']} errors, {res_item['failures']} failures")

                except FunctionTimedOut:
                    res_item['errors'] = 0
                    res_item['failures'] = 0
                    res_item['testsRun'] = 0
                    result[test_code][test_class] = res_item
                    logger.warning(f"          ✗ {test_class}: Test execution timed out after 5 seconds")
                except Exception as e:
                    res_item['errors'] = 0
                    res_item['failures'] = 0
                    res_item['testsRun'] = 0
                    result[test_code][test_class] = res_item
                    logger.error(f"          ✗ {test_class}: Test execution failed - {e}")
        return result

    def save_result(self, model_name, result, type_name):
        save_path = self.path_util.get_output_file(f"{model_name}_{type_name}_result")
        with open(save_path, 'w+') as f:
            json.dump(result, f, indent=4, sort_keys=True)
        logger.info(f"Saved {type_name} results to: {save_path}")

    def test_pipeline(self, model_name, gen_file_path):
        result_dict = {}
        code_list, min_solutions = self.gen_code_list(gen_file_path, model_name)
        self.min_solutions = min_solutions

        # Record initial files in project root for cleanup
        project_root = os.path.dirname(os.path.abspath(__file__))
        initial_files = set(os.listdir(project_root))

        # Generate test files
        logger.info("Generating test files...")
        total_tasks = len(code_list)
        for i, task_id in enumerate(code_list, 1):
            logger.info(f"  [{i}/{total_tasks}] Generating test files for task: {task_id}")
            test_code = self.eval_data[task_id]['test']
            task_code_list = code_list[task_id]
            self.gen_py_file(task_id, task_code_list, test_code)
            logger.info(f"    → Generated {len(task_code_list)} test files for task {task_id}")

        # Run unit tests
        logger.info("Running unit tests...")
        successful_tasks = 0
        failed_tasks = 0

        for i, task_id in enumerate(code_list, 1):
            logger.info(f"  [{i}/{total_tasks}] Testing task: {task_id}")
            task_code_list = code_list[task_id]
            test_classes = self.eval_data[task_id]['test_classes']
            logger.info(
                f"    → Task {task_id} has {len(task_code_list)} code samples and {len(test_classes)} test classes")
            try:
                result = self.test(len(task_code_list), task_id, test_classes, model_name)
                result_dict[task_id] = result
                successful_tasks += 1
                logger.info(f"    ✓ Task {task_id} testing completed successfully")
            except Exception as e:
                failed_tasks += 1
                logger.error(f"    ✗ Task {task_id} testing failed: {e}")
                continue

        logger.info(
            f"Testing summary: {successful_tasks} successful, {failed_tasks} failed out of {total_tasks} total tasks")

        # Save results
        self.save_result(model_name, result_dict, "class")
        time.sleep(2)

        # Clean up leaked files in project root
        self.clean_leaked_files(project_root, initial_files)
        self.tear_down()

        return result_dict

    def get_test_answer(self, test_result):
        if test_result['testsRun'] == 0 or test_result['errors'] == test_result['testsRun']:
            return 'error'
        if test_result['errors'] + test_result['failures'] == 0:
            return 'success'
        if test_result['errors'] + test_result['failures'] < test_result['testsRun']:
            return 'partial_success'
        return 'fail'

    def evaluate(self, model_list):
        result_dict = {}
        for model_name in model_list:
            model_result_path = self.path_util.get_output_file(f'{model_name}_class_result')
            with open(model_result_path, 'r') as f:
                model_result = json.load(f)
            result_dict[model_name] = {}
            for task in model_result:
                result_dict[model_name][task] = {}
                # TestClass 级别的统计只在 task 级别初始化一次
                result_dict[model_name][task]["TestClass"] = {}
                result_dict[model_name][task]["TestClass"]["ClassEachTestResult"] = []
                result_dict[model_name][task]["TestClass"]["class_success"] = 0
                result_dict[model_name][task]["TestClass"]["class_partial_success"] = 0
                result_dict[model_name][task]["TestClass"]["class_fail"] = 0
                for test_num in model_result[task]:
                    temp_result = {"success": 0, "partial_success": 0, "fail": 0, "error": 0}
                    for test_class in model_result[task][test_num]:
                        if test_class not in result_dict[model_name][task]:
                            result_dict[model_name][task][test_class] = {}
                            result_dict[model_name][task][test_class]['success'] = 0
                            result_dict[model_name][task][test_class]['partial_success'] = 0
                            result_dict[model_name][task][test_class]['fail'] = 0
                            result_dict[model_name][task][test_class]['error'] = 0
                            result_dict[model_name][task][test_class]["EachTestResult"] = []
                        test_answer = self.get_test_answer(model_result[task][test_num][test_class])
                        result_dict[model_name][task][test_class][test_answer] += 1
                        result_dict[model_name][task][test_class]["EachTestResult"].append(test_answer)
                        temp_result[test_answer] += 1
                    if temp_result['success'] == len(model_result[task][test_num]):
                        result_dict[model_name][task]["TestClass"]["class_success"] += 1
                        result_dict[model_name][task]["TestClass"]["ClassEachTestResult"].append("class_success")
                    elif temp_result['fail'] == 0 and temp_result['error'] == 0:
                        result_dict[model_name][task]["TestClass"]["class_partial_success"] += 1
                        result_dict[model_name][task]["TestClass"]["ClassEachTestResult"].append(
                            "class_partial_success")
                    else:
                        result_dict[model_name][task]["TestClass"]["class_fail"] += 1
                        result_dict[model_name][task]["TestClass"]["ClassEachTestResult"].append("class_fail")

        save_path = self.path_util.get_output_file("detailed_result")
        with open(save_path, 'w+') as f:
            json.dump(result_dict, f, indent=4, sort_keys=True)
        logger.info(f"Saved detailed results to: {save_path}")

    def cal_pass_at_k(self, n, k, k_success):
        total_combinations = comb(k, n)
        if k - k_success >= n:
            without_k_success_combinations = comb(k - k_success, n)
        else:
            without_k_success_combinations = 0
        with_k_success_combinations = total_combinations - without_k_success_combinations
        pass_at_k = with_k_success_combinations / total_combinations
        return pass_at_k

    def cal_metrics_pass_at_k(self, model_list, n, k):
        file_path = self.path_util.get_output_file("detailed_result")
        with open(file_path, 'r') as f:
            test_result = json.load(f)

        result = {}
        for model_name in model_list:
            sum_num = 0
            success_num = 0
            class_success_num = 0
            class_num = 0
            partial_success_num = 0
            partial_success_class_num = 0
            for task in test_result[model_name]:
                class_num += 1
                for test_class in test_result[model_name][task]:
                    try:
                        if test_result[model_name][task][test_class]['success'] != 0:
                            pass_at_k = self.cal_pass_at_k(n, k, test_result[model_name][task][test_class]['success'])
                            success_num += pass_at_k
                        if test_result[model_name][task][test_class]['success'] + \
                                test_result[model_name][task][test_class]['partial_success'] != 0:
                            pass_at_k = self.cal_pass_at_k(n, k, test_result[model_name][task][test_class]['success'] +
                                                           test_result[model_name][task][test_class]['partial_success'])
                            partial_success_num += pass_at_k
                        sum_num += 1
                    except:
                        if test_result[model_name][task][test_class]['class_success'] != 0:
                            pass_at_k = self.cal_pass_at_k(n, k,
                                                           test_result[model_name][task][test_class]['class_success'])
                            class_success_num += pass_at_k
                        k_success = test_result[model_name][task][test_class]['class_success'] + \
                                    test_result[model_name][task][test_class]['class_partial_success']
                        if k_success != 0:
                            pass_at_k = self.cal_pass_at_k(n, k, k_success)
                            partial_success_class_num += pass_at_k

            result[model_name] = {
                "class_success": class_success_num / class_num,
                "class_partial_success": partial_success_class_num / class_num,
                "fun_success": success_num / sum_num,
                "fun_partial_success": partial_success_num / sum_num,
            }
        return result

    def cal_avg_pass_at_1(self, model_list, k):
        """
        Calculate true average pass@1: for each sample, calculate pass@1 (0 or 1), then average
        This is different from pass@1 which uses combinatorial formula
        """
        file_path = self.path_util.get_output_file("detailed_result")
        with open(file_path, 'r') as f:
            test_result = json.load(f)

        result = {}
        for model_name in model_list:
            total_fun_samples = 0
            fun_success_samples = 0
            total_class_samples = 0
            class_success_samples = 0
            fun_partial_success_samples = 0
            class_partial_success_samples = 0

            for task in test_result[model_name]:
                for test_class in test_result[model_name][task]:
                    try:
                        # Function-level: calculate avg pass@1 by examining each sample
                        if "EachTestResult" in test_result[model_name][task][test_class]:
                            each_results = test_result[model_name][task][test_class]["EachTestResult"]
                            for sample_result in each_results:
                                total_fun_samples += 1
                                if sample_result == "success":
                                    fun_success_samples += 1
                                    fun_partial_success_samples += 1  # success implies partial success
                                elif sample_result == "partial_success":
                                    fun_partial_success_samples += 1

                        # Class-level: examine ClassEachTestResult
                        if "ClassEachTestResult" in test_result[model_name][task][test_class]:
                            class_results = test_result[model_name][task][test_class]["ClassEachTestResult"]
                            for sample_result in class_results:
                                total_class_samples += 1
                                if sample_result == "class_success":
                                    class_success_samples += 1
                                    class_partial_success_samples += 1  # success implies partial success
                                elif sample_result == "class_partial_success":
                                    class_partial_success_samples += 1

                    except KeyError:
                        # Handle TestClass case - it contains ClassEachTestResult
                        if test_class == "TestClass" and "ClassEachTestResult" in test_result[model_name][task][
                            test_class]:
                            class_results = test_result[model_name][task][test_class]["ClassEachTestResult"]
                            for sample_result in class_results:
                                total_class_samples += 1
                                if sample_result == "class_success":
                                    class_success_samples += 1
                                    class_partial_success_samples += 1  # success implies partial success
                                elif sample_result == "class_partial_success":
                                    class_partial_success_samples += 1
                        continue

            # Calculate averages
            class_success_rate = class_success_samples / total_class_samples if total_class_samples > 0 else 0
            class_partial_rate = class_partial_success_samples / total_class_samples if total_class_samples > 0 else 0
            fun_success_rate = fun_success_samples / total_fun_samples if total_fun_samples > 0 else 0
            fun_partial_rate = fun_partial_success_samples / total_fun_samples if total_fun_samples > 0 else 0

            logger.info(
                f"Avg Pass@1 for {model_name}: {total_class_samples} class samples, {total_fun_samples} function samples")

            result[model_name] = {
                "class_success": class_success_rate,
                "class_partial_success": class_partial_rate,
                "fun_success": fun_success_rate,
                "fun_partial_success": fun_partial_rate,
            }
        return result

    def clean_leaked_files(self, project_root, initial_files):
        """Clean up temporary files leaked to project root directory"""
        current_files = set(os.listdir(project_root))
        leaked_files = current_files - initial_files

        for file in leaked_files:
            file_path = os.path.join(project_root, file)
            try:
                # Only delete obvious temporary files, avoid deleting important files
                if (file.endswith(('.xml', '.csv', '.db', '.sqlite', '.json', '.zip', '.txt', '.xlsx'))
                        and not file.startswith(('README', 'config', 'run', 'main', 'inference'))):
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                    logger.info(f"Cleaned leaked file: {file}")
            except:
                pass

    def tear_down(self):
        """Clean up temporary files"""
        if os.path.exists(self.tmp_dir):
            # Clean all temporary files
            for file in os.listdir(self.tmp_dir):
                file_path = os.path.join(self.tmp_dir, file)
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                except:
                    pass


def run_evaluation(model_result_file, eval_data_file, model_name):
    """
    Run evaluation pipeline for a model

    Args:
        model_result_file: Path to model results JSON file
        eval_data_file: Path to evaluation data JSON file
        model_name: Name of the model

    Returns:
        Dictionary containing evaluation results
    """
    base_dir = os.path.dirname(os.path.abspath(model_result_file))

    logger.info("=" * 60)
    logger.info("Starting ClassEval evaluation...")
    logger.info("=" * 60)
    logger.info(f"Model result file: {model_result_file}")
    logger.info(f"Evaluation data: {eval_data_file}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Base directory: {base_dir}")
    logger.info("=" * 60)

    # Initialize testing
    auto_test = AutoTest(eval_data_file, base_dir)
    model_list = [model_name]

    # Run test pipeline
    logger.info(f"Running test pipeline for model: {model_name}")
    result = auto_test.test_pipeline(model_name, model_result_file)

    # Evaluate results
    logger.info("Evaluating results...")
    auto_test.evaluate(model_list)

    # Calculate Pass@K metrics based on available solutions
    result = {}
    min_solutions = auto_test.min_solutions

    if min_solutions >= 1:
        result["pass_1"] = auto_test.cal_metrics_pass_at_k(model_list, 1, min_solutions)
    if min_solutions >= 3:
        result["pass_3"] = auto_test.cal_metrics_pass_at_k(model_list, 3, min_solutions)
    if min_solutions >= 5:
        result["pass_5"] = auto_test.cal_metrics_pass_at_k(model_list, 5, min_solutions)

    # Calculate average pass@1 based on the highest available k
    result["avg_pass_1"] = auto_test.cal_avg_pass_at_1(model_list, min_solutions)
    logger.info(f"Calculated avg_pass@1 based on max k={min_solutions} samples")

    # Save final results
    save_path = auto_test.path_util.get_output_file("pass_at_k_result")

    if os.path.exists(save_path):
        with open(save_path, encoding='utf-8') as file:
            ori_data = json.load(file)

        # Update results for each available pass@k
        for pass_k in result:
            if pass_k in ori_data:
                ori_data[pass_k][model_name] = result[pass_k][model_name]
            else:
                ori_data[pass_k] = result[pass_k]
    else:
        ori_data = result

    with open(save_path, 'w+') as f:
        json.dump(ori_data, f, indent=4, sort_keys=True)

    logger.info(f"Evaluation completed! Results saved to: {save_path}")

    # Log result summary
    logger.info("=" * 60)
    logger.info("Evaluation Results Summary")
    logger.info("=" * 60)
    for model_name in model_list:
        logger.info(f"Model: {model_name}")
        for pass_k in sorted(result.keys()):
            if pass_k == "avg_pass_1":
                k_value = f"Avg 1 (based on k={min_solutions})"
            else:
                k_value = pass_k.split('_')[1]  # Extract k from "pass_k"
            metrics = result[pass_k][model_name]
            logger.info(f"  Pass@{k_value}:")
            logger.info(
                f"    Class Success Rate: {metrics['class_success']:.4f} ({metrics['class_success'] * 100:.2f}%)")
            logger.info(
                f"    Class Partial Success Rate: {metrics['class_partial_success']:.4f} ({metrics['class_partial_success'] * 100:.2f}%)")
            logger.info(
                f"    Function Success Rate: {metrics['fun_success']:.4f} ({metrics['fun_success'] * 100:.2f}%)")
            logger.info(
                f"    Function Partial Success Rate: {metrics['fun_partial_success']:.4f} ({metrics['fun_partial_success'] * 100:.2f}%)")
        logger.info("=" * 60)

    # remove the tmp directory
    shutil.rmtree(auto_test.tmp_dir)

    return result


def main(args):
    # For backward compatibility
    base_dir = os.path.dirname(os.path.abspath(args.source_file_name))

    # Support both full path and filename (for backward compatibility)
    model_result_file = args.source_file_name if args.source_file_name.endswith(
        '.json') else f"{args.source_file_name}.json"
    eval_data_file = args.eval_data if args.eval_data.endswith('.json') else f"{args.eval_data}.json"

    # Extract model name from file path
    model_name = os.path.splitext(os.path.basename(args.source_file_name))[0]

    run_evaluation(
        model_result_file=model_result_file,
        eval_data_file=eval_data_file,
        model_name=model_name
    )


def summarize_results(pattern):
    possible_folders = glob.glob(pattern)
    all_results = {}

    for folder in possible_folders:
        result_file = os.path.join(folder, "pass_at_k_result.json")
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for pass_k, pass_data in data.items():
                    if pass_k not in all_results:
                        all_results[pass_k] = {}
                    for model_name, metrics in pass_data.items():
                        # use the folder name as the model name
                        model_name = os.path.basename(folder)
                        all_results[pass_k][model_name] = metrics
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from {result_file}. Skipping.")
            except Exception as e:
                logger.error(f"Error processing {result_file}: {e}")

    # Calculate average pass@1 for each model-temperature combination
    def calculate_avg_pass_1(all_results):
        # Prefer avg_pass_1 if available, otherwise fall back to pass_1
        if 'avg_pass_1' in all_results:
            pass_1_data = all_results['avg_pass_1']
        elif 'pass_1' not in all_results:
            return {}
        else:
            pass_1_data = all_results['pass_1']

        avg_results = {}

        # Create simplified model names: model_temperature format
        for model_name, metrics in pass_1_data.items():
            parts = model_name.split('_')
            if len(parts) >= 4:  # <model>_20250616_120837_0.2
                base_name = parts[0]  # e.g., the model/run base name
                temp = parts[-1]  # e.g., "0.2"
                simplified_name = f"{base_name}_{temp}"
                avg_results[simplified_name] = metrics

        return avg_results

    output_dir = pattern.split("/")[0]
    Path(output_dir).mkdir(exist_ok=True)
    log_file_path = os.path.join(output_dir, "results.log")

    with open(log_file_path, 'w+', encoding='utf-8') as log_file:
        logger.info("=" * 80)
        logger.info("Aggregated Evaluation Results Summary")
        logger.info("=" * 80)
        log_file.write("Aggregated Evaluation Results Summary\n")
        log_file.write("=" * 80 + "\n")

        for pass_k, pass_data in sorted(all_results.items()):
            k_value = pass_k.split('_')[1]
            table = PrettyTable()
            table.title = f"Pass@{k_value} Results"
            table.field_names = ["Model", "Class Success", "Class Partial", "Func Success", "Func Partial"]

            for model_name, metrics in sorted(pass_data.items()):
                table.add_row([
                    model_name,
                    f"{metrics.get('class_success', 0) * 100:.1f}%",
                    f"{metrics.get('class_partial_success', 0) * 100:.1f}%",
                    f"{metrics.get('fun_success', 0) * 100:.1f}%",
                    f"{metrics.get('fun_partial_success', 0) * 100:.1f}%",
                ])

            table_string = str(table)
            log_file.write(table_string)
            log_file.write("\n\n")
            print(table_string)

        # Add average pass@1 table (prioritizing avg_pass_1 if available)
        avg_pass_1 = calculate_avg_pass_1(all_results)
        if avg_pass_1:
            data_source = "avg_pass_1" if "avg_pass_1" in all_results else "pass_1"
            table = PrettyTable()
            table.title = f"Average Pass@1 Results (from {data_source})"
            table.field_names = ["Model", "Class Success", "Class Partial", "Func Success", "Func Partial"]

            for model_name, metrics in sorted(avg_pass_1.items()):
                table.add_row([
                    model_name,
                    f"{metrics.get('class_success', 0) * 100:.1f}%",
                    f"{metrics.get('class_partial_success', 0) * 100:.1f}%",
                    f"{metrics.get('fun_success', 0) * 100:.1f}%",
                    f"{metrics.get('fun_partial_success', 0) * 100:.1f}%",
                ])

            table_string = str(table)
            log_file.write(table_string)
            log_file.write("\n\n")
            print(table_string)

    logger.info(f"Aggregated results saved to {log_file_path}")


def test_all_folders(pattern="output/*"):
    possible_folders = glob.glob(pattern)
    processes = []

    logger.info(f"Found {len(possible_folders)} folders to process based on pattern: '{pattern}'")

    for folder in possible_folders:
        model_name = os.path.basename(folder)
        model_result_file = os.path.join(folder, "results.json")
        eval_data_file = "data/ClassEval_data.json"

        if not os.path.exists(model_result_file):
            logger.warning(f"results.json not found in {folder}, skipping.")
            continue

        logger.info(f"Spawning evaluation process for model: {model_name}")

        p = None
        # This function was commented out for safety, but we could implement multiprocessing here

        processes.append(p)

    return processes


def summarize_results_of_target(pattern):
    # This function is kept for reference but currently not implemented
    logger.info("Target results summarization not implemented")
    return {}


def _cli():
    parser = argparse.ArgumentParser(description='ClassEval Testing Tool')
    parser.add_argument(
        "--source_file_name",
        type=str,
        default="output/results.json",
        help="Model output file path (can be full path with .json extension or just filename)"
    )

    parser.add_argument(
        "--eval_data",
        type=str,
        default='data/ClassEval_data.json',
        help="Evaluation dataset path (can be full path with .json extension or just filename)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default='default_model',
        help="Model name for evaluation"
    )

    args = parser.parse_args()

    # Run evaluation for a single file
    test_metrics = run_evaluation(
        model_result_file=args.source_file_name,
        eval_data_file=args.eval_data,
        model_name=args.model_name
    )

    if test_metrics:
        logger.info("Testing completed successfully!")


if __name__ == '__main__':
    _cli()
