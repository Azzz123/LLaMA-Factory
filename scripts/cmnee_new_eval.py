import json
import os
import re
import sys
from collections import Counter


def longest_common_substring(s1, s2):
    """计算两个字符串之间的最长公共子串（连续）的长度"""
    if not s1 or not s2:
        return 0
    # 将输入统一转为字符串，增加鲁棒性
    s1, s2 = str(s1), str(s2)
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    max_len = 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
    return max_len


def consistency(s1, s2, threshold=0.7):
    """
    计算一致性：Consis(S, T) = (2 * LLCS(S, T)) / (|S| + |T|)
    如果某个值为 None，则当作空字符串处理
    """
    s1 = s1 or ""
    s2 = s2 or ""
    s1, s2 = str(s1), str(s2)
    llcs = longest_common_substring(s1, s2)
    total_len = len(s1) + len(s2)
    score = (2 * llcs) / total_len if total_len > 0 else 0
    return score > threshold


def calculate_metrics(tp, fp, fn):
    """根据给定的TP, FP, FN计算精确率、召回率和F1分数"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def extract_elements_from_list(event_list):
    """
    从事件列表（list of dicts）中提取所有触发词和论元。
    """
    triggers = []
    arguments = []
    if not isinstance(event_list, list):
        return triggers, arguments

    for event in event_list:
        if not isinstance(event, dict):
            continue
        # 添加触发词
        trigger = event.get("trigger")
        if trigger:
            triggers.append(trigger)

        # 添加论元
        args_dict = event.get("arguments", {})
        if isinstance(args_dict, dict):
            for role, values in args_dict.items():
                if isinstance(values, list):
                    arguments.extend(values)
                elif values is not None:
                    arguments.append(values)
    return triggers, arguments


def evaluate_fuzzy(file_path):
    """
    主评估函数，使用基于最长公共子串的模糊匹配来评估。
    """
    total_lines = 0
    valid_samples = []
    error_info = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                pred_str = sample.get("predict")
                label_str = sample.get("label")

                if pred_str is None or label_str is None:
                    error_info.append({"line_num": line_num, "error": "Missing 'predict' or 'label' field"})
                    continue

                pred_json = json.loads(pred_str)
                label_json = json.loads(label_str)

                if not isinstance(pred_json, list) or not isinstance(label_json, list):
                    error_info.append({"line_num": line_num, "error": "'predict' or 'label' is not a list"})
                    continue

                valid_samples.append((pred_json, label_json))

            except (json.JSONDecodeError, TypeError) as e:
                error_info.append({"line_num": line_num, "error": str(e)})
                continue

    if not valid_samples:
        print("没有可供评估的有效数据。")
        return None, error_info

    # 初始化全局计数器
    trg_tp, trg_fp, trg_fn = 0, 0, 0
    arg_tp, arg_fp, arg_fn = 0, 0, 0

    for pred_events, gold_events in valid_samples:
        pred_triggers, pred_arguments = extract_elements_from_list(pred_events)
        gold_triggers, gold_arguments = extract_elements_from_list(gold_events)

        # --- 评估触发词 ---
        # 计算TP和FN
        for gold_trg in gold_triggers:
            if any(consistency(gold_trg, pred_trg) for pred_trg in pred_triggers):
                trg_tp += 1
            else:
                trg_fn += 1
        # 计算FP
        for pred_trg in pred_triggers:
            if not any(consistency(pred_trg, gold_trg) for gold_trg in gold_triggers):
                trg_fp += 1

        # --- 评估论元 ---
        # 计算TP和FN
        for gold_arg in gold_arguments:
            if any(consistency(gold_arg, pred_arg) for pred_arg in pred_arguments):
                arg_tp += 1
            else:
                arg_fn += 1
        # 计算FP
        for pred_arg in pred_arguments:
            if not any(consistency(pred_arg, gold_arg) for gold_arg in gold_arguments):
                arg_fp += 1

    # --- 计算最终指标 ---
    trg_precision, trg_recall, trg_f1 = calculate_metrics(trg_tp, trg_fp, trg_fn)
    arg_precision, arg_recall, arg_f1 = calculate_metrics(arg_tp, arg_fp, arg_fn)

    total_tp = trg_tp + arg_tp
    total_fp = trg_fp + arg_fp
    total_fn = trg_fn + arg_fn
    micro_p, micro_r, micro_f1 = calculate_metrics(total_tp, total_fp, total_fn)

    results = {
        "metadata": {
            "file_path": file_path,
            "total_lines": total_lines,
            "valid_samples": len(valid_samples),
            "error_count": len(error_info)
        },
        "trigger_metrics": {
            "precision": trg_precision, "recall": trg_recall, "f1_score": trg_f1,
            "TP": trg_tp, "FP": trg_fp, "FN": trg_fn
        },
        "argument_metrics": {
            "precision": arg_precision, "recall": arg_recall, "f1_score": arg_f1,
            "TP": arg_tp, "FP": arg_fp, "FN": arg_fn
        },
        "micro_average_metrics": {
            "precision": micro_p, "recall": micro_r, "f1_score": micro_f1,
            "TP": total_tp, "FP": total_fp, "FN": total_fn
        }
    }

    return results, error_info


def main():
    if len(sys.argv) < 2:
        print("用法: python your_script_name.py <input_jsonl_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)

    results_file = os.path.join(output_dir, f"eval_{base_name.replace('.jsonl', '')}.json")
    errors_file = os.path.join(output_dir, f"val_errors_{base_name.replace('.jsonl', '')}.json")

    results, errors = evaluate_fuzzy(input_file)

    if results:
        print("--- 评估结果 ---")
        print(json.dumps(results, indent=4, ensure_ascii=False))
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n评估结果已保存至: {results_file}")

    if errors:
        print(f"\n发现 {len(errors)} 个解析错误。")
        with open(errors_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=4, ensure_ascii=False)
        print(f"详细错误信息已保存至: {errors_file}")


if __name__ == "__main__":
    main()