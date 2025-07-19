import json
import os
import re


def longest_common_substring(s1, s2):
    """计算两个字符串之间的最长公共子串（连续）的长度
    """
    if not s1 or not s2:
        return 0
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    max_len = 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
    return max_len


def consistency(s1, s2):
    """计算一致性：Consis(S, T) = (2 * LLCS(S, T)) / (|S| + |T|)
    如果某个值为 None，则当作空字符串处理
    """
    s1 = s1 or ""
    s2 = s2 or ""
    llcs = longest_common_substring(s1, s2)
    total_len = len(s1) + len(s2)
    return (2 * llcs) / total_len if total_len > 0 else 0


def clean_text(text):
    """清理文本，去除形如 <｜...｜> 的标记
    """
    return re.sub(r"<｜.*?｜>", "", text)


def extract_event_elements(event, event_type):
    """从单个事件中提取触发词和论元
    event 格式例如：
      {"trigger": "部署", "arguments": {"主体": ["美国"], "军事力量": ["B-2", "B-52G"]}}
    返回 (trigger, arguments_list)
    """
    trigger = event.get("trigger", "")
    arguments = []
    if "arguments" in event and isinstance(event["arguments"], dict):
        for arg_key, arg_list in event["arguments"].items():
            if isinstance(arg_list, list):
                for a in arg_list:
                    # 转为字符串（防止异常）
                    arguments.append(str(a))
    return str(trigger), arguments


def evaluate(jsonl_path):
    """读取 jsonl 文件，对每一行数据进行以下操作：
      - 解析 predict 和 label 字段（若解析异常则跳过该行，并记录到 error_lines 和 error_info 中）
      - 从每个事件类型中提取所有触发词和论元（预测和标准答案分别聚合）
      - 对于每个标准答案的触发词/论元，若存在预测值与其一致性 Consis > 0.7，则认为匹配（TP），否则为 FN
      - 对于预测中的每个触发词/论元，若不存在标准答案中与其一致性大于 0.7 的值，则记 FP
    最后计算触发词、论元的 Precision、Recall、F1，以及微平均（Micro-Average）的综合 F1。
    同时，将无法解析或解析错误的行记录在 error_lines 中，并将错误信息记录在 error_info 中。
    """
    # 初始化计数
    TP_trigger, FP_trigger, FN_trigger = 0, 0, 0
    TP_argument, FP_argument, FN_argument = 0, 0, 0

    total_lines = 0
    valid_lines = 0
    error_lines = []  # 用于记录解析错误的行
    error_info = []  # 记录错误信息，包含原始行和错误描述

    with open(jsonl_path, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        total_lines += 1
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            error_lines.append(line)
            error_info.append({"line": line, "error": str(e)})
            continue

        # 获取 predict 与 label 字段（均为字符串形式的 JSON）
        predict_raw = data.get("predict", "")
        label_raw = data.get("label", "")
        # 清理无关标记
        predict_clean = clean_text(predict_raw)
        label_clean = clean_text(label_raw)

        try:
            predict_data = json.loads(predict_clean)
            label_data = json.loads(label_clean)
        except json.JSONDecodeError as e:
            error_lines.append(line)
            error_info.append({"line": line, "error": str(e)})
            continue

        # 要求数据为字典格式，键为事件类型
        if not isinstance(predict_data, dict) or not isinstance(label_data, dict):
            continue

        valid_lines += 1

        # 聚合预测和标准答案中的触发词及论元（跨事件类型）
        pred_triggers = []
        pred_arguments = []
        for event_type, events in predict_data.items():
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, dict):
                        trig, args = extract_event_elements(event, event_type)
                        pred_triggers.append(trig)
                        pred_arguments.extend(args)

        true_triggers = []
        true_arguments = []
        for event_type, events in label_data.items():
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, dict):
                        trig, args = extract_event_elements(event, event_type)
                        true_triggers.append(trig)
                        true_arguments.extend(args)

        # 对触发词进行匹配评估：对于每个标准答案的触发词，检查是否存在预测的触发词与其一致性大于 0.7
        for true_trig in true_triggers:
            matched = False
            for pred_trig in pred_triggers:
                if consistency(true_trig, pred_trig) > 0.7:
                    matched = True
                    break
            if matched:
                TP_trigger += 1
            else:
                FN_trigger += 1

        # 对预测触发词进行 FP 统计：如果预测触发词在标准答案中没有匹配项
        for pred_trig in pred_triggers:
            matched = False
            for true_trig in true_triggers:
                if consistency(pred_trig, true_trig) > 0.7:
                    matched = True
                    break
            if not matched:
                FP_trigger += 1

        # 同理，对论元进行匹配评估
        for true_arg in true_arguments:
            matched = False
            for pred_arg in pred_arguments:
                if consistency(true_arg, pred_arg) > 0.7:
                    matched = True
                    break
            if matched:
                TP_argument += 1
            else:
                FN_argument += 1

        for pred_arg in pred_arguments:
            matched = False
            for true_arg in true_arguments:
                if consistency(pred_arg, true_arg) > 0.7:
                    matched = True
                    break
            if not matched:
                FP_argument += 1

    # 计算评价指标
    def calc_scores(TP, FP, FN):
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    trig_precision, trig_recall, trig_f1 = calc_scores(TP_trigger, FP_trigger, FN_trigger)
    arg_precision, arg_recall, arg_f1 = calc_scores(TP_argument, FP_argument, FN_argument)

    # 微平均：触发词和论元合并计算
    total_TP = TP_trigger + TP_argument
    total_FP = FP_trigger + FP_argument
    total_FN = FN_trigger + FN_argument
    micro_precision, micro_recall, micro_f1 = calc_scores(total_TP, total_FP, total_FN)

    results = {
        "Trigger": {
            "Precision": trig_precision,
            "Recall": trig_recall,
            "F1-score": trig_f1,
            "TP": TP_trigger,
            "FP": FP_trigger,
            "FN": FN_trigger
        },
        "Argument": {
            "Precision": arg_precision,
            "Recall": arg_recall,
            "F1-score": arg_f1,
            "TP": TP_argument,
            "FP": FP_argument,
            "FN": FN_argument
        },
        "Micro_Average": {
            "Precision": micro_precision,
            "Recall": micro_recall,
            "F1-score": micro_f1,
            "TP": total_TP,
            "FP": total_FP,
            "FN": total_FN
        },
        "Total_lines": total_lines,
        "Valid_lines": valid_lines
    }
    return results, error_lines, error_info


def main(input_path, output_path):
    results, error_lines, error_info = evaluate(input_path)
    # 保存评价结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    # 构造 error_lines.jsonl 的输出路径（与 output_path 同目录）
    error_lines_path = os.path.join(os.path.dirname(output_path), "error_lines.jsonl")
    # 保存解析错误的行，每行写入原始文本
    with open(error_lines_path, 'w', encoding='utf-8') as f:
        for err_line in error_lines:
            f.write(err_line)
    # 构造 error_info.json 的输出路径（与 output_file 同目录）
    error_info_path = os.path.join(os.path.dirname(output_path), "error_info.json")
    # 保存错误信息列表
    with open(error_info_path, 'w', encoding='utf-8') as f:
        json.dump(error_info, f, ensure_ascii=False, indent=4)
    print(f"Evaluation results saved to {output_path}")
    print(f"Error lines saved to {error_lines_path}")
    print(f"Error info saved to {error_info_path}")


if __name__ == "__main__":
    # 请修改以下文件路径为你本地实际的路径
    input_file = "saves/DeepSeek/DeepSeek-R1-Distill-Llama-8B/un-ft/predict/generated_predictions.jsonl"
    output_file = "saves/DeepSeek/DeepSeek-R1-Distill-Llama-8B/un-ft/predict/evaluation_results.json"
    main(input_file, output_file)
