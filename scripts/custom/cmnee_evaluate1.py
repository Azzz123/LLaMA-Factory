import json


def load_valid_jsonl(file_path):
    """逐行读取 JSONL 文件，跳过无法解析整行 JSON，
    或predict/label字段解析失败的样本。
    同时对predict和label字段预处理，移除末尾的特殊标记。
    """
    valid_samples = []
    skipped_lines = 0
    end_marker = "<｜end▁of▁sentence｜>"
    with open(file_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                print(f"行 {line_number}: 无法解析整行 JSON，跳过。")
                skipped_lines += 1
                continue
            # 检查是否存在 predict 和 label 字段
            if "predict" not in sample or "label" not in sample:
                print(f"行 {line_number}: 缺少 predict 或 label 字段，跳过。")
                skipped_lines += 1
                continue

            # 预处理：移除predict和label字段末尾可能存在的结束标记
            pred_str = sample["predict"].strip()
            label_str = sample["label"].strip()
            if pred_str.endswith(end_marker):
                pred_str = pred_str[:-len(end_marker)].strip()
            if label_str.endswith(end_marker):
                label_str = label_str[:-len(end_marker)].strip()
            try:
                json.loads(pred_str)
                json.loads(label_str)
            except json.JSONDecodeError:
                print(f"行 {line_number}: predict 或 label 字段无法解析为 JSON，跳过。")
                skipped_lines += 1
                continue
            # 更新字段为处理后的字符串
            sample["predict"] = pred_str
            sample["label"] = label_str
            valid_samples.append(sample)
    return valid_samples, skipped_lines


def extract_events(event_dict):
    """解析事件字典，提取格式为 (event_type, trigger, arguments_set)
    若事件数据不是列表，则转换为列表。
    arguments_set 是一个 set，每个元素为 (role, text) 对。
    """
    events = []
    for event_type, event_list in event_dict.items():
        # 如果事件数据不是列表，则转换成列表
        if not isinstance(event_list, list):
            event_list = [event_list]
        for event in event_list:
            if not event:
                continue
            trigger = event.get("trigger", "")
            arguments = event.get("arguments", {})
            args_set = set()
            for role, texts in arguments.items():
                if isinstance(texts, list):
                    for text in texts:
                        args_set.add((role, text))
                else:
                    args_set.add((role, texts))
            events.append((event_type, trigger, args_set))
    return events


def compute_f1(correct, predicted, total_gold):
    """根据给定的正确数、预测数和标注数计算 Precision, Recall 和 F1"""
    precision = correct / predicted if predicted > 0 else 0.0
    recall = correct / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate(jsonl_file):
    data, skipped_lines = load_valid_jsonl(jsonl_file)
    total_lines = sum(1 for _ in open(jsonl_file, encoding="utf-8"))
    valid_samples_count = len(data)
    print(f"总行数：{total_lines}，成功解析的行数：{valid_samples_count}，跳过的行数：{skipped_lines}")

    # 初始化计数器
    correct_triggers = 0
    total_pred_triggers = 0
    total_gold_triggers = 0

    correct_args = 0
    total_pred_args = 0
    total_gold_args = 0

    for sample in data:
        try:
            pred_json = json.loads(sample["predict"])
            label_json = json.loads(sample["label"])
        except json.JSONDecodeError:
            continue

        pred_events = extract_events(pred_json)
        gold_events = extract_events(label_json)

        total_pred_triggers += len(pred_events)
        total_gold_triggers += len(gold_events)

        # 评估触发词：逐个匹配 gold 事件，匹配到第一个未使用的预测事件后即算作正确
        matched_pred_indices = set()
        for gold_event in gold_events:
            for i, pred_event in enumerate(pred_events):
                if i in matched_pred_indices:
                    continue
                if gold_event[:2] == pred_event[:2]:  # event_type 和 trigger 均匹配
                    correct_triggers += 1
                    matched_pred_indices.add(i)
                    break

        # 评估论元：对每个 gold 事件，从所有预测事件中找到同一 (event_type, trigger) 下论元匹配数最多的预测事件
        for gold_event in gold_events:
            best_match_args = 0
            for pred_event in pred_events:
                if pred_event[:2] == gold_event[:2]:
                    correct_count = len(pred_event[2] & gold_event[2])
                    if correct_count > best_match_args:
                        best_match_args = correct_count
            correct_args += best_match_args

        total_pred_args += sum(len(event[2]) for event in pred_events)
        total_gold_args += sum(len(event[2]) for event in gold_events)

    trg_precision, trg_recall, trg_f1 = compute_f1(correct_triggers, total_pred_triggers, total_gold_triggers)
    arg_precision, arg_recall, arg_f1 = compute_f1(correct_args, total_pred_args, total_gold_args)
    overall_f1 = (trg_f1 + arg_f1) / 2

    result_str = (
        f"总行数：{total_lines}\n"
        f"成功解析的行数：{valid_samples_count}\n"
        f"跳过的行数：{skipped_lines}\n\n"
        f"Trg-F1: Precision={trg_precision:.4f}, Recall={trg_recall:.4f}, F1={trg_f1:.4f}\n"
        f"Arg-F1: Precision={arg_precision:.4f}, Recall={arg_recall:.4f}, F1={arg_f1:.4f}\n"
        f"综合F1: {overall_f1:.4f}\n"
    )

    print(result_str)

    # 将结果保存到输入文件同一路径下，文件名后加 .evaluation.txt 后缀
    output_file = jsonl_file + ".evaluation.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result_str)
    print(f"评估结果已保存至 {output_file}")


if __name__ == "__main__":
    import sys

    evaluate(sys.argv[1])
