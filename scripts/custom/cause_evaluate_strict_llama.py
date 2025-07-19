import json
import os
import re
import sys
from collections import defaultdict

# 增加递归深度限制
sys.setrecursionlimit(10000)


def clean_text(text):
    """清理文本中的特殊标记和换行符"""
    # 去除所有特殊标记
    special_tokens = [
        r'<\|begin_of_text\|>',
        r'<\|start_header_id\|>',
        r'<\|end_header_id\|>',
        r'<\|eot_id\|>',
        'system\n\n',
        'user\n\n',
        'assistant\n\n'
    ]

    for token in special_tokens:
        text = re.sub(token, '', text)

    # 如果存在明确的JSON部分（使用```包围），提取它
    json_match = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # 移除多余的换行符和空格
    text = re.sub(r'\s+', ' ', text).strip()

    # 尝试找到JSON开始和结束的位置
    json_start = text.find('{')
    json_end = text.rfind('}')

    if json_start >= 0 and json_end > json_start:
        text = text[json_start:json_end + 1]

    return text


def normalize_value(value):
    """标准化字段值，用于严格模式比较"""
    if value is None or value == "":
        return "NAN"
    # 移除多余空格和换行符，标准化为单一格式
    return re.sub(r'\s+', ' ', str(value)).strip()


def extract_events(json_str):
    """安全地解析JSON字符串并提取事件列表，避免递归问题"""
    # 使用非递归方式处理
    try:
        # 尝试处理字符串中的转义问题
        if isinstance(json_str, str):
            # 检查是否已经是有效JSON
            try:
                data = json.loads(json_str)
            except:
                # 可能需要额外清理
                json_str = clean_text(json_str)

                # 尝试提取有效的JSON部分
                json_start = json_str.find('{')
                json_end = json_str.rfind('}')

                if json_start >= 0 and json_end > json_start:
                    json_str = json_str[json_start:json_end + 1]

                # 使用标准库解析，不依赖递归
                try:
                    data = json.loads(json_str)
                except:
                    print(f"Failed to parse JSON after cleaning: {json_str[:100]}...")
                    return []
        else:
            data = json_str  # 已经是解析过的对象

        # 提取事件列表
        events = []
        if isinstance(data, dict) and "causality_list" in data:
            causality_list = data["causality_list"]
            if isinstance(causality_list, list):
                for event_pair in causality_list:
                    if isinstance(event_pair, dict) and "cause" in event_pair and "effect" in event_pair:
                        cause = event_pair["cause"]
                        effect = event_pair["effect"]

                        if not isinstance(cause, dict) or not isinstance(effect, dict):
                            continue

                        # 规范化字段
                        normalized_cause = {}
                        normalized_effect = {}

                        for field in ["subject", "event_type", "trigger", "object", "date", "location"]:
                            normalized_cause[field] = normalize_value(cause.get(field, "NAN"))
                            normalized_effect[field] = normalize_value(effect.get(field, "NAN"))

                        events.append({
                            "cause": normalized_cause,
                            "effect": normalized_effect
                        })

        return events
    except Exception as e:
        print(f"Error in extract_events: {e}")
        return []


def evaluate(jsonl_path):
    """严格模式评估：读取jsonl文件，对每行提取predict与label字段，
    按各事件要素完全匹配(而非计算子串相似度)来判断TP/FP/FN，
    并计算各要素和微平均精确率/召回率/F1值
    """
    with open(jsonl_path, encoding='utf-8') as f:
        lines = f.readlines()

    # 用于统计每个(事件类型, 要素)的TP, FP, FN
    per_field_scores = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    total_tp, total_fp, total_fn = 0, 0, 0
    fields = ["subject", "event_type", "trigger", "object", "date", "location"]

    processed_count = 0
    error_count = 0

    for line_num, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
            processed_count += 1
        except json.JSONDecodeError:
            print(f"Error parsing line {line_num}")
            error_count += 1
            continue

        # 提取和清理predict和label
        predict_raw = data.get("predict", {})
        label_raw = data.get("label", {})

        # 提取事件数据
        predict_events = extract_events(predict_raw)
        label_events = extract_events(label_raw)

        # 针对每个字段，构建(事件类型, 字段, 值)的集合
        for field in fields:
            # 从标准答案中提取字段值
            label_set = set()
            for event_pair in label_events:
                if "cause" in event_pair and "effect" in event_pair:
                    cause = event_pair["cause"]
                    effect = event_pair["effect"]

                    if field in cause:
                        value = normalize_value(cause[field])
                        if value != "NAN":  # 可选：忽略NAN值
                            label_set.add(("cause", field, value))

                    if field in effect:
                        value = normalize_value(effect[field])
                        if value != "NAN":  # 可选：忽略NAN值
                            label_set.add(("effect", field, value))

            # 从预测结果中提取字段值
            predict_set = set()
            for event_pair in predict_events:
                if "cause" in event_pair and "effect" in event_pair:
                    cause = event_pair["cause"]
                    effect = event_pair["effect"]

                    if field in cause:
                        value = normalize_value(cause[field])
                        if value != "NAN":  # 可选：忽略NAN值
                            predict_set.add(("cause", field, value))

                    if field in effect:
                        value = normalize_value(effect[field])
                        if value != "NAN":  # 可选：忽略NAN值
                            predict_set.add(("effect", field, value))

            # 计算TP, FP, FN (严格匹配模式)
            tp_set = label_set.intersection(predict_set)
            fp_set = predict_set - label_set
            fn_set = label_set - predict_set

            # 更新每个(事件类型,字段)的统计数据
            for etype, field_name, _ in tp_set:
                per_field_scores[(etype, field_name)]["TP"] += 1
                total_tp += 1

            for etype, field_name, _ in fp_set:
                per_field_scores[(etype, field_name)]["FP"] += 1
                total_fp += 1

            for etype, field_name, _ in fn_set:
                per_field_scores[(etype, field_name)]["FN"] += 1
                total_fn += 1

        # 每处理100行打印一次进度
        if line_num % 100 == 0:
            print(f"Processed {line_num} lines...")

    # 汇总各字段分数，并按cause与effect分开存放
    results = {"cause": {}, "effect": {}}

    for (etype, field), scores in per_field_scores.items():
        TP, FP, FN = scores["TP"], scores["FP"], scores["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if etype not in results:
            results[etype] = {}

        results[etype][field] = {
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1-score": round(f1_score * 100, 2),
            "TP": TP,
            "FP": FP,
            "FN": FN
        }

    # 计算微平均分数
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (
                                                                                                  micro_precision + micro_recall) > 0 else 0

    results["Final_Score"] = {
        "Micro-Precision": round(micro_precision * 100, 2),
        "Micro-Recall": round(micro_recall * 100, 2),
        "Micro-F1": round(micro_f1 * 100, 2),
        "Total_TP": total_tp,
        "Total_FP": total_fp,
        "Total_FN": total_fn,
        "Processed_Lines": processed_count,
        "Error_Lines": error_count
    }

    # 保存结果
    output_path = os.path.join(os.path.dirname(jsonl_path), "evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Evaluation completed.")
    print(f"Processed {processed_count} lines, encountered {error_count} errors.")
    print(f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")
    print(f"Micro-F1: {round(micro_f1 * 100, 2)}%")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # 命令行参数接收文件路径
    if len(sys.argv) > 1:
        jsonl_file_path = sys.argv[1]
    else:
        jsonl_file_path = "saves/Llama/llama3-8B-Instruct-cause/qlora/predict/generated_predictions.jsonl"

    print(f"Evaluating file: {jsonl_file_path}")
    evaluate(jsonl_file_path)
