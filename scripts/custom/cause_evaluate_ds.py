import json
import os
import re
from collections import defaultdict


def longest_common_substring(s1, s2):
    """计算两个字符串之间的最长公共子串长度"""
    if not s1 or not s2:
        return 0
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    max_length = 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
    return max_length


def consistency(s1, s2):
    # 将 None 转为空字符串
    s1 = s1 or ""
    s2 = s2 or ""
    llcs = longest_common_substring(s1, s2)
    total_len = len(s1) + len(s2)
    return (2 * llcs) / total_len if total_len > 0 else 0


def clean_text(text):
    """清理文本，去除各种标记"""
    # 去除所有<｜...｜>标记
    text = re.sub(r"<｜.*?｜>", "", text)
    # 处理JSON字符串末尾可能的额外内容
    if "<" in text:
        text = text.split("<")[0]
    return text.strip()


def extract_events(json_str):
    """解析JSON字符串并提取事件列表"""
    try:
        # 尝试直接解析
        data = json.loads(json_str)
        return data.get("causality_list", [])
    except json.JSONDecodeError:
        try:
            # 如果失败，尝试清理后再解析
            cleaned = clean_text(json_str)
            data = json.loads(cleaned)
            return data.get("causality_list", [])
        except:
            return []


def evaluate(jsonl_path):
    """读取 jsonl 文件，对每一行提取 predict 与 label 字段，
    按各个事件要素（subject、event_type、trigger、object、date、location）
    基于一致性阈值 0.7 判断匹配情况，并统计 TP、FP、FN，最终计算各要素和微平均得分
    """
    with open(jsonl_path, encoding='utf-8') as f:
        lines = f.readlines()

    # 用于统计每个 (事件类型, 要素) 的 TP、FP、FN
    per_field_scores = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    total_tp, total_fp, total_fn = 0, 0, 0
    fields = ["subject", "event_type", "trigger", "object", "date", "location"]

    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        predict_raw = data.get("predict", "")
        label_raw = data.get("label", "")
        # 清理无关标记
        predict_events = extract_events(clean_text(predict_raw))
        label_events = extract_events(clean_text(label_raw))

        # 针对每个字段，构建 (事件类型, 字段, 值) 的集合，
        # 仅当事件为字典时才取字段值
        for field in fields:
            label_set = {
                (etype, field, event.get(field, ""))
                for event_pair in label_events
                for etype, event in event_pair.items()
                if isinstance(event, dict)
            }
            predict_set = {
                (etype, field, event.get(field, ""))
                for event_pair in predict_events
                for etype, event in event_pair.items()
                if isinstance(event, dict)
            }

            # 对标准答案中的每个要素，查找预测中是否有匹配（一致性大于 0.7）
            for etype, field, label_value in label_set:
                matched = False
                for _, _, pred_value in predict_set:
                    if consistency(label_value, pred_value) > 0.7:
                        per_field_scores[(etype, field)]["TP"] += 1
                        matched = True
                        break
                if not matched:
                    per_field_scores[(etype, field)]["FN"] += 1

            # 对于预测中的每个要素，若找不到匹配的标准答案则记 FP
            for etype, field, pred_value in predict_set:
                if all(consistency(pred_value, label_value) <= 0.7 for _, _, label_value in label_set):
                    per_field_scores[(etype, field)]["FP"] += 1

    # 汇总各字段分数，并按 cause 与 effect 分开存放
    results = {"cause": {}, "effect": {}}
    for (etype, field), scores in per_field_scores.items():
        TP, FP, FN = scores["TP"], scores["FP"], scores["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        if etype not in results:
            results[etype] = {}
        results[etype][field] = {"Precision": precision, "Recall": recall, "F1-score": f1_score}
        total_tp += TP
        total_fp += FP
        total_fn += FN

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (
                                                                                                      micro_precision + micro_recall) > 0 else 0
    results["Final_Score"] = {"Micro-F1": micro_f1}

    output_path = os.path.join(os.path.dirname(jsonl_path), "evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # 修改此处为你的 jsonl 文件路径
    jsonl_file_path = "saves/Qwen/qwen2.5-14B-Instruct/qlora/generated_predictions.jsonl"
    evaluate(jsonl_file_path)
