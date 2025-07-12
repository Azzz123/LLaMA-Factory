import json
import argparse
import os
from collections import defaultdict, Counter


def diff_events(pred_event, gold_event):
    """
    精细化地比较两个事件对象的差异，并返回一个描述差异的列表。
    """
    diffs = []
    # 1. 比较 trigger
    if pred_event.get('trigger') != gold_event.get('trigger'):
        diffs.append(f"Trigger Mismatch: pred='{pred_event.get('trigger')}' vs gold='{gold_event.get('trigger')}'")

    # 2. 比较 arguments
    pred_args = pred_event.get('arguments', {})
    gold_args = gold_event.get('arguments', {})

    all_arg_keys = set(pred_args.keys()) | set(gold_args.keys())

    for key in all_arg_keys:
        if key not in pred_args:
            diffs.append(f"Missing Argument in Prediction: '{key}' (value should be '{gold_args[key]}')")
        elif key not in gold_args:
            diffs.append(f"Extra Argument in Prediction: '{key}' (value is '{pred_args[key]}')")
        elif pred_args[key] != gold_args[key]:
            diffs.append(f"Argument Mismatch for '{key}': pred='{pred_args[key]}' vs gold='{gold_args[key]}'")

    return diffs


def evaluate_causality_extraction(prediction_file, output_dir):
    """
    计算P/R/F1，将结果保存到文件，并对内容不一致的TP进行详细的、量化的差异分析。
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    malformed_predictions = 0
    content_discrepancy_records = []
    discrepancy_type_counter = Counter()  # <<< MODIFIED: 引入差异类型计数器

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(prediction_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                try:
                    predict_list = json.loads(data.get("predict", "[]"))
                    if not isinstance(predict_list, list): predict_list = []
                except json.JSONDecodeError:
                    malformed_predictions += 1
                    print(
                        f"警告: 第 {line_num} 行的 'predict' 字段不是有效的JSON，已跳过。内容: {data.get('predict', '')[:100]}...")
                    predict_list = []

                label_list = json.loads(data.get("label", "[]"))
                if not isinstance(label_list, list): label_list = []

                predicted_map = {
                    (p["cause"]["event_id"], p["effect"]["event_id"]): p
                    for p in predict_list
                    if isinstance(p.get("cause"), dict) and isinstance(p.get("effect"), dict)
                       and p["cause"].get("event_id") and p["effect"].get("event_id")
                }
                golden_map = {
                    (p["cause"]["event_id"], p["effect"]["event_id"]): p
                    for p in label_list
                    if isinstance(p.get("cause"), dict) and isinstance(p.get("effect"), dict)
                       and p["cause"].get("event_id") and p["effect"].get("event_id")
                }

                predicted_ids, golden_ids = set(predicted_map.keys()), set(golden_map.keys())

                tp_ids = predicted_ids.intersection(golden_ids)
                total_tp += len(tp_ids)
                total_fp += len(predicted_ids - golden_ids)
                total_fn += len(golden_ids - predicted_ids)

                for pair_id in tp_ids:
                    pred_pair, gold_pair = predicted_map[pair_id], golden_map[pair_id]

                    cause_diffs = diff_events(pred_pair['cause'], gold_pair['cause'])
                    effect_diffs = diff_events(pred_pair['effect'], gold_pair['effect'])

                    if cause_diffs or effect_diffs:
                        content_discrepancy_records.append({
                            "line_number": line_num,
                            "mismatched_pair_id": list(pair_id),
                            "differences": {"cause_event_diffs": cause_diffs, "effect_event_diffs": effect_diffs},
                            "original_data": data
                        })

                        # <<< MODIFIED: 统计各类差异的数量
                        all_diffs = cause_diffs + effect_diffs
                        for diff_str in all_diffs:
                            if diff_str.startswith("Argument Mismatch"):
                                error_type = "Argument Mismatch"
                            else:
                                error_type = diff_str.split(':')[0].strip()
                            discrepancy_type_counter[error_type] += 1

            except Exception as e:
                print(f"处理行 {line_num} 时发生严重错误: {e}\n行内容: {line.strip()[:200]}...")
                continue

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # <<< MODIFIED: 构建最终的、包含差异统计的评估结果
    scores = {
        "evaluation_summary": {"precision": round(precision, 4), "recall": round(recall, 4), "f1_score": round(f1, 4)},
        "statistics": {"true_positives": total_tp, "false_positives": total_fp, "false_negatives": total_fn},
        "sanity_check": {
            "malformed_predictions": malformed_predictions,
            "content_discrepancies_in_tps": len(content_discrepancy_records),
            "discrepancy_type_counts": dict(discrepancy_type_counter)  # 将Counter转为普通dict以便JSON序列化
        }
    }

    scores_file_path = os.path.join(output_dir, "evaluation_scores.json")
    with open(scores_file_path, 'w', encoding='utf-8') as f_scores:
        json.dump(scores, f_scores, ensure_ascii=False, indent=4)

    if content_discrepancy_records:
        discrepancy_file_path = os.path.join(output_dir, "content_discrepancies.json")
        with open(discrepancy_file_path, 'w', encoding='utf-8') as f_disc:
            json.dump(content_discrepancy_records, f_disc, ensure_ascii=False, indent=4)

    print("\n--- 评估完成 ---")
    print(f"核心评估指标已保存至: {scores_file_path}")
    if content_discrepancy_records:
        print(f"内容不一致的详细样本已保存至: {discrepancy_file_path}")
    print("\n评估摘要:")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"\n在 {total_tp} 个正确预测中，发现 {len(content_discrepancy_records)} 个内容不一致（抄写错误）。")
    if discrepancy_type_counter:
        print("  详细错误类型统计:")
        for error_type, count in discrepancy_type_counter.items():
            print(f"    - {error_type}: {count} 次")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为因果关系抽取任务计算P/R/F1，并将结果保存到文件。")
    parser.add_argument("--prediction_file", type=str, required=True, help="包含模型预测结果和标签的JSONL文件路径。")
    parser.add_argument("--output_dir", type=str, default=None, help="保存评估结果文件的目录 (默认为预测文件所在目录)。")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.prediction_file))
        if not args.output_dir: args.output_dir = '.'
    evaluate_causality_extraction(args.prediction_file, args.output_dir)