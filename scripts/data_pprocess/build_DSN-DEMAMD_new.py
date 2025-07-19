import json
import argparse
import re
from collections import OrderedDict, Counter
import random
import os
import numpy as np

# ==============================================================================
# 1. Schema 定义 和 辅助函数
# ==============================================================================

SOURCE_TO_TARGET_MAP = {
    "class": "event_type", "action": "trigger", "actor": "主体",
    "object": "客体", "time": "日期", "location": "位置"
}
CORE_ARGUMENTS = {"主体", "客体", "位置"}


def create_event_structure(event_data, event_id):
    structured_event = OrderedDict([
        ("event_id", event_id), ("event_type", event_data.get('class', '未知类型')),
        ("trigger", event_data.get('action', 'N/A')),
    ])
    arguments = {}
    for source_key, target_key in SOURCE_TO_TARGET_MAP.items():
        if source_key in ['class', 'action']: continue
        value = event_data.get(source_key)
        if value and value.strip(): arguments[target_key] = value
    for key, value in event_data.items():
        if key not in SOURCE_TO_TARGET_MAP and value and value.strip():
            arguments[key] = value
    structured_event["arguments"] = arguments
    return structured_event


def generate_context_snippet(text, event_a, event_b, window=50):
    trigger_a, trigger_b = event_a.get('trigger'), event_b.get('trigger')
    if not trigger_a or not trigger_b: return text[:250].replace('\n', ' ')
    try:
        pos_a = [m.start() for m in re.finditer(re.escape(trigger_a), text)]
        pos_b = [m.start() for m in re.finditer(re.escape(trigger_b), text)]
        if not pos_a or not pos_b: return text[:250].replace('\n', ' ')
        min_dist, best_pair = float('inf'), (pos_a[0], pos_b[0])
        for p_a in pos_a:
            for p_b in pos_b:
                if abs(p_a - p_b) < min_dist:
                    min_dist, best_pair = abs(p_a - p_b), (p_a, p_b)
        start_pos, end_pos = best_pair
        start, end = min(start_pos, end_pos), max(start_pos + len(trigger_a), end_pos + len(trigger_b))
        snippet_start, snippet_end = max(0, start - window), min(len(text), end + window)
        return text[snippet_start:snippet_end].strip().replace('\n', ' ')
    except Exception:
        return text[:250].replace('\n', ' ')


# ==============================================================================
# 2. 主转换函数 (黄金标准版)
# ==============================================================================

def build_gold_standard_dataset(input_file, output_file, sentence_distance_threshold=2, negative_ratio=1.0):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"错误：无法读取或解析输入文件 {input_file}。原因: {e}")
        return

    final_alpaca_data = []
    stats_data = []

    system_prompt = "你是一名专精于识别和推断事件之间因果关系的军事地缘政治分析师。"
    instruction_prompt = (
        "你的任务是作为一名因果关系分析专家，对'input'中的'candidate_pairs'列表进行筛选。"
        "请仔细分析每个候选对中'event_1'和'event_2'的结构化论元信息（arguments），并结合其'context_snippet'提供的文本线索以及'text'中的全局上下文，判断它们之间是否存在因果关系。"
        "最终，请以JSON列表的格式，输出所有你确认为真实因果关系的事件对。每个因果对都必须包含'cause'和'effect'两个键。"
    )

    for original_entry in original_data:
        text = original_entry.get('text', '')
        causality_list_raw = original_entry.get('causality_list', [])
        doc_id = original_entry.get('document_id', 'N/A')
        if not text: continue

        all_unique_events_raw = {frozenset(e.items()): e for p in causality_list_raw for k in ['cause', 'effect'] if
                                 (e := p.get(k))}

        id_to_struct_map, raw_to_id_map = {}, {}
        if all_unique_events_raw:
            id_counter = 1
            for raw_tuple, raw_event in all_unique_events_raw.items():
                event_id = f"E{id_counter}"
                id_to_struct_map[event_id] = create_event_structure(raw_event, event_id)
                raw_to_id_map[raw_tuple] = event_id
                id_counter += 1

        event_list_struct = list(id_to_struct_map.values())

        rule_based_candidates = set()
        if len(event_list_struct) >= 2:
            sentences = [s.strip() for s in re.split(r'[。？！]', text) if s.strip()]
            event_to_sentence_idx = {e['event_id']: i for i, s in enumerate(sentences) for e in event_list_struct if
                                     e['trigger'] in s}
            for i in range(len(event_list_struct)):
                for j in range(len(event_list_struct)):
                    if i == j: continue
                    e_i, e_j = event_list_struct[i], event_list_struct[j]
                    is_strong = any(arg in e_i['arguments'] and arg in e_j['arguments'] and e_i['arguments'][arg] ==
                                    e_j['arguments'][arg] for arg in CORE_ARGUMENTS)
                    if is_strong: rule_based_candidates.add((e_i['event_id'], e_j['event_id'])); continue
                    idx_i, idx_j = event_to_sentence_idx.get(e_i['event_id']), event_to_sentence_idx.get(
                        e_j['event_id'])
                    if idx_i is not None and idx_j is not None and abs(idx_i - idx_j) < sentence_distance_threshold:
                        rule_based_candidates.add((e_i['event_id'], e_j['event_id']))

        positive_pairs_ids = {(raw_to_id_map.get(frozenset(p.get('cause', {}).items())),
                               raw_to_id_map.get(frozenset(p.get('effect', {}).items()))) for p in causality_list_raw}
        positive_pairs_ids = {p for p in positive_pairs_ids if p[0] and p[1]}

        # <<< MODIFIED: 核心逻辑修改
        # 1. 强制注入所有正样本
        # 2. 从规则生成的候选中筛选出高质量的负样本
        negative_candidates_ids = list(rule_based_candidates - positive_pairs_ids)
        num_neg_to_keep = int(len(positive_pairs_ids) * negative_ratio)
        random.shuffle(negative_candidates_ids)
        sampled_negative_ids = negative_candidates_ids[:num_neg_to_keep]

        # 最终的候选池 = 全部的正样本 + 采样后的负样本
        final_candidate_ids = list(positive_pairs_ids) + sampled_negative_ids

        stats_data.append({'count': len(final_candidate_ids), 'source_doc_id': doc_id, 'source_text': text,
                           'positive_count': len(positive_pairs_ids), 'negative_count': len(sampled_negative_ids)})

        candidate_pairs_for_input = []
        if final_candidate_ids:
            for id1, id2 in final_candidate_ids:
                # 确保ID存在于映射中，防止因数据不一致导致的KeyError
                if id1 in id_to_struct_map and id2 in id_to_struct_map:
                    event_a, event_b = id_to_struct_map[id1], id_to_struct_map[id2]
                    snippet = generate_context_snippet(text, event_a, event_b)
                    candidate_pairs_for_input.append(
                        OrderedDict([("event_1", event_a), ("event_2", event_b), ("context_snippet", snippet)]))
            random.shuffle(candidate_pairs_for_input)

        output_list = [OrderedDict([("cause", c['event_1']), ("effect", c['event_2'])]) for c in
                       candidate_pairs_for_input if
                       (c['event_1']['event_id'], c['event_2']['event_id']) in positive_pairs_ids]

        input_dict = {"text": text, "candidate_pairs": candidate_pairs_for_input}

        final_alpaca_data.append({
            "instruction": instruction_prompt,
            "input": json.dumps(input_dict, ensure_ascii=False, separators=(',', ':')),
            "output": json.dumps(output_list, ensure_ascii=False, separators=(',', ':')),
            "system": system_prompt
        })

    random.shuffle(final_alpaca_data)
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(final_alpaca_data, f_out, ensure_ascii=False, indent=4)

    print(f"\n--- 转换完成！ ---\n")
    print(f"原始数据总数: {len(original_data)}")
    print(f"成功生成的训练样本数: {len(final_alpaca_data)}")

    # --- 生成最终版统计文件 ---
    if stats_data:
        counts = [item['count'] for item in stats_data]
        docs_with_no_candidates = sum(1 for c in counts if c == 0)

        min_count, max_count = (int(np.min(counts)), int(np.max(counts))) if counts else (0, 0)
        mean_val, median_val, std_dev_val = (
        float(np.mean(counts)), float(np.median(counts)), float(np.std(counts))) if counts else (0.0, 0.0, 0.0)

        max_case_example = next((item for item in stats_data if item['count'] == max_count), None)
        min_case_example = next((item for item in stats_data if item['count'] == min_count), None)

        stats_summary = {
            "overall_summary": {
                "total_docs_processed": len(stats_data),
                "docs_with_empty_candidates": docs_with_no_candidates,
            },
            "candidate_count_stats": {
                "frequency_distribution": {str(k): v for k, v in sorted(Counter(counts).items())},
                "descriptive_stats": {
                    "mean": f"{mean_val:.2f}", "median": median_val,
                    "std_dev": f"{std_dev_val:.2f}", "min": min_count, "max": max_count
                },
                "max_candidates_case": {
                    "source_doc_id": max_case_example.get('source_doc_id'),
                    "count": max_case_example.get('count'),
                    "positive_count": max_case_example.get('positive_count'),
                    "negative_count": max_case_example.get('negative_count'),
                } if max_case_example else None,
                "min_candidates_case": {
                    "source_doc_id": min_case_example.get('source_doc_id'),
                    "count": min_case_example.get('count'),
                    "positive_count": min_case_example.get('positive_count'),
                    "negative_count": min_case_example.get('negative_count'),
                } if min_case_example else None
            }
        }

        stats_file_path = os.path.join(output_dir, 'dataset_statistics_train.json')
        with open(stats_file_path, 'w', encoding='utf-8') as f_stats:
            json.dump(stats_summary, f_stats, ensure_ascii=False, indent=4)
        print(f"详细的最终统计信息已保存至: {stats_file_path}")


# ==============================================================================
# 3. 命令行接口
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将原始因果数据转换为最终的“候选池过滤”Alpaca格式，并提供详细的统计分析。")
    parser.add_argument("--input_file", type=str, required=True, help="原始数据JSON文件路径。")
    parser.add_argument("--output_file", type=str, required=True, help="转换后的Alpaca格式JSON文件保存路径。")
    parser.add_argument("--sentence_distance", type=int, default=2, help="弱关联筛选的句子距离阈值。")
    parser.add_argument("--negative_ratio", type=float, default=1.0,
                        help="负样本与正样本的比例。例如，1.5表示每1个正样本，保留1.5个负样本。")
    args = parser.parse_args()
    build_gold_standard_dataset(args.input_file, args.output_file, args.sentence_distance, args.negative_ratio)