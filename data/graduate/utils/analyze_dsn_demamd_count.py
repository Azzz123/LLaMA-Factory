import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict


def event_to_tuple(event_obj):
    """
    Converts an event dictionary to a frozenset of its items for reliable hashing and uniqueness.
    Only considers non-empty string values.
    """
    # 只包含有实际内容的论元字段
    return frozenset(item for item in event_obj.items() if isinstance(item[1], str) and item[1])


def analyze_source_split(file_path):
    """
    Analyzes a single DSN-DEMAMD source file and returns its true statistics.
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在，已跳过: {file_path}")
        return None

    print(f"正在从源文件分析: {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_samples = len(data)
    num_causal_relations = 0

    # <<< ULTIMATE CORRECTION V3: 使用集合对事件内容进行全局去重 >>>
    unique_events_set = set()

    for sample in tqdm(data, desc=f"处理 {os.path.basename(file_path)}"):
        causality_list = sample.get("causality_list", [])
        if isinstance(causality_list, list):
            num_causal_relations += len(causality_list)
            for pair in causality_list:
                cause_event = pair.get("cause")
                effect_event = pair.get("effect")
                if isinstance(cause_event, dict):
                    unique_events_set.add(event_to_tuple(cause_event))
                if isinstance(effect_event, dict):
                    unique_events_set.add(event_to_tuple(effect_event))

    # 3. 统计唯一事件数量
    num_event_instances = len(unique_events_set)

    # 4. 统计事件论元数量
    num_arguments = 0
    for event_tuple in unique_events_set:
        # 元组的长度就是非空论元的数量
        num_arguments += len(event_tuple)

    return {
        "样本数量": num_samples,
        "因果关系数量": num_causal_relations,
        "事件实例数量": num_event_instances,
        "事件论元数量": num_arguments
    }


def main():
    parser = argparse.ArgumentParser(description="从DSN-DEMAMD源文件生成最权威的数据集统计信息。")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="包含原始train1.json和valid.json的数据集目录路径。")
    args = parser.parse_args()

    # 根据你的描述，train1.json -> 训练集, valid.json -> 测试集
    splits_map = {
        "train": "train1.json",
        "test": "valid.json"
    }

    all_stats = {}
    for split_name, file_name in splits_map.items():
        file_path = os.path.join(args.source_dir, file_name)
        stats = analyze_source_split(file_path)
        if stats:
            all_stats[split_name] = stats

    # 计算总计
    total_stats = defaultdict(int)
    # 总计的事件和论元需要重新计算以保证全局唯一性
    total_unique_events_set = set()

    for split_name, file_name in splits_map.items():
        file_path = os.path.join(args.source_dir, file_name)
        if not os.path.exists(file_path): continue
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for sample in data:
            causality_list = sample.get("causality_list", [])
            for pair in causality_list:
                if pair.get("cause"): total_unique_events_set.add(event_to_tuple(pair["cause"]))
                if pair.get("effect"): total_unique_events_set.add(event_to_tuple(pair["effect"]))

        # 累加其他简单指标
        if split_name in all_stats:
            total_stats["样本数量"] += all_stats[split_name]["样本数量"]
            total_stats["因果关系数量"] += all_stats[split_name]["因果关系数量"]

    total_stats["事件实例数量"] = len(total_unique_events_set)
    total_stats["事件论元数量"] = sum(len(event_tuple) for event_tuple in total_unique_events_set)
    all_stats["total"] = dict(total_stats)

    print("\n--- MilCause 数据集真实统计信息 (源数据权威版) ---")

    header = "| 统计项         | 训练集   | 测试集   | 总计     |"
    separator = "|----------------|----------|----------|----------|"
    print(header)
    print(separator)

    stat_keys = ["样本数量", "因果关系数量", "事件实例数量", "事件论元数量"]
    for key in stat_keys:
        train_val = all_stats.get("train", {}).get(key, 0)
        test_val = all_stats.get("test", {}).get(key, 0)
        total_val = all_stats.get("total", {}).get(key, 0)
        print(f"| {key:<14} | {train_val:<8} | {test_val:<8} | {total_val:<8} |")


if __name__ == "__main__":
    # 使用方法:
    # python analyze_dsn_demamd_count.py --source_dir data/graduate/DSN-DEMAMD
    main()