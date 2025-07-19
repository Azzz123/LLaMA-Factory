import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict


def analyze_milcause_split(file_path):
    """
    Analyzes a MilCause data split to get task-specific counts:
    - Sample Count
    - Candidate Pair Count
    - Causal Relation Count
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在，已跳过: {file_path}")
        return None

    print(f"正在从MilCause文件分析任务指标: {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_samples = len(data)
    num_causal_relations = 0
    num_candidate_pairs = 0

    for sample in tqdm(data, desc=f"处理 {os.path.basename(file_path)}"):
        # 1. 统计因果关系数量 (来自output)
        try:
            output_list = json.loads(sample.get("output", "[]"))
            if isinstance(output_list, list):
                num_causal_relations += len(output_list)
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. 统计候选对数量 (来自input)
        try:
            input_data = json.loads(sample.get("input", "{}"))
            candidate_pairs = input_data.get("candidate_pairs", [])
            if isinstance(candidate_pairs, list):
                num_candidate_pairs += len(candidate_pairs)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "样本数量": num_samples,
        "候选对数量": num_candidate_pairs,
        "因果关系数量": num_causal_relations,
    }


def main():
    parser = argparse.ArgumentParser(description="从MilCause文件生成任务相关的统计信息。")
    parser.add_argument("--milcause_dir", type=str, required=True,
                        help="包含train.json和test.json的MilCause数据集目录路径。")
    args = parser.parse_args()

    splits = ["train", "test"]
    all_stats = {}
    for split in splits:
        file_path = os.path.join(args.milcause_dir, f"{split}.json")
        stats = analyze_milcause_split(file_path)
        if stats:
            all_stats[split] = stats

    print("\n--- MilCause 任务指标统计 ---")
    stat_keys = ["样本数量", "候选对数量", "因果关系数量"]
    for key in stat_keys:
        train_val = all_stats.get("train", {}).get(key, 0)
        test_val = all_stats.get("test", {}).get(key, 0)
        print(f"- {key}:")
        print(f"  - 训练集: {train_val}")
        print(f"  - 测试集: {test_val}")
        print(f"  - 总计: {train_val + test_val}")


if __name__ == "__main__":
    # 使用方法:
    # python analyze_milcause_counts.py --milcause_dir data/graduate/DSN-DEMAMD-new
    main()