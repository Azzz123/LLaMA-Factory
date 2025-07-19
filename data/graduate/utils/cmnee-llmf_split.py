import json
import random
import math
from collections import defaultdict, Counter
from typing import List, Dict


def load_data(input_file: str) -> List[Dict]:
    """加载 JSON 数据"""
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def count_event_types(dataset: List[Dict]) -> Counter:
    """统计数据集中指定事件类型的个数"""
    event_types = ["试验", "演习", "部署", "支援", "意外事故", "展示", "冲突", "伤亡"]
    counter = Counter()
    for doc in dataset:
        output_data = json.loads(doc["output"])  # 解析 output 字段
        for et in event_types:
            counter[et] += len(output_data.get(et, []))
    return counter


def build_type_index(dataset: List[Dict]) -> Dict[str, List[int]]:
    """构建基于事件类型的索引"""
    event_types = ["试验", "演习", "部署", "支援", "意外事故", "展示", "冲突", "伤亡"]
    type_index = defaultdict(list)
    for idx, doc in enumerate(dataset):
        output_data = json.loads(doc["output"])
        for et in event_types:
            if output_data.get(et):
                type_index[et].append(idx)
    return type_index


def split_dataset(input_file: str, output_config: List[int], seed: int = 42):
    """累积划分数据集，保证事件类型尽量均衡"""
    dataset = load_data(input_file)
    total_docs = len(dataset)
    print(f"原始数据集共有 {total_docs} 条数据")

    type_index = build_type_index(dataset)
    random.seed(seed)
    doc_indices = list(range(total_docs))
    random.shuffle(doc_indices)

    cumulative_indices = []
    prev_indices = set()
    required_types = set(type_index.keys())

    for i, percent in enumerate(sorted(output_config)):
        target_size = math.ceil(total_docs * percent / 100)
        print(f"\n=== 生成 {percent}% 数据集 (目标: {target_size} 条) ===")

        # 计算还需添加多少数据
        remaining_needed = target_size - len(cumulative_indices)

        if remaining_needed > 0:
            remaining = [idx for idx in doc_indices if idx not in cumulative_indices]
            cumulative_indices.extend(remaining[:remaining_needed])

        cumulative_indices = cumulative_indices[:target_size]  # 确保精确数量
        subset = [dataset[idx] for idx in cumulative_indices]

        # 统计事件类型数量
        type_counts = count_event_types(subset)

        output_path = f"{input_file.replace('.json', '')}{percent}pct.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)

        print(f"子集文档数量: {len(subset)}，已保存至: {output_path}")
        print("子集中事件类型统计:")
        for et, count in type_counts.items():
            print(f"  {et}: {count}")


def main():
    input_file = "../cmnee_llmf/train_balanced_50pertype.json"  # 直接指定输入文件路径
    output_config = [100]  # 直接指定划分比例

    split_dataset(input_file, output_config)


if __name__ == "__main__":
    main()
