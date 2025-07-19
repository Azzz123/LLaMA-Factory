import json
import random
import os
from collections import defaultdict, Counter
from typing import List, Dict


def load_data(input_file: str) -> List[Dict]:
    """加载 JSON 数据"""
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_event_types() -> List[str]:
    """预定义的事件类型列表"""
    return ["试验", "演习", "部署", "支援", "意外事故", "展示", "冲突", "伤亡"]


def build_exclusive_type_index(dataset: List[Dict]) -> Dict[str, List[int]]:
    """
    为每个文档专属分配一个事件类型（按照预定义顺序），构建事件类型到文档索引的映射。
    如果一个文档同时包含多个事件类型，则只取第一个出现的类型。
    """
    event_types = get_event_types()
    type_index = defaultdict(list)
    for idx, doc in enumerate(dataset):
        output_data = json.loads(doc["output"])
        for et in event_types:
            # 如果该类型存在且至少包含一条事件，则将文档分配给该类型
            if et in output_data and output_data.get(et) and len(output_data.get(et)) > 0:
                type_index[et].append(idx)
                break  # 每个文档只归入一个类别
    return type_index


def count_event_types_in_subset(dataset: List[Dict], indices: List[int]) -> Counter:
    """统计给定文档索引列表中各事件类型出现的事件数量（注意一个文档可能包含多个事件）"""
    event_types = get_event_types()
    counter = Counter()
    for idx in indices:
        doc = dataset[idx]
        output_data = json.loads(doc["output"])
        for et in event_types:
            if et in output_data and output_data.get(et):
                counter[et] += len(output_data.get(et))
    return counter


def split_dataset_balanced(input_file: str, target_per_type: int, seed: int = 42):
    """
    划分数据集，确保在采样后每个事件类型至少（理想情况）包含 target_per_type 个文档。
    最终的平衡数据集保存至与原始文件相同目录下的新文件。
    """
    dataset = load_data(input_file)
    total_docs = len(dataset)
    print(f"原始数据集共有 {total_docs} 条数据")

    # 构建专属事件类型索引
    type_index = build_exclusive_type_index(dataset)
    random.seed(seed)

    selected_indices = []
    for et in get_event_types():
        indices = type_index.get(et, [])
        if len(indices) < target_per_type:
            print(f"Warning: 事件类型 '{et}' 只有 {len(indices)} 条数据，目标为 {target_per_type} 条")
            chosen = indices  # 不足则全部采样
        else:
            chosen = random.sample(indices, target_per_type)
        selected_indices.extend(chosen)
        print(f"事件类型 '{et}' 选取了 {len(chosen)} 条数据")

    # 排序选定的索引（非必须，只是为了结果更整齐）
    selected_indices = sorted(selected_indices)
    subset = [dataset[idx] for idx in selected_indices]

    # 构造输出文件路径：与原始文件在同一目录，文件名添加后缀
    input_dir = os.path.dirname(input_file)
    input_basename = os.path.basename(input_file)
    output_filename = os.path.splitext(input_basename)[0] + f"_balanced_{target_per_type}pertype.json"
    output_path = os.path.join(input_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"\n平衡数据集共包含 {len(subset)} 条数据，已保存至: {output_path}")

    # 统计并输出平衡数据集中各事件类型的出现情况（按 output 内事件数量计）
    overall_counts = count_event_types_in_subset(dataset, selected_indices)
    print("平衡数据集中各事件类型统计（按 output 内事件数量计）：")
    for et in get_event_types():
        print(f"  {et}: {overall_counts.get(et, 0)}")


def main():
    input_file = "../cmnee_llmf/test.json"  # 指定输入文件路径
    target_per_type = 1000  # 希望每个事件类型选取的文档数
    split_dataset_balanced(input_file, target_per_type)


if __name__ == "__main__":
    main()
