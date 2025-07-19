import json
import random
import argparse


def load_json(file_path: str):
    """加载 JSON 文件，返回列表数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path: str):
    """将数据保存为 JSON 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_datasets(file1_path: str, file2_path: str):
    """
    合并两个数据集：
    1. 加载 file1 和 file2（均为 JSON 列表）。
    2. 将 file1 的数据随机打乱后，依次随机插入到 file2 中。
    3. 返回合并后的列表。
    """
    dataset1 = load_json(file1_path)
    dataset2 = load_json(file2_path)

    # 随机打乱 dataset1 的数据
    random.shuffle(dataset1)

    # 以 file2 为基础，将 dataset1 的每个元素随机插入到 dataset2 中
    merged = dataset2[:]  # 复制 file2 数据
    for item in dataset1:
        # 随机生成一个插入位置，范围为 0 到当前 merged 列表长度
        insert_index = random.randint(0, len(merged))
        merged.insert(insert_index, item)

    return merged


def main():
    file1 = "../cmnee_llmf/train_balanced_200pertype.json"
    file2 = "../领域事件多因果关联挖掘/train_100pct.json"
    output = "../CMNEE_DSN_mixed/train.json"

    merged_data = merge_datasets(file1, file2)
    save_json(merged_data, output)
    print(f"合并后的数据集共包含 {len(merged_data)} 条数据，已保存至: {output}")


if __name__ == "__main__":
    main()
