import json
from collections import defaultdict


def analyze_dataset(file_path):
    """
    分析数据集并生成统计报告
    :param file_path: JSON文件路径
    :return: 统计结果字典
    """
    # 初始化统计容器
    class_stats = defaultdict(int)
    missing_causality_docs = set()
    missing_causality_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"文件读取失败: {str(e)}")
        return None

    for item in dataset:
        try:
            doc_id = item["document_id"]
            causality_list = item.get("causality_list", [])

            for causality in causality_list:
                # 统计缺少causality_type的情况
                if "causality_type" not in causality:
                    missing_causality_docs.add(doc_id)
                    missing_causality_count += 1

                # 统计class字段
                for role in ["cause", "effect"]:
                    event = causality.get(role, {})
                    if "class" in event and event["class"]:
                        class_name = event["class"].strip()
                        class_stats[class_name] += 1

        except KeyError as e:
            print(f"数据格式错误，缺少必要字段: {str(e)}")
            continue

    return {
        "class_distribution": dict(class_stats),
        "missing_causality": {
            "total": missing_causality_count,
            "doc_ids": list(missing_causality_docs)
        }
    }


def print_report(result):
    """打印统计报告"""
    print("=== Class类型分布 ===")
    for cls, count in result["class_distribution"].items():
        print(f"{cls}: {count}个")

    print("\n=== 缺失causality_type统计 ===")
    print(f"总缺失条目数: {result['missing_causality']['total']}")
    print(f"涉及文档ID: {', '.join(map(str, result['missing_causality']['doc_ids']))}")


# 使用示例
if __name__ == "__main__":
    result = analyze_dataset("../领域事件多因果关联挖掘/train1.json")
    if result:
        print_report(result)
    else:
        print("分析失败，请检查输入文件")