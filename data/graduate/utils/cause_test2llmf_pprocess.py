import json
import random
import math
import re
from collections import defaultdict, Counter
from typing import Dict, List


def transform_causality_structure(doc: dict) -> dict:
    """执行字段转换和删除操作"""
    if "causality_list" not in doc:
        return doc

    transformed = []
    for causality in doc["causality_list"]:
        causality.pop("causality_type", None)
        field_mapping = {
            "actor": "subject",
            "class": "event_type",
            "action": "trigger",
            "time": "date"
        }

        for role in ["cause", "effect"]:
            if role in causality:
                original = causality[role]
                transformed_event = {}
                for old_field, new_field in field_mapping.items():
                    if old_field in original:
                        transformed_event[new_field] = original[old_field]
                for field in ["location", "object"]:
                    if field in original:
                        transformed_event[field] = original[field]
                causality[role] = transformed_event
        transformed.append(causality)

    doc["causality_list"] = transformed
    return doc

def format_compact_json(data: dict) -> str:
    """安全生成紧凑型JSON字符串"""
    try:
        # 使用json模块自动处理转义
        return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    except Exception as e:
        print(f"JSON生成错误: {str(e)}")
        return "{}"


def generate_instruction(event_types: list) -> str:
    """生成动态指令"""
    return f"""你是一名专门进行事件因果关系抽取的专家。请按以下要求从input中抽取出符合system定义格式的事件因果关系：
1. 识别文本中具有因果关联的cause、effect事件对
2. 每个事件需包含：
   - subject（执行主体，必填）
   - event_type（从以下预定义列表选择：{", ".join(event_types)}）
   - trigger（标志事件触发的动词短语，必填）
   - date（原文中的事件发生时间，空值填NAN）
   - location（原文中的事件发生地点，空值填NAN） 
   - object（受影响对象，必填）
3. 请严格按照system定义的JSON格式输出，不要输出think过程"""


def convert_to_final_format(doc: dict, event_types: list) -> dict:
    """转换单个文档到最终格式（修复字段映射问题）"""
    # 处理实际数据
    output_data = {"causality_list": []}
    for causality in doc.get("causality_list", []):
        new_causality = {}
        for role in ["cause", "effect"]:
            event = causality.get(role, {})
            # 最终字段映射（date→time）
            new_event = {
                "subject": event.get("subject", "NAN"),
                "event_type": event.get("event_type", "NAN"),
                "trigger": event.get("trigger", "NAN"),
                "date": event.get("date", "NAN"),  # 关键映射：date→time
                "location": event.get("location", "NAN"),
                "object": event.get("object", "NAN")
            }
            # 处理空值
            for k, v in new_event.items():
                if v == "":
                    new_event[k] = "NAN"
            new_causality[role] = new_event
        output_data["causality_list"].append(new_causality)

    # 系统示例模板
    system_example = {
        "causality_list": [{
            "cause": {
                "subject": "示例主体",
                "event_type": "示例类型",
                "trigger": "示例动作",
                "date": "NAN",
                "location": "NAN",
                "object": "示例对象"
            },
            "effect": {
                "subject": "示例主体",
                "event_type": "示例类型",
                "trigger": "示例动作",
                "date": "NAN",
                "location": "NAN",
                "object": "示例对象"
            }
        }]
    }

    return {
        "instruction": generate_instruction(event_types),
        "input": doc["text"],
        "output": format_compact_json(output_data),
        "system": format_compact_json(system_example)
    }


def process_subset(subset: List[dict]) -> List[dict]:
    """处理子集数据转换"""
    event_types = set()
    for doc in subset:
        for causality in doc.get("causality_list", []):
            for role in ["cause", "effect"]:
                if et := causality.get(role, {}).get("event_type"):
                    event_types.add(et)
    return [convert_to_final_format(doc, sorted(event_types)) for doc in subset]

def generate_subsets(input_file: str, output_prefix: str, output_config: List[int], seed: int = 42) -> None:
    """精确累积划分数据集并统计类型分布"""
    # 加载并预处理数据
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_dataset = json.load(f)
    dataset = [transform_causality_structure(doc) for doc in raw_dataset]
    total_docs = len(dataset)
    print(f"原始数据集验证: 共{total_docs}个文档")

    # 构建类型索引
    type_index = defaultdict(list)
    for idx, doc in enumerate(dataset):
        types = set()
        for causality in doc.get("causality_list", []):
            for role in ["cause", "effect"]:
                if et := causality.get(role, {}).get("event_type"):
                    types.add(et)
        for et in types:
            type_index[et].append(idx)

    # 全局随机种子
    random.seed(seed)
    doc_indices = list(range(total_docs))
    random.shuffle(doc_indices)

    # 执行累积划分
    cumulative_indices = []
    prev_indices = set()
    required_types = set(type_index.keys())  # 所有现有类型

    for i, percent in enumerate(sorted(output_config)):
        target = math.ceil(total_docs * percent / 100)
        print(f"\n=== 正在生成 {percent}% 数据集 ===")

        # 阶段处理
        while len(cumulative_indices) < target:
            # 确保覆盖所有类型
            missing_types = required_types - get_current_types(dataset, cumulative_indices)
            if missing_types:
                # 优先补充缺失类型
                for et in missing_types:
                    candidates = [idx for idx in type_index[et] if idx not in cumulative_indices]
                    if candidates:
                        cumulative_indices.append(random.choice(candidates))
                        break
            else:
                # 常规随机补充
                remaining = [idx for idx in doc_indices if idx not in cumulative_indices]
                if not remaining:
                    break
                cumulative_indices.append(remaining[0])

        # 截取精确数量
        cumulative_indices = cumulative_indices[:target]
        subset = [dataset[idx] for idx in cumulative_indices]

        # 类型分布统计
        type_counter = count_event_types(subset)

        # 保存数据
        processed_subset = process_subset(subset)
        output_path = f"{output_prefix}_{percent}pct.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_subset, f, ensure_ascii=False, indent=2)

        # 输出统计
        print(f"文档数量: {len(processed_subset)}")
        print("事件类型分布:")
        for et, count in type_counter.most_common():
            print(f"  {et}: {count}")

        # 包含性验证
        if i > 0:
            assert prev_indices.issubset(set(cumulative_indices)), "包含关系验证失败"
        prev_indices = set(cumulative_indices.copy())

    # 最终验证
    assert len(cumulative_indices) == total_docs, "最终数据集不完整"
    print("\n=== 最终验证通过 ===")

def get_current_types(dataset: List[dict], indices: List[int]) -> set:
    """获取当前数据集包含的类型"""
    types = set()
    for idx in indices:
        doc = dataset[idx]
        for causality in doc.get("causality_list", []):
            for role in ["cause", "effect"]:
                if et := causality.get(role, {}).get("event_type"):
                    types.add(et)
    return types

def count_event_types(subset: List[dict]) -> Counter:
    """统计事件类型分布"""
    counter = Counter()
    for doc in subset:
        for causality in doc.get("causality_list", []):
            for role in ["cause", "effect"]:
                if et := causality.get(role, {}).get("event_type"):
                    counter[et] += 1
    return counter

if __name__ == "__main__":
    CONFIG = [100]  # 必须按升序排列

    generate_subsets(
        input_file="../领域事件多因果关联挖掘/test1.json",
        output_prefix="../领域事件多因果关联挖掘/test",
        output_config=CONFIG,
        seed=42
    )