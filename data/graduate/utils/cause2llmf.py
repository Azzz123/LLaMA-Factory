import json
import argparse
from collections import OrderedDict
import random

# 1. Schema 定义 (无变化)
# ==============================================================================
TARGET_ARGUMENT_ROLES = {"主体", "客体", "日期", "位置"}
SOURCE_TO_TARGET_MAP = {
    "actor": "主体",
    "object": "客体",
    "time": "日期",
    "location": "位置"
}


# 2. 核心转换函数 (无变化)
# ==============================================================================
def create_event_structure(event_data, event_id):
    """
    根据原始事件数据和预定义的Schema，创建一个标准的、结构化的事件对象。
    """
    event_type = event_data.get('class', '未知类型')
    trigger = event_data.get('action', 'N/A')
    arguments = {}
    for source_key, target_key in SOURCE_TO_TARGET_MAP.items():
        value = event_data.get(source_key)
        if value and value.strip():
            arguments[target_key] = [value]

    structured_event = OrderedDict([
        ("event_id", event_id),
        ("event_type", event_type),
        ("trigger", trigger),
        ("arguments", arguments)
    ])
    return structured_event


# 3. 主转换函数 (已修改)
# ==============================================================================
def convert_to_system_format(input_file, output_file):
    """
    主转换函数：读取原始因果关系数据，并将其转换为包含 system 字段的Alpaca格式。
    核心逻辑是将每个文档的所有事件和因果关系对转换为一条训练/评估样本。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            original_data = json.load(f_in)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {input_file} 不是有效的JSON格式。")
        return

    alpaca_data = []

    # <<< MODIFIED: 将原始 instruction 拆分为 system_prompt 和 new_instruction
    system_prompt = "你是一名顶级的军事事件因果关系分析抽取专家。"
    new_instruction = (
        "你的任务是分析'input'中提供的'text'和'event_list'，识别出所有存在的因果关系对。"
        "你需要以JSON列表的格式输出所有你找到的因果对。每个因果对都应包含'cause'和'effect'两个键，其值是完整的事件对象，并包含'event_id'。"
        "输出的事件对象格式必须严格遵循 { 'event_id': '...', 'event_type': '...', 'trigger': '...', 'arguments': {...} } 的结构。"
        "判断必须基于所提供的文本证据，并结合常规事件因果逻辑。"
        "如果文本中不存在任何因果关系，请直接输出一个空列表：[]。"
    )

    all_docs = []
    for entry in original_data:
        text = entry.get('text', '')
        if not text:
            continue

        causality_list = entry.get('causality_list', [])

        # 识别并聚合文档中的所有独立事件
        all_events_in_doc = {}
        # 优先从 event_list 获取
        for item in entry.get('event_list', []):
            event_tuple = frozenset(item.items())
            if event_tuple not in all_events_in_doc:
                all_events_in_doc[event_tuple] = item

        # 如果没有 event_list，则从 causality_list 中提取
        if not all_events_in_doc:
            for pair in causality_list:
                for event_key in ['cause', 'effect']:
                    event_data_dict = pair.get(event_key, {})
                    if not event_data_dict: continue
                    event_tuple = frozenset(event_data_dict.items())
                    if event_tuple not in all_events_in_doc:
                        all_events_in_doc[event_tuple] = event_data_dict

        all_docs.append({
            "text": text,
            "all_events_raw": list(all_events_in_doc.values()),
            "causality_list_raw": causality_list
        })

    # 统一处理所有文档，生成Alpaca格式数据
    for doc_data in all_docs:
        text = doc_data["text"]
        raw_events = doc_data["all_events_raw"]
        causality_list = doc_data["causality_list_raw"]

        if len(raw_events) < 1:
            continue

        events_with_ids = {f"E{i + 1}": event for i, event in enumerate(raw_events)}
        event_to_id_map = {frozenset(v.items()): k for k, v in events_with_ids.items()}

        # 构建模型的输入(input)部分
        input_events_list = [
            create_event_structure(event_data, event_id)
            for event_id, event_data in sorted(events_with_ids.items())
        ]
        input_json = {"text": text, "event_list": input_events_list}
        input_text = json.dumps(input_json, ensure_ascii=False, separators=(',', ':'))

        # 构建模型的目标输出(output)部分
        output_list = []
        for pair in causality_list:
            cause_data = pair.get('cause', {})
            effect_data = pair.get('effect', {})
            cause_id = event_to_id_map.get(frozenset(cause_data.items()))
            effect_id = event_to_id_map.get(frozenset(effect_data.items()))
            if not cause_id or not effect_id: continue

            cause_event_struct = create_event_structure(cause_data, cause_id)
            effect_event_struct = create_event_structure(effect_data, effect_id)
            output_list.append(OrderedDict([
                ("cause", cause_event_struct),
                ("effect", effect_event_struct),
                ("reason", "")
            ]))

        output_text = json.dumps(output_list, ensure_ascii=False, separators=(',', ':'))

        # <<< MODIFIED: 使用 OrderedDict 来构建最终条目，确保 'system' 字段在最后
        alpaca_entry = OrderedDict([
            ("instruction", new_instruction),
            ("input", input_text),
            ("output", output_text),
            ("system", system_prompt)  # 将 system 字段放在最后
        ])

        alpaca_data.append(alpaca_entry)

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(alpaca_data, f_out, ensure_ascii=False, indent=4)

    print(f"转换完成！已将 {len(original_data)} 条原始文档转换为 {len(alpaca_data)} 条带 'system' 字段的训练样本。")
    print(f"文件已保存至: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw event data to a document-level Alpaca format with a 'system' field for causal relation extraction.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the output Alpaca-formatted JSON file with a system prompt.")

    args = parser.parse_args()

    # <<< MODIFIED: 调用更新后的函数名
    convert_to_system_format(args.input_file, args.output_file)