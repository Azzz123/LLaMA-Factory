import json
import os
import re
import argparse
import random
from tqdm import tqdm
from collections import Counter, defaultdict

# --- 1. 核心模板和逻辑 (这部分保持不变) ---

LEAN_SYSTEM_PROMPT = "你是一个军事领域文档级多事件抽取专家。严格遵循用户指令中的Schema和规则，将文本中事件转换为结构化的JSON。禁止创造Schema之外的任何字段。"
LEAN_INSTRUCTION_PROMPT = """任务：从输入文本中，根据以下Schema抽取所有军事事件。

# Schema
一、事件类型 (可选):
["试验", "演习", "部署", "支援", "意外事故", "展示", "冲突", "伤亡"]
二、论元角色 (可选):
["主体", "装备", "日期", "位置", "演习内容", "军事力量", "客体", "物资", "事故后果", "数量", "区域"]
三、事件-论元映射:
- 试验: [主体, 装备, 日期, 位置]
- 演习: [主体, 演习内容, 日期, 区域]
- 部署: [主体, 军事力量, 日期, 位置]
- 支援: [主体, 客体, 物资, 日期]
- 意外事故: [主体, 事故后果, 日期, 位置]
- 展示: [主体, 装备, 日期, 位置]
- 冲突: [主体, 客体, 日期, 位置]
- 伤亡: [主体, 数量, 日期, 位置]

# 规则
1. 严格匹配：事件类型和论元角色名必须与Schema完全一致。
2. 缺省值：若无对应论元，值设为 "NAN"。
3. 空结果：若无事件，返回空列表 []。
4. 思维链：在最终JSON前，必须按格式输出<think>推理过程</think>。"""
LEAN_EXPLANATION_TEMPLATES = {
    "试验": "描述了对“{装备}”等装备的性能测试活动。",
    "演习": "描述了由“{主体}”进行的模拟军事行动。",
    "部署": "描述了“{主体}”将“{军事力量}”等力量进行调动安置。",
    "支援": "描述了“{主体}”向“{客体}”提供帮助的行为。",
    "意外事故": "描述了造成“{事故后果}”等后果的突发情况。",
    "展示": "描述了公开显示“{装备}”等军事能力。",
    "冲突": "描述了“{主体}”与“{客体}”间的敌对交战。",
    "伤亡": "描述了人员伤亡情况，涉及数量“{数量}”。"
}


def find_evidence_sentence(text, trigger):
    sentences = re.split(r'([.。!！?？\n])', text)
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    if not sentences: sentences = [text]
    for sentence in sentences:
        if trigger in sentence: return sentence.strip()
    return "（整个输入文本为上下文）"


def generate_final_cot_item(item):
    input_text = item.get("input", "")
    original_output_str = item.get("output", "[]")
    try:
        events = json.loads(original_output_str)
    except json.JSONDecodeError:
        return item
    if not events:
        thinking_process = "经过分析，输入文本中未发现任何符合Schema定义的军事事件。"
    else:
        analysis_section = "1. 事件分析:\n"
        for event in events:
            event_type, trigger = event.get("event_type"), event.get("trigger")
            template = LEAN_EXPLANATION_TEMPLATES.get(event_type, "符合该事件类型的定义。")
            arg_values = {role: '、'.join(v) if isinstance(v, list) else v for role, v in
                          event.get("arguments", {}).items()}
            try:
                reason = template.format(**arg_values)
            except KeyError:
                reason = "符合该事件类型的定义。"
            evidence = find_evidence_sentence(input_text, trigger)
            analysis_section += f"- 事件: {event_type}, 触发词: {trigger}, 理由: {reason} 证据句: “{evidence}”。\n"
        argument_section = "2. 论元抽取:\n"
        for event in events:
            argument_section += f"- {event.get('event_type')}事件 ({event.get('trigger')}):\n"
            for role, values in event.get('arguments', {}).items():
                value_str = "NAN" if "NAN" in str(values) else (
                    '、'.join(map(str, values)) if isinstance(values, list) else str(values))
                argument_section += f"  - {role}: {value_str}\n"
        finalization_section = "3. 格式化:\n- 整合信息生成JSON。"
        thinking_process = f"{analysis_section.strip()}\n\n{argument_section.strip()}\n\n{finalization_section.strip()}"
    final_output = f"<think>\n{thinking_process}\n</think>\n\n{original_output_str}"
    return {"instruction": LEAN_INSTRUCTION_PROMPT, "input": input_text, "output": final_output,
            "system": LEAN_SYSTEM_PROMPT}


def stratified_sample_data(input_file, output_file, target_size, min_per_class):
    print(f"▶️  开始对文件进行分层采样: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    print("  - 步骤1: 分析原始数据事件分布...")
    event_counts = Counter()
    items_by_event = defaultdict(list)
    no_event_items = []

    for i, item in enumerate(tqdm(original_data, desc="  分析中")):
        try:
            events = json.loads(item.get("output", "[]"))
            if not events:
                no_event_items.append(i)
                continue

            item_event_types = set(e.get("event_type") for e in events)
            for event_type in item_event_types:
                if event_type:
                    event_counts[event_type] += 1
                    items_by_event[event_type].append(i)
        except (json.JSONDecodeError, AttributeError):
            continue

    print("\n  原始事件频次分布:")
    for event, count in event_counts.most_common():
        print(f"  - {event}: {count} 次")

    print("\n  - 步骤2: 计算采样策略...")
    total_events = sum(event_counts.values())
    proportions = {event: count / total_events for event, count in event_counts.items()}

    # 按比例分配目标数量，但要保证不低于min_per_class
    target_counts = {event: max(min_per_class, int(proportions[event] * target_size)) for event in event_counts}

    print("\n  目标采样事件频次 (已考虑最低阈值):")
    for event, count in target_counts.items():
        print(f"  - {event}: 约 {count} 条样本")

    print("\n  - 步骤3: 执行分层采样...")
    selected_indices = set()
    for event_type, target_count in tqdm(target_counts.items(), desc="  采样中"):
        available_indices = items_by_event[event_type]
        # 如果可用样本不足，则全部选取
        num_to_sample = min(target_count, len(available_indices))
        # 从可用样本中随机抽取，不重复
        sampled_indices = random.sample(available_indices, num_to_sample)
        selected_indices.update(sampled_indices)

    print(f"\n  - 步骤4: 调整至最终数量 {target_size}...")
    # 如果当前选中的样本不足，从无事件样本和剩余样本中补充
    current_size = len(selected_indices)
    if current_size < target_size:
        remaining_indices = list(set(range(len(original_data))) - selected_indices)
        num_to_add = target_size - current_size
        if len(remaining_indices) >= num_to_add:
            selected_indices.update(random.sample(remaining_indices, num_to_add))

    # 如果超出，则随机剔除
    while len(selected_indices) > target_size:
        selected_indices.remove(random.choice(list(selected_indices)))

    final_data = [original_data[i] for i in sorted(list(selected_indices))]

    print(f"\n✅ 分层采样完成！最终样本数: {len(final_data)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"  采样后的文件已保存至: {output_file}")


# --- 主函数和命令行解析 ---
def main():
    parser = argparse.ArgumentParser(
        description="一个集成了CoT转换和分层采样功能的数据处理工具。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='选择要执行的命令')

    # CoT转换命令
    parser_convert = subparsers.add_parser('convert', help='将原始数据集转换为精益CoT格式。')
    parser_convert.add_argument('-i', '--input-dir', required=True, help='包含原始 .json 文件的输入目录。')
    parser_convert.add_argument('-o', '--output-dir', required=True, help='用于保存转换后文件的输出目录。')

    # 分层采样命令
    parser_sample = subparsers.add_parser('sample', help='对大型数据集进行分层采样。')
    parser_sample.add_argument('-i', '--input-file', required=True, help='原始的 .json 数据集文件路径。')
    parser_sample.add_argument('-o', '--output-file', required=True, help='保存采样后数据的文件路径。')
    parser_sample.add_argument('-s', '--size', type=int, default=5000, help='采样后的目标数据集大小 (默认: 4000)。')
    parser_sample.add_argument('--min-class', type=int, default=400, help='每个事件类别保证的最低样本数 (默认: 50)。')

    args = parser.parse_args()

    if args.command == 'convert':
        if not os.path.isdir(args.input_dir):
            print(f"❌ 错误: 输入目录 '{args.input_dir}' 不存在。")
            return
        files_to_process = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
        for filename in files_to_process:
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            os.makedirs(args.output_dir, exist_ok=True)
            # --- 执行CoT转换 ---
            print(f"▶️  开始CoT转换: {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f: data = json.load(f)
            processed_data = [generate_final_cot_item(item) for item in tqdm(data, desc=f"  转换中 for {filename}")]
            with open(output_path, 'w', encoding='utf-8') as f: json.dump(processed_data, f, ensure_ascii=False,
                                                                          indent=2)
            print(f"✅ CoT转换完成: {output_path}\n")

    elif args.command == 'sample':
        if not os.path.isfile(args.input_file):
            print(f"❌ 错误: 输入文件 '{args.input_file}' 不存在。")
            return
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        # --- 执行分层采样 ---
        stratified_sample_data(args.input_file, args.output_file, args.size, args.min_class)


if __name__ == '__main__':
    main()