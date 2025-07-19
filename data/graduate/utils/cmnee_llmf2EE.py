import json
import os

# -----------------------------------------------------------------------------
# 1. 定义Schema和Prompt模板
# -----------------------------------------------------------------------------

# 基于你提供的Schema定义
SCHEMA = {
    "event_types": [
        "试验", "演习", "部署", "支援", "意外事故", "展示", "冲突", "伤亡"
    ],
    "argument_roles": [
        "主体", "装备", "日期", "区域", "演习内容", "位置", "军事力量", "客体", "物资", "事故后果", "数量"
    ],
    "event_argument_mapping": {
        "试验": ["主体", "装备", "日期", "位置"],
        "演习": ["主体", "演习内容", "日期", "区域"],
        "部署": ["主体", "军事力量", "日期", "位置"],
        "支援": ["主体", "客体", "物资", "日期"],
        "意外事故": ["主体", "事故后果", "日期", "位置"],
        "展示": ["主体", "装备", "日期", "位置"],
        "冲突": ["主体", "客体", "日期", "位置"],
        "伤亡": ["主体", "数量", "日期", "位置"]
    }
}

# 格式化Schema以插入到指令中
event_types_str = ", ".join(SCHEMA["event_types"])
argument_roles_str = ", ".join(SCHEMA["argument_roles"])
event_argument_mapping_str = "\n".join(
    [f"- {event_type}: [{', '.join(roles)}]" for event_type, roles in SCHEMA["event_argument_mapping"].items()]
)

# 定义指令模板 (Instruction)
INSTRUCTION_TEMPLATE = f"""请根据以下Schema定义，从输入文本中抽取出所有符合条件的军事事件。

Schema:
事件类型: {event_types_str}
论元角色: {argument_roles_str}

事件与论元对应关系:
{event_argument_mapping_str}
请以JSON列表格式返回结果。列表中的每个对象代表一个独立的事件，并为其分配一个从1开始的唯一ID（如 "E1", "E2", ...）。
每个事件对象应包含 'event_id', 'event_type', 'trigger', 和 'arguments' 四个字段。如果文本中没有事件，请返回一个空列表[]
"""

# 定义系统提示 (System Prompt)
SYSTEM_PROMPT = ("你是一名专门进行文本内多军事事件抽取的专家。你的任务是严格按照给定的Schema，"
                 "从文本中识别所有军事事件，为每个事件分配一个唯一的ID，并将它们以结构化的JSON列表形式输出。")


# -----------------------------------------------------------------------------
# 2. 数据转换函数
# -----------------------------------------------------------------------------

def transform_old_format_to_new_with_id(old_output):
    """
    将旧的 {事件类型: [事件...]} 格式转换为新的 [{事件}, {事件}, ...] 格式，并添加 event_id。
    """
    new_event_list = []
    # 如果输入是字符串，先尝试解析为Python字典
    if isinstance(old_output, str):
        try:
            data = json.loads(old_output)
        except json.JSONDecodeError:
            print(f"警告: 无法解析JSON字符串: {old_output}")
            return []
    elif isinstance(old_output, dict):
        data = old_output
    else:
        print(f"警告: 输入格式不正确，既不是字符串也不是字典: {type(old_output)}")
        return []

    # 按事件类型排序，以确保ID分配的稳定性（可选，但推荐）
    # sorted_event_types = sorted(data.keys())

    event_counter = 1
    # 按照Schema中定义的事件类型顺序来遍历，确保ID分配的顺序一致
    for event_type in SCHEMA["event_types"]:
        if event_type not in data or not data[event_type]:
            continue

        events = data[event_type]
        for event_instance in events:
            if not isinstance(event_instance, dict):
                print(f"警告: 在事件类型 '{event_type}' 下发现非字典类型的事件: {event_instance}")
                continue

            # 创建新的事件对象，并添加 event_id
            new_event = {
                "event_id": f"E{event_counter}",
                "event_type": event_type,
                "trigger": event_instance.get("trigger", "N/A"),
                "arguments": event_instance.get("arguments", {})
            }
            new_event_list.append(new_event)
            event_counter += 1

    return new_event_list


def convert_to_alpaca_format(original_data_path, output_path):
    """
    读取原始JSON文件，将其转换为Alpaca格式，并保存到新文件。
    """
    try:
        with open(original_data_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{original_data_path}'。请确保文件存在于脚本同目录下。")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 '{original_data_path}' 不是有效的JSON格式。")
        return

    alpaca_data = []
    for i, item in enumerate(original_data):
        input_text = item.get("input")
        original_output = item.get("output")

        if not input_text or not original_output:
            print(f"警告: 第 {i + 1} 条记录缺少 'input' 或 'output' 字段，已跳过。")
            continue

        # [*** 更新 ***] 调用带有ID生成的新转换函数
        new_output_list = transform_old_format_to_new_with_id(original_output)

        output_str = json.dumps(new_output_list, ensure_ascii=False)

        alpaca_item = {
            "instruction": INSTRUCTION_TEMPLATE,
            "input": input_text,
            "output": output_str,
            "system": SYSTEM_PROMPT
        }
        alpaca_data.append(alpaca_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"处理完成！成功将 {len(alpaca_data)} 条记录转换为Alpaca格式，并保存到 '{output_path}'。")
    if alpaca_data:
        print("\n--- 第一条转换后的数据示例 ---")
        print(json.dumps(alpaca_data[0], ensure_ascii=False, indent=4))


# -----------------------------------------------------------------------------
# 3. 主程序入口
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # 定义输入和输出文件名
    input_filename = "../cmnee_llmf/train.json"
    output_filename = "../CMNEE_LLMF_EE/train.json"

    # 创建一个示例输入文件，如果它不存在
    if not os.path.exists(input_filename):
        print(f"未找到输入文件 '{input_filename}'。正在创建一个示例文件...")
        sample_data_for_script = [
            {
                "instruction": "",
                "input": "据报道，俄罗斯海军于2023年10月26日在黑海地区试射了一枚‘锆石’高超音速导弹。",
                "output": "{\"试验\": [{\"trigger\": \"试射\", \"arguments\": {\"主体\": [\"俄罗斯海军\"], \"装备\": [\"‘锆石’高超音速导弹\"], \"日期\": [\"2023年10月26日\"], \"位置\": [\"黑海地区\"]}}], \"演习\": [], \"部署\": [], \"支援\": [], \"意外事故\": [], \"展示\": [], \"冲突\": [], \"伤亡\": []}"
            },
            {
                "instruction": "",
                "input": "鉴于一名驻阿富汗美国士兵在训练中因XM25式25毫米半自动榴弹发射器出现故障而受伤，美国陆军已经撤回其投放到阿富汗的XM25榴弹发射器样枪及其配用的空爆弹。2013年1月，美国陆军将第二批XM25半自动榴弹发射器样枪及其空爆弹投放到阿富汗进行前方作战评估，第一批5具XM25榴弹发射器自2010年11月～2012年4月部署至阿富汗。美国陆军发言人称，在2013年2月进行的实弹射击试验中，一具XM25半自动榴弹发射器因供弹过程出现故障而导致底火过早地为25毫米空爆榴弹点火，尽管手工制成的空爆弹的空爆战斗部没有起爆，但士兵确实受伤，且该枪无法使用，训练立即停止。事故发生后，陆军从阿富汗撤回了所有的XM25半自动榴弹发射器样枪及其弹药，以便调查事故根源和总结改进方案。陆军已经终止了XM25榴弹发射器的野外试验，旨在进一步制定改进方案，确保武器安全。一旦查清事故原因并提出改进方案，美国陆军将决定该武器何时再次投入使用。同时，XM25半自动榴弹发射器的工程与制造研发阶段工作继续开展，以便进一步完善和改进武器设计。",
                "output": "{\"试验\": [{\"trigger\": \"试验\", \"arguments\": {\"主体\": [\"美国陆军\"], \"装备\": [\"XM25半自动榴弹发射器\"], \"日期\": [\"2013年2月\"], \"位置\": [\"NAN\"]}}], \"演习\": [], \"部署\": [{\"trigger\": \"投放\", \"arguments\": {\"主体\": [\"美国陆军\"], \"军事力量\": [\"XM25半自动榴弹发射器样枪\", \"空爆弹\"], \"日期\": [\"2013年1月\"], \"位置\": [\"阿富汗\"]}}, {\"trigger\": \"部署\", \"arguments\": {\"主体\": [\"美国陆军\"], \"军事力量\": [\"XM25榴弹发射器\"], \"日期\": [\"2010年11月～2012年4月\"], \"位置\": [\"阿富汗\"]}}], \"支援\": [], \"意外事故\": [{\"trigger\": \"点火\", \"arguments\": {\"主体\": [\"XM25半自动榴弹发射器\"], \"事故后果\": [\"士兵确实受伤\", \"该枪无法使用\"], \"日期\": [\"2013年2月\"], \"位置\": [\"NAN\"]}}], \"展示\": [], \"冲突\": [], \"伤亡\": [{\"trigger\": \"受伤\", \"arguments\": {\"主体\": [\"驻阿富汗美国士兵\"], \"数量\": [\"一名\"], \"日期\": [\"NAN\"], \"位置\": [\"NAN\"]}}]}"
            }
        ]
        with open(input_filename, 'w', encoding='utf-8') as f:
            json.dump(sample_data_for_script, f, ensure_ascii=False, indent=2)
        print(f"示例文件 '{input_filename}' 已创建。")

    # 运行转换过程
    convert_to_alpaca_format(input_filename, output_filename)