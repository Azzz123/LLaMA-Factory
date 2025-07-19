import json
import argparse
from collections import defaultdict

# 中英文映射字典
convert2chinese = {
    # 事件类型映射
    "Experiment": "试验",
    "Manoeuvre": "演习",
    "Deploy": "部署",
    "Support": "支援",
    "Accident": "意外事故",
    "Exhibit": "展示",
    "Conflict": "冲突",
    "Injure": "伤亡",

    # 论元角色映射
    "Subject": "主体",
    "Equipment": "装备",
    "Date": "日期",
    "Location": "位置",
    "Area": "区域",
    "Content": "演习内容",
    "Militaryforce": "军事力量",
    "Object": "客体",
    "Materials": "物资",
    "Result": "事故后果",
    "Quantity": "数量"
}

# 事件类型参数结构定义
event_args_structure = {
    "试验": ["主体", "装备", "日期", "位置"],
    "演习": ["主体", "演习内容", "日期", "区域"],
    "部署": ["主体", "军事力量", "日期", "位置"],
    "支援": ["主体", "客体", "物资", "日期"],
    "意外事故": ["主体", "事故后果", "日期", "位置"],
    "展示": ["主体", "装备", "日期", "位置"],
    "冲突": ["主体", "客体", "日期", "位置"],
    "伤亡": ["主体", "数量", "日期", "位置"]
}

# 示例触发词和论元
example_trigger_and_arguments = {
    "试验": {"trigger": "试验", "arguments": {"主体": ["NAN"], "装备": ["NAN"], "日期": ["NAN"], "位置": ["NAN"]}},
    "演习": {"trigger": "演习", "arguments": {"主体": ["NAN"], "演习内容": ["NAN"], "日期": ["NAN"], "区域": ["NAN"]}},
    "部署": {"trigger": "部署", "arguments": {"主体": ["NAN"], "军事力量": ["NAN"], "日期": ["NAN"], "位置": ["NAN"]}},
    "支援": {"trigger": "支援", "arguments": {"主体": ["NAN"], "客体": ["NAN"], "物资": ["NAN"], "日期": ["NAN"]}},
    "意外事故": {"trigger": "事故", "arguments": {"主体": ["NAN"], "事故后果": ["NAN"], "日期": ["NAN"], "位置": ["NAN"]}},
    "展示": {"trigger": "展示", "arguments": {"主体": ["NAN"], "装备": ["NAN"], "日期": ["NAN"], "位置": ["NAN"]}},
    "冲突": {"trigger": "冲突", "arguments": {"主体": ["NAN"], "客体": ["NAN"], "日期": ["NAN"], "位置": ["NAN"]}},
    "伤亡": {"trigger": "伤害", "arguments": {"主体": ["NAN"], "数量": ["NAN"], "日期": ["NAN"], "位置": ["NAN"]}}
}


def convert_event(event):
    """转换单个事件"""
    # 转换事件类型
    cn_type = convert2chinese.get(event["event_type"], "其他")

    # 构建参数模板
    arg_template = {arg: [] for arg in event_args_structure.get(cn_type, [])}

    # 转换论元
    for arg in event["arguments"]:
        cn_role = convert2chinese.get(arg["role"], arg["role"])
        if cn_role in arg_template:
            arg_template[cn_role].append(arg["text"])

    # 确保每个角色至少是一个值（即使没有论元）
    for role in arg_template:
        if not arg_template[role]:
            arg_template[role] = ["NAN"]

    return {
        "trigger": event["trigger"]["text"],
        "arguments": arg_template
    }


def process_item(item, instruction_template):
    """处理单个数据项"""
    # 初始化事件容器
    event_container = defaultdict(list)

    # 转换所有事件
    for event in item["event_list"]:
        converted = convert_event(event)
        cn_type = convert2chinese.get(event["event_type"], "其他")
        if cn_type in event_args_structure:
            event_container[cn_type].append(converted)

    # 填充空事件类型
    for event_type in event_args_structure.keys():
        if event_type not in event_container:
            event_container[event_type] = []

    # 按照system定义的顺序对事件进行排序
    ordered_event_container = {event_type: event_container[event_type] for event_type in event_args_structure.keys()}

    return {
        "instruction": instruction_template,
        "input": item["text"],
        "output": json.dumps(ordered_event_container, ensure_ascii=False),
        "system": json.dumps({**event_args_structure, **example_trigger_and_arguments}, ensure_ascii=False)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="../cmnee/test.json", help="input json file")
    parser.add_argument("-o", "--output", default="../cmnee_llmf/test1.json", help="output json file")
    parser.add_argument("--instruction", default="你是一名专门进行文本内多事件抽取的专家。"
                                                 "请按以下要求从input中抽取出符合system定义格式的所有事件："
                                                 "\n1. 不存在的事件返回空列表，不存在的论元返回NAN列表"
                                                 "\n2. 装备名称须带型号完整提取"
                                                 "\n3. 请严格按照system定义的JSON格式输出"
                                                 "\n4. 不需要显示输出think的过程",
                        help="自定义指令模板")

    args = parser.parse_args()

    # 读取原始数据
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理数据
    processed = [process_item(item, args.instruction) for item in data]

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
