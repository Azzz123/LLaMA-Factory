import json
import re


def extract_last_json_block(text):
    # 匹配所有 ```json\n...``` 块并取最后一个
    matches = re.findall(r'```json\n(.*?)```', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def format_json_string(json_str):
    try:
        json_obj = json.loads(json_str)
        # 如果解析结果为列表，则包装为{"causality_list": [...]}
        if isinstance(json_obj, list):
            json_obj = {"causality_list": json_obj}
        return json.dumps(json_obj, ensure_ascii=False, separators=(',', ':'))
    except json.JSONDecodeError:
        # 如果解析失败，不做额外处理，直接返回原字符串
        return json_str


def process_predict_field(predict_str):
    # 尝试直接解析 predict 字符串（因为部分数据直接就是字符串化的JSON数组）
    try:
        json_obj = json.loads(predict_str)
        if isinstance(json_obj, list):
            # 如果是列表，则包装为目标格式
            return json.dumps({"causality_list": json_obj}, ensure_ascii=False, separators=(',', ':'))
        # 如果本身已经是对象则直接返回标准化后的字符串
        return json.dumps(json_obj, ensure_ascii=False, separators=(',', ':'))
    except json.JSONDecodeError:
        # 如果直接解析失败，则尝试提取最后一个 JSON 块
        json_block = extract_last_json_block(predict_str)
        if not json_block:
            return predict_str
        return format_json_string(json_block)


def main(input_path, output_path):
    with open(input_path, encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                if 'predict' in data:
                    processed = process_predict_field(data['predict'])
                    data['predict'] = processed
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"解析失败的行：{line}")


if __name__ == "__main__":
    # 修改以下文件路径
    input_file = "saves/DeepSeek/DeepSeek-R1-Distill-Llama-8B/un-ft/predict/generated_predictions.jsonl"
    output_file = "saves/DeepSeek/DeepSeek-R1-Distill-Llama-8B/un-ft/predict/cleaned_predictions.jsonl"
    main(input_file, output_file)
