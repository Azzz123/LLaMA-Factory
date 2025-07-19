import argparse
import ast  # 引入ast模块，用于安全地解析字符串化的Python字面量
import json
import os
import re

from tqdm import tqdm


def extract_and_purify_json(raw_response):
    """【通用版】从模型的原始、混乱的输出字符串中，稳健地提取并修复JSON。
    """
    if not isinstance(raw_response, str):
        return "[]"

    # 1. 【升级】使用更通用的正则表达式，匹配 ```json ... ``` 或 ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response, re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        # 2. 如果没有代码块，仍然尝试寻找第一个 '[' 开始的JSON列表
        start_brace = raw_response.find('[')
        if start_brace == -1:
            return "[]"  # 如果连'['都找不到，直接放弃
        # 从第一个'['开始，尝试找到与之匹配的最后一个']'
        # 这是一个更稳健的方法，避免截取到末尾的垃圾字符
        try:
            # 这是一个简化的平衡括号查找，对于大多数情况有效
            open_brackets = 0
            for i, char in enumerate(raw_response[start_brace:]):
                if char == '[':
                    open_brackets += 1
                elif char == ']':
                    open_brackets -= 1
                if open_brackets == 0:
                    json_str = raw_response[start_brace: start_brace + i + 1]
                    break
            else:  # 如果循环正常结束（没有break），说明括号不匹配
                json_str = "[]"
        except Exception:
            json_str = "[]"

    # 3. 【核心升级】尝试解析，并处理Qwen那种内部值为字符串的特殊情况
    try:
        python_obj = json.loads(json_str)

        # 检查并修复内部结构
        if isinstance(python_obj, list):
            for item in python_obj:
                if isinstance(item, dict):
                    for key in ['cause', 'effect']:
                        if key in item and isinstance(item[key], str):
                            try:
                                # 尝试用ast.literal_eval将字符串'{"key":...}'转为真正的字典
                                # 它能安全地处理单引号和双引号的字典字符串
                                item[key] = ast.literal_eval(item[key])
                            except (ValueError, SyntaxError, MemoryError, TypeError):
                                # 如果转换失败，说明它可能就是一个普通字符串，保持原样
                                pass

        # 4. 最终，将修复好的Python对象重新序列化为紧凑的JSON字符串
        compact_json_str = json.dumps(python_obj, ensure_ascii=False, separators=(',', ':'))
        return compact_json_str

    except json.JSONDecodeError:
        return "[]"


def main(prediction_file, output_file=None):
    """读取模型的预测文件，用通用清洗逻辑处理'predict'字段，并保存为新文件。
    """
    if output_file is None:
        base, ext = os.path.splitext(prediction_file)
        output_file = f"{base}_cleaned{ext}"

    cleaned_data = []

    print(f"Reading raw predictions from: {prediction_file}")
    with open(prediction_file, encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Universally Cleaning and Minifying"):
        try:
            data = json.loads(line.strip())
            raw_prediction = data.get("predict", "")

            # 使用全新的通用清洗函数
            compact_json = extract_and_purify_json(raw_prediction)

            data['predict'] = compact_json
            cleaned_data.append(data)

        except Exception as e:
            print(f"Error processing line: {line.strip()[:100]}... Error: {e}")
            continue

    print(f"\nWriting cleaned data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n--- Post-processing Complete ---")
    print(f"Total samples processed: {len(cleaned_data)}")
    print(f"You can now run the evaluation script on the cleaned file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Universally clean and minify model prediction outputs to extract pure JSON.")
    parser.add_argument("--prediction_file", type=str, required=True,
                        help="The raw prediction file from the model (JSONL format).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the cleaned file. (Default: [prediction_file]_cleaned.jsonl)")

    args = parser.parse_args()
    main(args.prediction_file, args.output_file)
