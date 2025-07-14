import json
import argparse
import os
import re
from tqdm import tqdm


def extract_and_minify_json(raw_response):
    """
    从模型的原始输出字符串中提取JSON，并将其压缩为单行、无多余空格的字符串。
    """
    if not isinstance(raw_response, str):
        return "[]"

    # 1. 优先尝试匹配 ```json ... ``` 代码块
    match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_response)
    if match:
        json_str = match.group(1).strip()
    else:
        # 2. 否则，尝试寻找第一个 '[' 或 '{' 开始的JSON结构
        start_brace = raw_response.find('[')
        start_curly = raw_response.find('{')
        if start_brace == -1 and start_curly == -1:
            return "[]"

        start_index = start_brace if start_brace != -1 and (
                    start_brace < start_curly or start_curly == -1) else start_curly
        json_str = raw_response[start_index:]

    # --- <<< UPGRADED SECTION >>> ---
    # 3. 解析并重新序列化为紧凑格式，以消除所有不必要的空格和换行符
    try:
        # 将提取出的（可能带格式的）JSON字符串加载为Python对象
        python_obj = json.loads(json_str)
        # 将Python对象重新转储为紧凑的JSON字符串
        # separators=(',', ':') 是实现压缩的关键，它消除了所有多余的空格
        compact_json_str = json.dumps(python_obj, ensure_ascii=False, separators=(',', ':'))
        return compact_json_str
    except json.JSONDecodeError:
        # 如果提取出的字符串本身就是无效的JSON，则返回空列表字符串
        return "[]"


def main(prediction_file, output_file=None):
    """
    读取模型的预测文件，清洗并压缩'predict'字段，并保存为新文件。
    """
    # <<< UPGRADED SECTION >>>: 智能生成输出文件名
    if output_file is None:
        base, ext = os.path.splitext(prediction_file)
        # 例如: 'test_1shot.jsonl' -> 'test_1shot_cleaned.jsonl'
        output_file = f"{base}_cleaned{ext}"

    cleaned_data = []

    print(f"Reading raw predictions from: {prediction_file}")
    with open(prediction_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Cleaning and Minifying"):
        try:
            data = json.loads(line.strip())
            raw_prediction = data.get("predict", "")

            # 清洗、提取并压缩JSON
            compact_json = extract_and_minify_json(raw_prediction)

            # 用压缩后的纯净JSON替换原始的'predict'字段
            data['predict'] = compact_json

            cleaned_data.append(data)

        except Exception as e:
            print(f"Error processing line: {line.strip()[:100]}... Error: {e}")
            continue

    print(f"\nWriting cleaned and minified data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n--- Post-processing Complete ---")
    print(f"Total samples processed: {len(cleaned_data)}")
    print(f"You can now run the evaluation script on the cleaned file: {output_file}")

    # 打印一个样本对比效果
    if lines and cleaned_data:
        print("\n--- Example of Cleaning ---")
        print("Before (Raw predict):")
        print(json.loads(lines[0])['predict'][:200] + '...')
        print("\nAfter (Cleaned predict):")
        print(cleaned_data[0]['predict'])
        print("---------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and minify model prediction outputs to extract pure JSON.")
    parser.add_argument("--prediction_file", type=str, required=True,
                        help="The raw prediction file from the model (JSONL format).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the cleaned file. (Default: [prediction_file]_cleaned.jsonl)")

    args = parser.parse_args()
    main(args.prediction_file, args.output_file)