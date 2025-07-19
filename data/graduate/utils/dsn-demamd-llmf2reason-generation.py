import json
import os


def transform_causal_data_to_system_format(input_file_path: str, output_file_path: str):
    """
    将因果关系抽取数据转换为包含'system'字段的结构化任务数据，并保存为单个JSON列表文件。

    该函数会:
    1. 读取原始的JSON列表文件。
    2. 对每个样本，解析其'input'和'output'字段。
    3. 将每个因果对拆分为一个独立的任务。
    4. 为每个任务生成独立的'system'角色提示和'instruction'任务指令。
    5. 构建包含'context'和'causality_list'的JSON字符串作为'input'字段。
    6. 将所有生成的新任务收集到列表中，并作为JSON数组写入输出文件。
    """

    # 1. 定义新的 System Prompt 和 Instruction Template
    new_system_prompt = "你是一名军事事件间因果逻辑分析专家。"
    new_instruction_template = (
        "请分析'input'中的JSON数据，并基于'context'文本，解释'causality_list'中的'cause'事件为何导致了'effect'事件。"
        "你的解释需简洁清晰，直接阐述逻辑关联，并控制在50字以内。"
    )

    all_new_records = []  # 用于收集所有转换后的任务
    processed_pairs_count = 0
    original_items_count = 0

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            original_data = json.load(f_in)
            original_items_count = len(original_data)

            for item in original_data:
                try:
                    input_content = json.loads(item['input'])
                    output_content = json.loads(item['output'])

                    context_text = input_content.get('text', '')

                    if not isinstance(output_content, list):
                        continue

                    for causal_pair in output_content:
                        if 'cause' not in causal_pair or 'effect' not in causal_pair:
                            continue

                        new_input_dict = {
                            "context": context_text,
                            "causality_list": [{
                                "cause": causal_pair['cause'],
                                "effect": causal_pair['effect']
                            }]
                        }

                        new_input_str = json.dumps(new_input_dict, ensure_ascii=False, separators=(',', ':'))

                        # 2. 构建包含 system 字段的新记录
                        new_record = {
                            "instruction": new_instruction_template,
                            "input": new_input_str,
                            "output": "",
                            "system": new_system_prompt
                        }

                        all_new_records.append(new_record)
                        processed_pairs_count += 1

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Skipping an item due to a processing error: {e}")
                    continue

        # 3. 确保输出目录存在
        output_dir = os.path.dirname(output_file_path)
        if output_dir:  # 确保路径不是空字符串
            os.makedirs(output_dir, exist_ok=True)

        # 4. 所有数据处理完毕后，一次性写入JSON文件
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(all_new_records, f_out, ensure_ascii=False, indent=4)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
        return

    print("\n--- Data Transformation Complete ---")
    print(f"Input file: '{input_file_path}'")
    print(f"Output file: '{output_file_path}'")
    print(f"Total original items processed: {original_items_count}")
    print(f"Total causal pairs extracted (new tasks created): {processed_pairs_count}")
    print("The output file is a single JSON file containing a list of all tasks, formatted with a 'system' field.")


if __name__ == '__main__':
    # 使用您指定的输入和输出文件路径
    input_filename = 'data/graduate/DSN-DEMAMD-LLMF/train.json'
    output_filename = 'data/graduate/DSN-DEMAMD-reason/train.json'

    if not os.path.exists(input_filename):
        # 提供了更精确的错误提示
        print(f"Error: Input file not found at the specified path: '{input_filename}'")
    else:
        transform_causal_data_to_system_format(input_filename, output_filename)