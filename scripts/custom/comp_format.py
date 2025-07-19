import json
import re
from json import JSONDecodeError


def fix_predict_json(predict_str):
    """尝试修复受损的predict字段格式，提取"causality_list"数组中的各个对象。
    返回修复后的字典，格式为：{"causality_list": [obj1, obj2, ...]}
    """
    try:
        # 定位"causality_list"键及其数组开始位置
        key_index = predict_str.find('"causality_list"')
        if key_index == -1:
            raise ValueError("没有找到'causality_list'键")
        bracket_start = predict_str.find('[', key_index)
        if bracket_start == -1:
            raise ValueError("没有找到'['开始的causality_list数组")

        # 找到对应的数组结束位置（简单匹配）
        count = 0
        end_index = bracket_start
        for i in range(bracket_start, len(predict_str)):
            char = predict_str[i]
            if char == '[':
                count += 1
            elif char == ']':
                count -= 1
                if count == 0:
                    end_index = i
                    break
        array_content = predict_str[bracket_start + 1:end_index]

        # 利用括号匹配算法，提取数组内的每个JSON对象字符串
        objects = []
        obj_start = None
        brace_count = 0
        for i, char in enumerate(array_content):
            if char == '{':
                if brace_count == 0:
                    obj_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and obj_start is not None:
                    obj_str = array_content[obj_start:i + 1]
                    objects.append(obj_str)
                    obj_start = None

        # 尝试解析每个提取的对象
        causality_list = []
        for obj in objects:
            try:
                parsed_obj = json.loads(obj)
                causality_list.append(parsed_obj)
            except Exception:
                # 尝试修复引号问题
                try:
                    fixed_obj = obj.replace("'", "\"")
                    parsed_obj = json.loads(fixed_obj)
                    causality_list.append(parsed_obj)
                except Exception:
                    continue
        return {"causality_list": causality_list}
    except Exception as e:
        raise e


def process_data(input_path, output_path, error_file_path):
    output = []
    error_lines = []
    doc_id = 0

    with open(input_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            current_line = line.strip()
            try:
                # 解析原始JSONL
                data = json.loads(current_line)
            except Exception:
                error_lines.append(current_line)
                continue

            # 提取prompt字段中指定部分内容
            try:
                prompt_content = data.get('prompt', '')
                text_match = re.search(
                    r'不要输出think过程[\n\s]*(.*?)<｜Assistant｜>',
                    prompt_content,
                    re.DOTALL
                )
                text_content = text_match.group(1).strip() if text_match else ""
            except Exception:
                error_lines.append(current_line)
                continue

            # 处理predict字段
            causality_list = []
            try:
                predict_str = data['predict'].strip()
            except KeyError:
                error_lines.append(current_line)
                continue

            try:
                predict_data = json.loads(predict_str)
            except JSONDecodeError:
                try:
                    fixed_str = predict_str.replace("'", "\"")
                    predict_data = json.loads(fixed_str)
                except Exception:
                    # 如果仍然失败，尝试调用fix_predict_json函数修复
                    try:
                        predict_data = fix_predict_json(predict_str)
                    except Exception:
                        error_lines.append(current_line)
                        continue

            # 清洗NAN值并映射为新结构
            try:
                for causality in predict_data.get('causality_list', []):
                    for role in ['cause', 'effect']:
                        event = causality.get(role, {})
                        for key in event:
                            if isinstance(event[key], str) and event[key].upper() == 'NAN':
                                event[key] = ""
                    new_causality = {
                        "causality_type": "直接",
                        "cause": {
                            "actor": causality['cause'].get('subject', ''),
                            "class": causality['cause'].get('event_type', ''),
                            "action": causality['cause'].get('trigger', ''),
                            "time": causality['cause'].get('date', ''),
                            "location": causality['cause'].get('location', ''),
                            "object": causality['cause'].get('object', '')
                        },
                        "effect": {
                            "actor": causality['effect'].get('subject', ''),
                            "class": causality['effect'].get('event_type', ''),
                            "action": causality['effect'].get('trigger', ''),
                            "time": causality['effect'].get('date', ''),
                            "location": causality['effect'].get('location', ''),
                            "object": causality['effect'].get('object', '')
                        }
                    }
                    causality_list.append(new_causality)
            except Exception:
                error_lines.append(current_line)
                continue

            # 构建最终文档
            output.append({
                "document_id": doc_id,
                "text": text_content,
                "causality_list": causality_list
            })
            doc_id += 1

    # 写入格式化JSON（严格按照给定格式换行和缩进）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4, separators=(',', ': '))

    # 将出错的原始行写入error.jsonl，每行保持原始格式
    with open(error_file_path, 'w',
              encoding='utf-8') as f:
        for err_line in error_lines:
            f.write(err_line + "\n")


if __name__ == '__main__':
    # 示例调用：
    input = 'saves/DeepSeek/DeepSeek-R1-Distill-Llama-8B-cmnee-cause/qlora/predict/100cause+200type/DSN-DEMAND/generated_predictions.jsonl'
    output = 'saves/DeepSeek/DeepSeek-R1-Distill-Llama-8B-cmnee-cause/qlora/predict/100cause+200type/DSN-DEMAND/compte/test.json'
    error = "saves/DeepSeek/DeepSeek-R1-Distill-Llama-8B-cmnee-cause/qlora/predict/100cause+200type/DSN-DEMAND/compte/errors.jsonl"
    process_data(input, output, error)
    # 若需要使用命令行参数，可取消下面的注释：
    # import sys
    # if len(sys.argv) < 3:
    #     print("Usage: python clean_data.py <input_file.jsonl> <output_file.jsonl>")
    #     sys.exit(1)
    # process_data(sys.argv[1], sys.argv[2])
