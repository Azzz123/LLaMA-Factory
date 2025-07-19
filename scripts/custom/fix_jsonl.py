import json
import re
from json import JSONDecodeError


def diagnose_error(error_msg):
    error_types = []
    if 'Unterminated string' in error_msg:
        error_types.append('unterminated_string')
    if 'Expecting \',\' delimiter' in error_msg:
        error_types.append('missing_comma')
    if 'Expecting \'}\' delimiter' in error_msg or 'Expecting \']\' delimiter' in error_msg:
        error_types.append('unclosed_bracket')
    if 'Expecting property name enclosed in double quotes' in error_msg:
        error_types.append('unquoted_key')
    if 'Invalid control character' in error_msg:
        error_types.append('invalid_control_char')
    return error_types


def fix_unterminated_string(json_str, pos):
    # 在指定位置插入缺失的引号
    return json_str[:pos] + '"' + json_str[pos:]


def fix_unclosed_bracket(json_str):
    # 统计括号数量
    open_braces = json_str.count('{') - json_str.count('}')
    open_brackets = json_str.count('[') - json_str.count(']')

    # 添加缺失的闭合括号
    if open_braces > 0:
        json_str += '}' * open_braces
    if open_brackets > 0:
        json_str += ']' * open_brackets

    return json_str


def fix_unquoted_keys(json_str):
    # 匹配未加引号的键名
    pattern = r'(?<![{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'
    return re.sub(pattern, r'"\1":', json_str)


def fix_missing_comma(json_str, pos):
    # 在指定位置插入逗号
    return json_str[:pos] + ',' + json_str[pos:]


def fix_invalid_control_chars(json_str):
    # 替换控制字符
    return re.sub(r'[\x00-\x1F]', lambda m: f'\\u{ord(m.group()):04x}', json_str)


def smart_json_repair(json_str):
    original_str = json_str
    max_attempts = 3

    for _ in range(max_attempts):
        try:
            json.loads(json_str)
            return json_str
        except JSONDecodeError as e:
            error_types = diagnose_error(str(e))
            original_error = str(e)
            pos = e.pos
            modified = False

            if 'unterminated_string' in error_types:
                json_str = fix_unterminated_string(json_str, pos)
                modified = True
            if 'unclosed_bracket' in error_types:
                json_str = fix_unclosed_bracket(json_str)
                modified = True
            if 'unquoted_key' in error_types:
                json_str = fix_unquoted_keys(json_str)
                modified = True
            if 'missing_comma' in error_types:
                json_str = fix_missing_comma(json_str, pos)
                modified = True
            if 'invalid_control_char' in error_types:
                json_str = fix_invalid_control_chars(json_str)
                modified = True

            if not modified:  # 未知错误类型尝试通用修复
                json_str = fix_unclosed_bracket(json_str)
                json_str = fix_unquoted_keys(json_str)
                json_str = fix_invalid_control_chars(json_str)

            if json_str == original_str:
                break  # 避免无限循环

    return json_str


def process_error_info(input_path):
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    fixed_lines = []
    for item in data:
        if 'line' in item and 'error' in item:
            original_line = item['line']
            fixed_line = smart_json_repair(original_line)
            fixed_lines.append({'fixed': fixed_line})

    output_path = input_path.replace('error_info.json', 'fixed_lines.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python fix_json.py <error_info.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    process_error_info(input_file)
