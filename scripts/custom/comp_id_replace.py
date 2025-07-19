import json
import os

# 文件路径
file1_path = 'data/DSN-DEMAND/test.json'
file2_path = 'saves/DeepSeek/DeepSeek-R1-Distill-Llama-8B-cmnee-cause/qlora/predict/100cause+200type/DSN-DEMAND/compte/test.json'

# 获取 file2 目录路径
file2_dir = os.path.dirname(file2_path)
updated_file_path = os.path.join(file2_dir, 'updated_test.json')
unmatched_file_path = os.path.join(file2_dir, 'unmatched_test.json')

# 读取 file1 并建立 text 到 document_id 的映射
with open(file1_path, encoding='utf-8') as f1:
    file1_data = json.load(f1)
text_to_id = {item["text"]: item["document_id"] for item in file1_data}

# 读取 file2 并替换 document_id
with open(file2_path, encoding='utf-8') as f2:
    file2_data = json.load(f2)

matched_count = 0
unmatched_data = []

for item in file2_data:
    if item["text"] in text_to_id:
        item["document_id"] = text_to_id[item["text"]]
        matched_count += 1
    else:
        unmatched_data.append(item)

# 保存更新后的 file2 数据
with open(updated_file_path, 'w', encoding='utf-8') as out_file:
    json.dump(file2_data, out_file, ensure_ascii=False, indent=4)

# 保存未匹配的 file2 数据
with open(unmatched_file_path, 'w', encoding='utf-8') as unmatched_file:
    json.dump(unmatched_data, unmatched_file, ensure_ascii=False, indent=4)

# 输出匹配结果
total_file2_entries = len(file2_data)
print(f"总条目数: {total_file2_entries}")
print(f"成功匹配并替换的条目数: {matched_count}")
print(f"未匹配的条目数: {len(unmatched_data)}")
print(f"更新后的数据已保存到: {updated_file_path}")
print(f"未匹配的数据已保存到: {unmatched_file_path}")
