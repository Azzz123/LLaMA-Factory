import json

def merge_datasets(train_file, valid_file, output_file):
    """
    合并 train.json 和 valid.json 文件中的数据，并保存为新的 JSON 文件。

    参数：
    train_file (str): 训练数据文件路径
    valid_file (str): 验证数据文件路径
    output_file (str): 输出合并后的文件路径
    """
    # 读取 train 数据
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 读取 valid 数据
    with open(valid_file, 'r', encoding='utf-8') as f:
        valid_data = json.load(f)

    # 合并两个数据集
    merged_data = train_data + valid_data

    # 保存合并后的数据到新的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

# 调用函数
train_file = '../cmnee_llmf/train.json'  # 训练数据文件路径
valid_file = '../cmnee_llmf/valid.json'  # 验证数据文件路径
output_file = '../cmnee_llmf/train1.json'  # 输出合并后的文件路径

merge_datasets(train_file, valid_file, output_file)