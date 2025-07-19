import json
import random
import argparse
import os
from tqdm import tqdm


def load_json_list(file_path):
    """加载一个包含JSON对象的列表的文件。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"错误: 加载 {file_path} 失败: {e}")
        return None


def save_json_list(data, file_path):
    """将一个数据列表保存为JSON文件。"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def format_exemplar_as_io_pair(exemplar_item):
    """将一个示范样本格式化为 Input -> Response 的文本对。"""
    input_str = exemplar_item.get('input', '')
    output_str = exemplar_item.get('output', '')
    return f"Input:\n{input_str}\n\nResponse:\n{output_str}"


def get_specific_exemplar_pool(train_data):
    """
    【核心升级】根据您指定的严格标准筛选示范样例池。
    """
    print("应用固定的样例筛选标准...")
    print("- 候选对数量必须等于4")
    print("- 必须存在真实的因果关系 (output不为空)")

    exemplar_pool = []
    for item in train_data:
        try:
            # 解析input和output字段
            input_dict = json.loads(item['input'])
            output_list = json.loads(item['output'])

            # 应用筛选条件
            if (len(input_dict.get('candidate_pairs', [])) == 4 and
                    isinstance(output_list, list) and len(output_list) > 0):
                exemplar_pool.append(item)
        except (json.JSONDecodeError, TypeError):
            continue  # 如果解析失败则跳过

    print(f"筛选完成，共找到 {len(exemplar_pool)} 个符合条件的样例。")
    if not exemplar_pool:
        raise ValueError("错误: 未能从训练集中找到任何符合指定条件的样例。")

    return exemplar_pool


def build_nested_few_shot_datasets(exemplar_pool, test_data, log_handle=None):
    """
    【核心升级】为每个测试样本构建嵌套式的k=1,3,5数据集。
    """
    datasets = {1: [], 3: [], 5: []}

    if len(exemplar_pool) < 5:
        raise ValueError(f"示范样例池不足 ({len(exemplar_pool)})，无法为每个测试样本选择5个不同的样例。")

    print("\n开始为每个测试样本构建嵌套式(k=1,3,5)的Few-shot提示...")
    for test_idx, test_item in enumerate(tqdm(test_data, desc="处理测试样本")):
        # 为每个测试样本随机选择5个独一无二的、固定的示范样例
        # 这是实现嵌套关系的关键
        base_5_exemplars = random.sample(exemplar_pool, 5)

        # 记录这5个基础样例
        if log_handle:
            log_handle.write(f"--- 测试项索引: {test_idx} ---\n")
            log_handle.write("为其分配的5个基础示范样例:\n")
            log_handle.write(json.dumps(base_5_exemplars, indent=2, ensure_ascii=False))
            log_handle.write("\n------------------------------------------\n\n")

        # 循环构建k=1, 3, 5的数据
        for k in [1, 3, 5]:
            # 从5个基础样例中，按顺序取出k个
            exemplars_for_this_k = base_5_exemplars[:k]

            task_instruction = test_item.get('instruction', '')

            demonstrations = [f"--- Example {i + 1} ---\n{format_exemplar_as_io_pair(ex)}" for i, ex in
                              enumerate(exemplars_for_this_k)]
            demonstration_block = "\n\n".join(demonstrations)

            new_instruction = (
                f"{task_instruction}\n\n"
                f"这里有 {k} 个任务示范，请学习它们的逻辑和格式:\n\n"
                f"{demonstration_block}\n\n"
                f"--- 你的任务 ---\n"
                f"现在，请根据以上示范，对以下输入应用相同的逻辑。"
            )

            new_item = {
                "instruction": new_instruction, "input": test_item["input"],
                "output": test_item["output"], "system": test_item.get("system", "")
            }
            datasets[k].append(new_item)

    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="一次性构建嵌套式的k=1, 3, 5的Few-shot测试文件，使用固定的样例选择标准。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--train_file", type=str, required=True, help="训练集JSON文件路径。")
    parser.add_argument("--test_file", type=str, required=True, help="原始测试集JSON文件路径。")
    parser.add_argument("--output_dir", type=str, required=True, help="保存生成的测试文件的目录。")
    parser.add_argument("--seed", type=int, default=42, help="用于选择样例的随机种子，确保可复现性。")

    args = parser.parse_args()

    print(f"使用随机种子: {args.seed}")
    random.seed(args.seed)

    train_dataset = load_json_list(args.train_file)
    test_dataset = load_json_list(args.test_file)

    os.makedirs(args.output_dir, exist_ok=True)

    if train_dataset and test_dataset:
        # 1. 根据固定标准筛选示范样例池
        exemplar_pool = get_specific_exemplar_pool(train_dataset)

        log_filename = os.path.join(args.output_dir, f"exemplar_log_nested_seed_{args.seed}.jsonl")

        with open(log_filename, 'w', encoding='utf-8') as log_f:
            log_f.write(f"# 嵌套式样例日志: 种子={args.seed}\n")
            log_f.write(f"# 筛选标准: candidate_pairs==4, output不为空\n\n")

            # 2. 一次性构建k=1,3,5的嵌套数据集
            all_datasets = build_nested_few_shot_datasets(
                exemplar_pool=exemplar_pool, test_data=test_dataset,
                log_handle=log_f
            )

            # 3. 分别保存文件
            for k, data in all_datasets.items():
                output_path = os.path.join(args.output_dir, f"test_k{k}.json")
                save_json_list(data, output_path)
                print(f"成功创建嵌套式 k={k} 数据集，已保存至: {output_path}")

        print(f"\n--- 所有任务完成 ---")
        print(f"所有选择的样例日志已保存至: {log_filename}")