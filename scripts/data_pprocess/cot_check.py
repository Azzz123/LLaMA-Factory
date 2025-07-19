import json
import os
import argparse
from tqdm import tqdm
from collections import Counter
import pandas as pd


# --- 辅助函数 ---

def is_valid_json_list(s):
    """检查一个字符串是否是合法的JSON列表。"""
    s = s.strip()
    if not s.startswith('[') or not s.endswith(']'):
        return False
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


# --- 主检查逻辑 ---

def inspect_dataset(file_path):
    """
    对指定的数据集文件进行全面的健康检查。
    """
    print(f"🩺 开始对数据集进行健康体检: {file_path}\n")

    # --- 1. 初始化报告和计数器 ---
    report = {
        "file_path": file_path,
        "total_samples": 0,
        "error_counts": Counter(),
        "error_samples": [],
        "statistics": {
            "event_type_distribution": Counter(),
            "samples_with_no_events": 0,
            "output_token_lengths": []
        }
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        report["total_samples"] = len(data)
    except Exception as e:
        print(f"❌ 致命错误: 文件无法作为JSON加载。错误: {e}")
        return

    # --- 2. 逐条样本进行检查 ---
    for i, sample in enumerate(tqdm(data, desc="🔍 检查样本中")):
        errors_in_sample = []

        # 2.1 基础格式校验
        required_keys = ["instruction", "input", "output", "system"]
        missing_keys = [key for key in required_keys if key not in sample]
        if missing_keys:
            errors_in_sample.append(f"缺少关键字段: {', '.join(missing_keys)}")
            report["error_counts"]["missing_keys"] += 1

        output = sample.get("output", "")

        # 2.2 CoT格式专项校验
        if "<think>" not in output or "</think>" not in output:
            errors_in_sample.append("CoT格式错误: 缺少<think>或</think>标签")
            report["error_counts"]["missing_think_tags"] += 1

        if "</think>\n\n" not in output:
            errors_in_sample.append("CoT格式错误: </think>后缺少'\\n\\n'分隔符")
            report["error_counts"]["missing_separator"] += 1

        try:
            parts = output.split("</think>\n\n")
            if len(parts) != 2:
                raise ValueError("分隔符数量不为1")

            think_part, json_part = parts

            if not is_valid_json_list(json_part):
                errors_in_sample.append("JSON答案错误: 分隔符后的内容不是一个合法的JSON列表")
                report["error_counts"]["invalid_json_answer"] += 1
            else:
                # 2.3 内容逻辑一致性校验 & 统计
                json_answer = json.loads(json_part)

                # 统计事件类型分布
                if not json_answer:
                    report["statistics"]["samples_with_no_events"] += 1
                    if "未发现" not in think_part:
                        errors_in_sample.append("逻辑不一致: 答案为空列表，但思维链未提及'未发现'")
                        report["error_counts"]["logic_empty_mismatch"] += 1
                else:
                    for event in json_answer:
                        event_type = event.get("event_type")
                        if event_type:
                            report["statistics"]["event_type_distribution"][event_type] += 1

                    num_events_in_json = len(json_answer)
                    # 简单地通过'事件:'关键词来估计思维链中的事件数
                    num_events_in_think = think_part.count("- 事件:")
                    if num_events_in_think != num_events_in_json and num_events_in_think > 0:
                        errors_in_sample.append(
                            f"逻辑不一致: 思维链中事件数({num_events_in_think})与JSON答案数({num_events_in_json})不匹配")
                        report["error_counts"]["logic_count_mismatch"] += 1

        except Exception as e:
            errors_in_sample.append(f"CoT格式错误: 无法按'</think>\\n\\n'分割或处理。错误: {e}")
            report["error_counts"]["split_error"] += 1

        # 统计Token长度
        report["statistics"]["output_token_lengths"].append(len(output))

        # 记录有错误的样本
        if errors_in_sample:
            report["error_samples"].append({"index": i, "errors": errors_in_sample})

    # --- 3. 生成并打印体检报告 ---
    print("\n" + "=" * 20 + " 数据集健康体检报告 " + "=" * 20)
    print(f"文件路径: {report['file_path']}")
    print(f"总样本数: {report['total_samples']}")

    total_errors = sum(report["error_counts"].values())
    if total_errors == 0:
        print("\n✅ 恭喜！未发现任何格式或逻辑错误。数据集状态良好！")
    else:
        print(f"\n❌ 发现 {total_errors} 个问题，涉及 {len(report['error_samples'])} 个样本。")
        print("错误类型统计:")
        for error_type, count in report["error_counts"].items():
            print(f"  - {error_type}: {count} 次")

        print("\n部分错误样本示例 (最多显示5条):")
        for err_sample in report["error_samples"][:5]:
            print(f"  - 样本索引 {err_sample['index']}: {err_sample['errors']}")

    # --- 打印统计分析 ---
    print("\n--- 统计分析 ---")
    print(f"无事件样本数: {report['statistics']['samples_with_no_events']}")

    print("\n事件类型分布 (按频次排序):")
    event_dist = report["statistics"]["event_type_distribution"]
    if not event_dist:
        print("  数据集中未包含任何事件。")
    else:
        for event_type, count in event_dist.most_common():
            print(f"  - {event_type}: {count} 次")

    print("\nOutput字段Token长度分布:")
    if report["statistics"]["output_token_lengths"]:
        lengths_series = pd.Series(report["statistics"]["output_token_lengths"])
        print(lengths_series.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())

    print("\n" + "=" * 58)

    # 可选：将报告保存为JSON文件
    report_save_path = os.path.splitext(file_path)[0] + "_inspection_report.json"
    # 清理报告以便保存
    del report["statistics"]["output_token_lengths"]
    with open(report_save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"\n一份详细的JSON格式体检报告已保存至: {report_save_path}")


# --- 命令行入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="一个全面的数据集健康检查工具，用于校验CoT格式微调数据的质量。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'file_path',
        type=str,
        help="需要进行健康检查的数据集 .json 文件路径。"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file_path):
        print(f"❌ 错误: 文件不存在 -> {args.file_path}")
    else:
        inspect_dataset(args.file_path)