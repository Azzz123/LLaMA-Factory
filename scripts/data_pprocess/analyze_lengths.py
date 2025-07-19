import json
import os

import fire
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 导入 ptflops 库
try:
    from ptflops import get_model_complexity_info

    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False


def analyze_cost_and_length(
        model_path: str,
        dataset_path: str,
        quantization_bit: int = None,
):
    """
    Analyzes token length distribution and estimates FLOPs.
    Saves a comprehensive report, including decision-making helpers, to a JSON file.
    """
    # --- 1. Tokenizer and Dataset Loading ---
    print(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 检查 chat_template 是否存在
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        # 如果没有，尝试设置一个通用的模板
        print("Warning: Tokenizer does not have a `chat_template`. Applying a generic one.")
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}{% endif %}{% endfor %}"

    print(f"Loading dataset from: {dataset_path}")
    try:
        with open(dataset_path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- 2. Token Length Analysis ---
    lengths = []
    print("Analyzing token lengths...")
    for sample in tqdm(data, total=len(data)):
        messages = []
        if sample.get("system"):
            messages.append({"role": "system", "content": sample["system"]})
        user_content = sample.get("instruction", "")
        if sample.get("input"):
            user_content += "\n\n" + sample["input"]
        messages.append({"role": "user", "content": user_content})

        # 考虑到输出也是模型学习的一部分，在训练时需要计算其长度
        if sample.get("output"):
            messages.append({"role": "assistant", "content": sample["output"]})

        try:
            # 在训练时，我们不需要 'add_generation_prompt'
            token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
            lengths.append(len(token_ids))
        except Exception as e:
            print(f"\nWarning: Failed to process one sample. Error: {e}")
            continue

    if not lengths:
        print("\nNo samples were processed.")
        return

    lengths_series = pd.Series(lengths)
    # 基础统计数据
    length_stats = lengths_series.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    length_stats = {k: int(v) for k, v in length_stats.items()}

    # --- 打印长度分布和推荐值 ---
    print("\n--- Token Length Distribution ---")
    print(lengths_series.describe(percentiles=[0.5, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995]).to_string())

    # <<< MODIFICATION START >>>
    # --- 准备 Decision Making Helper 数据 ---
    print("\n--- Decision Making Helper ---")
    decision_helper_data = {
        "notes": "Recommended `cutoff_len` values to balance data coverage and computational cost.",
        "longest_sample_tokens": int(lengths_series.max()),
        "recommendations": []
    }

    recommendation_str_list = []
    for p in [0.90, 0.95, 0.98, 0.99, 0.995]:
        len_at_p = int(lengths_series.quantile(p))
        # 向上取整到最近的64倍数
        recommended_len = (len_at_p + 63) // 64 * 64

        rec_entry = {
            "coverage_percent": p * 100,
            "token_length_required": len_at_p,
            "suggested_cutoff_len": recommended_len
        }
        decision_helper_data["recommendations"].append(rec_entry)

        # 准备打印的字符串
        recommendation_str_list.append(
            f"- To cover {rec_entry['coverage_percent']:.1f}% of data, you need >= {len_at_p} tokens. (Suggest: {recommended_len})"
        )

    # 打印到控制台
    print(f"The longest sample has {decision_helper_data['longest_sample_tokens']} tokens.")
    print("Recommended `cutoff_len` values:")
    for s in recommendation_str_list:
        print(s)
    # <<< MODIFICATION END >>>

    # --- 3. FLOPs Estimation ---
    flops_stats = {"status": "Not calculated", "reason": ""}
    if not PTFLOPS_AVAILABLE:
        flops_stats["reason"] = "ptflops library not installed."
    else:
        try:
            print(f"\nLoading model for FLOPs estimation: {model_path}")
            model_kwargs = {"trust_remote_code": True}
            if quantization_bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=quantization_bit == 4,
                    load_in_8bit=quantization_bit == 8,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

            mean_length = length_stats['mean']
            print(f"Analyzing model complexity with input shape: (1, {mean_length})")

            with torch.cuda.device(0):
                macs, params = get_model_complexity_info(
                    model, (1, mean_length),
                    input_constructor=lambda shape: {'input_ids': torch.randint(0, tokenizer.vocab_size, shape).cuda()},
                    as_strings=False, print_per_layer_stat=False, verbose=False
                )
            total_flops = 2 * macs
            flops_stats = {
                "status": "Success (estimated with ptflops)",
                "gflops": round(total_flops / 1e9, 2),
                "params_in_millions": round(params / 1e6, 2),
                "based_on_mean_length": mean_length
            }
            print("\n--- FLOPs Estimation Results ---")
            print(json.dumps(flops_stats, indent=2))

        except Exception as e:
            print(f"\nError during FLOPs estimation: {e}")
            flops_stats["status"] = "Failed"
            flops_stats["reason"] = str(e)

    # --- 4. Save Combined Report ---
    final_report = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "total_samples_analyzed": len(lengths),
        "length_statistics": length_stats,
        "decision_helper": decision_helper_data,  # <<< MODIFICATION: 添加决策助手数据
        "flops_statistics": flops_stats
    }

    report_dir = os.path.join(os.path.dirname(dataset_path), "cost_analysis")
    os.makedirs(report_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    report_save_path = os.path.join(report_dir, f"{base_name}_analysis_report.json")

    with open(report_save_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)

    print(f"\n✅ Comprehensive analysis report saved to: {report_save_path}")


if __name__ == "__main__":
    fire.Fire(analyze_cost_and_length)