import json
import os

import fire
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# <<< MODIFICATION START >>>
# 导入 ptflops 库
try:
    from ptflops import get_model_complexity_info

    PTFLOPS_AVAILABLE = True
except ImportError:
    print("Warning: ptflops is not installed. Please run: pip install ptflops")
    PTFLOPS_AVAILABLE = False


# <<< MODIFICATION END >>>


def analyze_cost_and_length(
        model_path: str,
        dataset_path: str,
        quantization_bit: int = None,  # 支持QLoRA模型加载
):
    """Analyzes token length distribution and estimates representative FLOPs for a given dataset.
    Saves the analysis results to a JSON file in a 'cost_analysis' subfolder.

    Args:
        model_path (str): Path to the Hugging Face model and tokenizer.
        dataset_path (str): Path to the JSON dataset file.
        quantization_bit (int, optional): The bit number for quantization (e.g., 4 or 8 for QLoRA).
    """
    # --- 1. Tokenizer and Dataset Loading ---
    print(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    if tokenizer.chat_template is None:
        print("Error: The tokenizer does not have a `chat_template` defined.")
        return

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

        try:
            token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            lengths.append(len(token_ids))
        except Exception as e:
            print(f"\nWarning: Failed to process one sample. Error: {e}")
            continue

    if not lengths:
        print("\nNo samples were processed.")
        return

    lengths_series = pd.Series(lengths)
    length_stats = lengths_series.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    length_stats = {k: (int(v) if isinstance(v, (int, float)) else v) for k, v in length_stats.items()}

    # --- 打印长度分布和推荐值 ---
    print("\n--- Token Length Distribution ---")
    # 使用 to_string() 获得更美观的 pandas 输出
    print(lengths_series.describe(percentiles=[0.5, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995]).to_string())

    print("\n--- Decision Making Helper ---")
    max_len = lengths_series.max()
    print(f"The longest sample has {max_len} tokens.")
    print("Recommended `cutoff_len` values based on data coverage:")
    for p in [0.90, 0.95, 0.98, 0.99]:
        len_at_p = int(lengths_series.quantile(p))
        # 向上取整到最近的64或128倍数，这通常对性能更好
        recommended_len = (len_at_p // 64 + 1) * 64
        print(
            f"- To cover {p * 100:.0f}% of your data, you need a length of at least {len_at_p} tokens. (Suggest: {recommended_len})")

    # --- 3. FLOPs Estimation (using ptflops) ---
    flops_stats = {"status": "Not calculated", "reason": ""}
    if not PTFLOPS_AVAILABLE:
        flops_stats["reason"] = "ptflops library not installed."
    else:
        try:
            print("\nEstimating FLOPs using ptflops...")
            print(f"Loading model from: {model_path}")
            model_kwargs = {"trust_remote_code": True}
            if quantization_bit is not None:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=quantization_bit == 4,
                    load_in_8bit=quantization_bit == 8,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

            mean_length = int(length_stats['mean'])
            print(f"Analyzing model complexity with input shape: (1, {mean_length})")

            # <<< MODIFICATION START >>>
            # 使用 ptflops 计算复杂度
            # 注意：ptflops 需要一个函数来构造输入，这里我们用 lambda
            # 同时，QLoRA量化后的模型可能不支持直接分析，我们先尝试
            with torch.cuda.device(0):  # 确保在GPU上进行
                macs, params = get_model_complexity_info(
                    model,
                    (1, mean_length),
                    input_constructor=lambda shape: {'input_ids': torch.randint(0, tokenizer.vocab_size, shape).cuda()},
                    as_strings=False,
                    print_per_layer_stat=False,  # 关闭逐层打印，保持输出简洁
                    verbose=False
                )

            # FLOPs 约等于 2 * MACs (一次乘加运算)
            total_flops = 2 * macs

            flops_stats = {
                "status": "Success (estimated with ptflops)",
                "total_flops": total_flops,
                "gflops": round(total_flops / 1e9, 2),
                "tflops": round(total_flops / 1e12, 2),
                "gmacs": round(macs / 1e9, 2),
                "params_in_millions": round(params / 1e6, 2),
                "based_on_mean_length": mean_length
            }
            # <<< MODIFICATION END >>>

            print("\n--- FLOPs Estimation Results ---")
            print(json.dumps(flops_stats, indent=4))

        except Exception as e:
            print(f"\nError during FLOPs estimation with ptflops: {e}")
            flops_stats["status"] = "Failed"
            flops_stats["reason"] = str(e)

    # --- 4. Save Combined Results (with new path logic) ---
    final_report = {
        "dataset_path": dataset_path,
        "model_path": model_path,
        "length_statistics": length_stats,
        "flops_statistics": flops_stats
    }

    # <<< MODIFICATION START >>>
    # Get the directory of the input dataset
    input_dir = os.path.dirname(dataset_path)
    # Create the 'cost_analysis' subdirectory
    report_dir = os.path.join(input_dir, "cost_analysis")
    os.makedirs(report_dir, exist_ok=True)

    # Use the original dataset's name for the report file
    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    report_save_path = os.path.join(report_dir, f"{base_name}.json")
    # <<< MODIFICATION END >>>

    with open(report_save_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)

    print(f"\nAnalysis report saved to: {report_save_path}")


if __name__ == "__main__":
    fire.Fire(analyze_cost_and_length)
