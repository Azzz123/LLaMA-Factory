import json
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import fire

def analyze_dataset_lengths_auto(
    model_path: str,
    dataset_path: str,
):
    """
    Automatically analyzes and reports the token length distribution of a dataset
    by using the model's own chat_template from its tokenizer_config.json.

    This script is universal and works for any model with a defined chat_template.

    Args:
        model_path (str): The path to the Hugging Face model and tokenizer.
        dataset_path (str): The path to the JSON dataset file.
    """
    print(f"Loading tokenizer from: {model_path}")
    try:
        # 加载 tokenizer，它会自动读取并准备好 chat_template
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure the model path is correct and contains tokenizer files.")
        return

    # 检查 tokenizer 是否真的加载了 chat_template
    if tokenizer.chat_template is None:
        print("Error: The tokenizer does not have a `chat_template` defined in its config.")
        print("This script requires a model with a configured chat template.")
        return

    print(f"Loading dataset from: {dataset_path}")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset path is correct and it's a valid JSON list.")
        return

    lengths = []
    print("Tokenizing all samples using the model's native chat_template...")
    for sample in tqdm(data, total=len(data)):
        # 1. 将你的数据格式转换成 apply_chat_template 需要的对话列表格式
        messages = []
        if sample.get("system"):
            messages.append({"role": "system", "content": sample["system"]})

        # 将 instruction 和 input 合并为一条 user 消息
        user_content = sample.get("instruction", "")
        if sample.get("input"):
            user_content += "\n\n" + sample["input"]
        messages.append({"role": "user", "content": user_content})

        # 2. 调用官方方法来应用模板并进行分词
        # 这是最关键、最准确的一步！
        try:
            # add_generation_prompt=True 会在末尾加上 'Assistant:' 或类似提示，
            # 这对于计算训练时的输入长度至关重要。
            token_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True
            )
            lengths.append(len(token_ids))
        except Exception as e:
            print(f"\nWarning: Failed to apply chat template for one sample. Error: {e}")
            print(f"Problematic sample: {sample}")
            continue

    if not lengths:
        print("\nNo samples were processed successfully.")
        return

    lengths_series = pd.Series(lengths)
    print("\n--- Token Length Distribution (Auto-Detected Template) ---")
    print(lengths_series.describe(percentiles=[0.5, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995]))

    print("\n--- Decision Making Helper ---")
    max_len = lengths_series.max()
    print(f"The longest sample has {max_len} tokens.")
    print("Recommended `cutoff_len` values based on data coverage:")
    for p in [0.90, 0.95, 0.98, 0.99]:
        len_at_p = int(lengths_series.quantile(p))
        recommended_len = (len_at_p // 512 + 1) * 512 if len_at_p % 512 != 0 else len_at_p
        print(f"- To cover {p*100:.0f}% of your data, you need a length of at least {len_at_p} tokens. (Suggest: {recommended_len})")


if __name__ == "__main__":
    fire.Fire(analyze_dataset_lengths_auto)