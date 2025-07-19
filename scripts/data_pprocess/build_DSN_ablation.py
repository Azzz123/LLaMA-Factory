import os
import json
import argparse
from copy import deepcopy

def ablate_item(item, mode):
    data = deepcopy(item)
    # Load structured input
    input_data = json.loads(data["input"])

    # Adjust instruction based on mode
    instr = data.get("instruction", "")
    if mode == "wo_argument":
        # Remove arguments, event_type, trigger
        for pair in input_data.get("candidate_pairs", []):
            for ev in ["event_1", "event_2"]:
                for key in ["event_type", "trigger", "arguments"]:
                    pair[ev].pop(key, None)
        # Update instruction: remove mention of arguments
        instr = instr.replace("请仔细分析每个候选对中'event_1'和'event_2'的结构化论元信息（arguments），并", "请分析候选对中'event_1'和'event_2'，并")

    elif mode == "wo_context":
        # Remove context snippet
        for pair in input_data.get("candidate_pairs", []):
            pair.pop("context_snippet", None)
        # Update instruction: remove mention of context_snippet
        instr = instr.replace("其'context_snippet'提供的文本线索以及","")

    elif mode == "wo_global":
        # Remove global text
        input_data.pop("text", None)
        # Update instruction: remove mention of 全局上下文
        instr = instr.replace("以及'text'中的全局上下文，","，")

    # Assign modified instruction and input
    data["instruction"] = instr
    data["input"] = json.dumps(input_data, ensure_ascii=False)
    return data

def process_split(split_path, mode):
    with open(split_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return [ablate_item(item, mode) for item in dataset]

def save_dataset(dataset, output_dir, split_name):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, split_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def main(input_dir, output_base_dir):
    splits = ['train.json', 'test.json']
    ablation_modes = {
        'wo_argument': 'DSN-DEMAMD-wo-argument',
        'wo_context':  'DSN-DEMAMD-wo-context',
        'wo_global':   'DSN-DEMAMD-wo-global'
    }

    for mode, out_dir_name in ablation_modes.items():
        dest_dir = os.path.join(output_base_dir, out_dir_name)
        for split in splits:
            split_path = os.path.join(input_dir, split)
            new_dataset = process_split(split_path, mode)
            split_label = 'train' if 'train' in split else 'test'
            out_name = f"{split_label}_{mode}.json"
            save_dataset(new_dataset, dest_dir, out_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate ablation datasets based on defined modes and adjust instructions")
    parser.add_argument('--input_dir',  required=True, help='Path to original dataset folder (with train.json and test.json)')
    parser.add_argument('--output_dir', required=True, help='Base path to save ablation dataset folders')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
