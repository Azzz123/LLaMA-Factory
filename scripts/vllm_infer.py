# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
from typing import Optional

import fire
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

import os  # <<< MODIFICATION START >>>
import time  # <<< MODIFICATION START >>>


def vllm_infer(
        model_name_or_path: str,
        adapter_name_or_path: str = None,
        dataset: str = "alpaca_en_demo",
        dataset_dir: str = "data",
        template: str = "default",
        cutoff_len: int = 2048,
        max_samples: Optional[int] = 20000,
        vllm_config: str = "{}",
        save_name: str = "generated_predictions.jsonl",
        temperature: float = 0.5,
        top_p: float = 0.7,
        top_k: int = 50,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.0,
        skip_special_tokens: bool = True,
        default_system: Optional[str] = None,
        enable_thinking: bool = False,
        seed: Optional[int] = None,
        pipeline_parallel_size: int = 1,
        image_max_pixels: int = 768 * 768,
        image_min_pixels: int = 32 * 32,
        video_fps: float = 2.0,
        video_maxlen: int = 128,
        batch_size: int = 1024,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm = LLM(**engine_args)

    # load datasets
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # <<< MODIFICATION START >>>
    # --- Performance Metrics Initialization ---
    start_time = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    # <<< MODIFICATION END >>>

    # Store all results in these lists
    all_prompts, all_preds, all_labels = [], [], []

    # Add batch process to avoid the issue of too many files opened
    for i in tqdm(range(0, len(train_dataset), batch_size), desc="Processing batched inference"):
        vllm_inputs, prompts, labels = [], [], []
        batch = train_dataset[i: min(i + batch_size, len(train_dataset))]

        for j in range(len(batch["input_ids"])):
            if batch["images"][j] is not None:
                image = batch["images"][j]
                multi_modal_data = {
                    "image": template_obj.mm_plugin._regularize_images(
                        image, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                    )["images"]
                }
            elif batch["videos"][j] is not None:
                video = batch["videos"][j]
                multi_modal_data = {
                    "video": template_obj.mm_plugin._regularize_videos(
                        video,
                        image_max_pixels=image_max_pixels,
                        image_min_pixels=image_min_pixels,
                        video_fps=video_fps,
                        video_maxlen=video_maxlen,
                    )["videos"]
                }
            elif batch["audios"][j] is not None:
                audio = batch["audios"][j]
                audio_data = template_obj.mm_plugin._regularize_audios(
                    audio,
                    sampling_rate=16000,
                )
                multi_modal_data = {"audio": zip(audio_data["audios"], audio_data["sampling_rates"])}
            else:
                multi_modal_data = None

            vllm_inputs.append({"prompt_token_ids": batch["input_ids"][j], "multi_modal_data": multi_modal_data})
            prompts.append(tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens))
            labels.append(
                tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                    skip_special_tokens=skip_special_tokens,
                )
            )

            # <<< MODIFICATION START >>>
            total_input_tokens += len(batch["input_ids"][j])
            # <<< MODIFICATION END >>>

        results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
        preds = [result.outputs[0].text for result in results]

        # <<< MODIFICATION START >>>
        for pred_text in preds:
            total_output_tokens += len(tokenizer.encode(pred_text))
        # <<< MODIFICATION END >>>

        # Accumulate results
        all_prompts.extend(prompts)
        all_preds.extend(preds)
        all_labels.extend(labels)
        gc.collect()

    # <<< MODIFICATION START >>>
    # --- Performance Metrics Calculation & Saving ---
    end_time = time.time()
    total_time_seconds = end_time - start_time
    num_samples = len(all_prompts)

    samples_per_second = num_samples / total_time_seconds if total_time_seconds > 0 else 0
    output_tokens_per_second = total_output_tokens / total_time_seconds if total_time_seconds > 0 else 0
    total_tokens_per_second = (
                                      total_input_tokens + total_output_tokens) / total_time_seconds if total_time_seconds > 0 else 0

    performance_metrics = {
        "num_samples": num_samples,
        "total_time_seconds": round(total_time_seconds, 2),
        "avg_input_tokens": round(total_input_tokens / num_samples, 2) if num_samples > 0 else 0,
        "avg_output_tokens": round(total_output_tokens / num_samples, 2) if num_samples > 0 else 0,
        "samples_per_second": round(samples_per_second, 2),
        "output_tokens_per_second": round(output_tokens_per_second, 2),
        "total_tokens_per_second": round(total_tokens_per_second, 2),
    }

    # Determine the directory of the dataset to save the metrics file
    # This assumes `dataset` is a file path in your case.
    # If `dataset` is a name, you might need to adjust this logic based on `dataset_dir`.
    try:
        dataset_file_path = data_args.dataset_path
        if dataset_file_path:
            output_dir = os.path.dirname(dataset_file_path)
            metrics_save_path = os.path.join(output_dir, f"inference_metrics_{os.path.basename(save_name)}.json")
        else:  # Fallback
            metrics_save_path = f"inference_metrics_{os.path.basename(save_name)}.json"
    except Exception:
        metrics_save_path = f"inference_metrics_{os.path.basename(save_name)}.json"

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(performance_metrics, f, indent=4, ensure_ascii=False)

    print("\n--- Inference Performance Metrics ---")
    print(json.dumps(performance_metrics, indent=4))
    print(f"Performance metrics saved to: {metrics_save_path}")
    print("-" * 35)
    # <<< MODIFICATION END >>>

    # Write all results at once outside the loop
    with open(save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(all_prompts, all_preds, all_labels):
            f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(all_prompts)} total generated results have been saved at {save_name}.")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(vllm_infer)
