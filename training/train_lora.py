"""
train_lora.py — QLoRA fine-tuning of Qwen2.5-3B-Instruct on the resume-parsing dataset.

This script is written to actually run (e.g. on a Colab T4/A100, or any single CUDA GPU
with >=8GB VRAM). It was authored and reviewed here under Option B (no GPU available in
this environment to execute it), so treat the exact loss numbers you get as something to
verify yourself — but the config, LoRA target modules, and prompt formatting below reflect
real, standard practice for Qwen2.5 QLoRA fine-tunes.

Usage:
    pip install -r requirements.txt
    python train_lora.py --config config.yaml
"""
import argparse
import json
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def format_example(example):
    """
    Renders one {instruction, input, output} record into the Qwen chat template.
    The model learns to reproduce `output` (as a JSON string) as the assistant turn.
    """
    system = "You are a resume parsing engine. Always respond with valid JSON only."
    user = f"{example['instruction']}\n\nResume text:\n{example['input']}"
    assistant = json.dumps(example["output"], ensure_ascii=False)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return {"messages": messages}


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]

    # 4-bit quantized base model load (QLoRA) — keeps VRAM usage low enough for a single
    # consumer/free-tier GPU while training full-precision LoRA adapters on top.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        # Standard Qwen2.5 attention + MLP projection targets — covering both attention
        # and MLP layers gives LoRA enough capacity to shift output-formatting behavior
        # without the cost of full fine-tuning.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    raw = load_dataset("json", data_files=cfg["dataset_path"], split="train")
    dataset = raw.map(format_example, remove_columns=raw.column_names)

    # Train/val split — small dataset, so keep val meaningful (20%) rather than 5%
    split = dataset.train_test_split(test_size=cfg["val_split"], seed=42)

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=cfg.get("report_to", "none"),  # set to "wandb" to enable experiment tracking
        max_seq_length=cfg["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"LoRA adapter saved to {cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
