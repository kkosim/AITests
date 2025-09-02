import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =========================
# Helper: target modules
# =========================
def guess_lora_target_modules(model) -> List[str]:
    """
    Пытаемся угадать имена линейных слоёв под LoRA для популярных архитектур (LLaMA/Mistral и др).
    Если не уверены — оставьте по умолчанию список ниже.
    """
    common = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "W_pack"  # встречается у некоторых сборок
    ]
    model_names = set([n for n, _ in model.named_modules()])
    # Фильтруем по реально существующим
    return [m for m in common if any(m in name for name in model_names)] or common

# =========================
# Prompt building
# =========================
def build_prompt(question: str) -> str:
    """
    Минимальный и универсальный промпт (без привязки к конкретной chat-разметке).
    Если используете chat-модели с шаблонами — можно включить --use_chat_template.
    """
    return f"Вопрос: {question}\nОтвет:"

def build_answer(answer: str, law_id: Optional[str], add_source_tag: bool) -> str:
    ans = answer.strip()
    if add_source_tag and law_id:
        ans += f" [[SOURCE: {law_id}]]"
    return ans

# =========================
# Data Collator (mask prompt labels)
# =========================
@dataclass
class SupervisedDataCollator:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        attention_masks = [torch.ones_like(ids) for ids in input_ids]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
        }

# =========================
# Preprocess function
# =========================
def preprocess_examples(
    examples,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    use_chat_template: bool,
    add_source_tag: bool,
) -> Dict[str, List[List[int]]]:
    input_ids_batch = []
    labels_batch = []

    questions = examples["question"]
    answers = examples["answer"]
    law_ids = examples.get("law_id", [None] * len(questions))

    for q, a, lid in zip(questions, answers, law_ids):
        # 1) Сборка промпта и ответа
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": ""},  # ответ будет подставлен ниже
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # чтобы шаблон заканчивался "assistant:"
            )
        else:
            prompt_text = build_prompt(q)

        answer_text = build_answer(a, lid, add_source_tag=add_source_tag)

        # 2) Токенизация
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True, truncation=True, max_length=max_seq_len)["input_ids"]
        full_text = prompt_text + " " + answer_text
        full_ids = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=max_seq_len)["input_ids"]

        # 3) Маска меток: токены промпта не обучаются (label = -100), обучаем только на ответе
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # Может случиться, что ответ полностью отрезан при трънкации — пропустим такие примеры
        if all(l == -100 for l in labels):
            continue

        input_ids_batch.append(full_ids)
        labels_batch.append(labels)

    return {"input_ids": input_ids_batch, "labels": labels_batch}

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning (HF + PEFT + bitsandbytes)")

    # Модель и данные
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Базовая СХ модель (например, mistralai/Mistral-7B-Instruct-v0.2).")
    parser.add_argument("--data_path", type=str, default="qa_dataset.jsonl", help="JSONL с полями question, answer, law_id.")
    parser.add_argument("--output_dir", type=str, default="./model_tuned")
    parser.add_argument("--val_size", type=float, default=0.02, help="Доля данных под валидацию.")
    parser.add_argument("--max_seq_len", type=int, default=2048)

    # Обучение
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)

    # QLoRA / PEFT
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bias", type=str, default="none")
    parser.add_argument("--use_chat_template", action="store_true", help="Использовать tokenizer.apply_chat_template (для chat-моделей).")
    parser.add_argument("--add_source_tag", action="store_true", help="Добавлять [[SOURCE: law_id]] к таргету во время обучения.")

    # Mixed precision / 4-bit
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # Сохранение/мерджинг
    parser.add_argument("--merge_lora", action="store_true", help="Смёржить LoRA в базовую модель на выходе (потребует больше памяти).")

    args = parser.parse_args()
    set_seed(args.seed)

    # ===== Dataset =====
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    if args.val_size and 0.0 < args.val_size < 1.0:
        split = dataset.train_test_split(test_size=args.val_size, seed=args.seed)
        train_data = split["train"]
        eval_data = split["test"]
    else:
        train_data = dataset
        eval_data = None

    # ===== Tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        # для Causal LM удобно паддить до eos
        tokenizer.pad_token = tokenizer.eos_token

    # ===== 4bit quantization =====
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    # ===== Base model (4-bit) =====
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # ===== PEFT LoRA =====
    target_modules = guess_lora_target_modules(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ===== Preprocess =====
    def _map_fn(batch):
        return preprocess_examples(
            batch,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            use_chat_template=args.use_chat_template,
            add_source_tag=args.add_source_tag,
        )

    train_dataset = train_data.map(
        _map_fn,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Tokenizing train",
    )

    eval_dataset = None
    if eval_data is not None:
        eval_dataset = eval_data.map(
            _map_fn,
            batched=True,
            remove_columns=eval_data.column_names,
            desc="Tokenizing eval",
        )

    # ===== Trainer =====
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        bf16=args.bf16,
        fp16=args.fp16,
        optim="paged_adamw_8bit",  # оптимизировано под 4-bit
        report_to="none",
        save_total_limit=2,
    )

    collator = SupervisedDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)   # сохранит адаптеры LoRA
    tokenizer.save_pretrained(args.output_dir)

    # ===== Optional: merge LoRA into base weights =====
    if args.merge_lora:
        # Перезагружаем базовую модель в FP16/BF16 (без 4bit), накатываем LoRA и сливаем
        print("Merging LoRA adapters into the base model weights...")
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            device_map="auto",
        )
        base = get_peft_model(base, lora_cfg)
        # Загрузим обученные адаптеры
        base.load_adapter(args.output_dir, adapter_name="default")
        merged = base.merge_and_unload()
        merged.save_pretrained(os.path.join(args.output_dir, "merged"))
        print("Merged model saved to:", os.path.join(args.output_dir, "merged"))

    print("Done.")
    

if __name__ == "__main__":
    main()
