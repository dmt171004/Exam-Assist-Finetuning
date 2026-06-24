import os
import torch

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)
from peft import PeftModel

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


BASE_MODEL = ROOT / "Qwen3-VL-4B-It"
LORA_MODEL = ROOT / "outputs" / "qwen3_examassist_4b_domain_lora/checkpoint-118"
MERGED_MODEL = ROOT / "outputs" / "qwen3_examassist_merged_4b_domain"

print(BASE_MODEL)
print(LORA_MODEL)
print(MERGED_MODEL)

print("Loading base model...")

base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",          # merge không cần GPU
    trust_remote_code=True,
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
)

print("Merging LoRA weights...")

merged_model = model.merge_and_unload()

print("Saving merged model...")

os.makedirs(MERGED_MODEL, exist_ok=True)

merged_model.save_pretrained(
    MERGED_MODEL,
    safe_serialization=True,
)

print("Saving processor/tokenizer...")

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

processor.save_pretrained(MERGED_MODEL)

print(f"\nDone!")
print(f"Merged model saved to: {MERGED_MODEL}")