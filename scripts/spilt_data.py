import json
import random
from pathlib import Path

def split_dataset(input_file, output_dir, train_ratio=0.9, seed=42):
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đọc toàn bộ raw data
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    random.shuffle(lines)

    # Chia train/val
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # Ghi ra file
    with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(output_dir / "val.jsonl", "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"✅ Train: {len(train_lines)}, Val: {len(val_lines)}")

if __name__ == "__main__":
    split_dataset(
        input_file="data/raw/conversations.jsonl",
        output_dir="data/processed",
        train_ratio=0.9
    )
