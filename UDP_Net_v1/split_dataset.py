import random

random.seed(42)  # reproducible split

with open('dataset/allweather/allweather.txt', 'r') as f:
    lines = f.readlines()

random.shuffle(lines)

n = len(lines)
split_idx = int(0.8 * n)

train_lines = lines[:split_idx]
val_lines   = lines[split_idx:]

with open('dataset/allweather/train.txt', 'w') as f:
    f.writelines(train_lines)

with open('dataset/allweather/val.txt', 'w') as f:
    f.writelines(val_lines)

print(f"Total samples: {n}")
print(f"Train samples: {len(train_lines)}")
print(f"Val samples:   {len(val_lines)}")
