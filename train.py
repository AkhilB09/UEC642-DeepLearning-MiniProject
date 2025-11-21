import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SwinForImageClassification
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

# --- CONFIGURATION ---
# Using 'beans' as a proxy for PlantVillage (simpler download)
DATASET_ID = "beans" 
MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"
OUTPUT_DIR = "results"
BATCH_SIZE = 32
EPOCHS = 5
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 1. Setup Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load & Prepare Data
    print(f">>> Loading Dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    def transforms(examples):
        examples["pixel_values"] = [
            processor(image.convert("RGB"), return_tensors="pt").pixel_values[0] 
            for image in examples["image"]
        ]
        return examples

    dataset.set_transform(transforms)
    train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

    # 3. Model Setup (The Recent Tech: LoRA)
    print(">>> Initializing Swin Transformer with LoRA...")
    model = SwinForImageClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3, 
        ignore_mismatched_sizes=True
    )

    # LoRA Configuration (Rank=16 for efficiency)
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["query", "value", "dense"], 
        modules_to_save=["classifier"],
        lora_dropout=0.1
    )

    lora_model = get_peft_model(model, peft_config)
    lora_model.to(DEVICE)
    lora_model.print_trainable_parameters()

    # 4. Training Loop
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=LR)
    loss_history = []

    print(">>> Starting Training...")
    lora_model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            pixel_values = torch.stack([item['pixel_values'] for item in batch]).to(DEVICE)
            labels = torch.tensor([item['labels'] for item in batch]).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = lora_model(pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

    # 5. Save Model & History
    print(f">>> Saving model to {OUTPUT_DIR}...")
    lora_model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapters"))
    
    with open(os.path.join(OUTPUT_DIR, "loss_history.json"), "w") as f:
        json.dump(loss_history, f)
        
    print("✅ Training Complete.")

if __name__ == "__main__":
    main()