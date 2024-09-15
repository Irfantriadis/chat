import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def prepare_dataset_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text_data = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            for response in intent['responses']:
                # Paraphrase or add variations
                text_data.append(f"Human: {pattern}\nBot: {response}\n")
                text_data.append(f"Human: {pattern}\nBot: {response} Bisa saya bantu lagi?\n")
                text_data.append(f"Human: {pattern}\nBot: {response} Ada yang lain?\n")
    
    # Save augmented dataset
    with open('processed_dataset.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_data))
    
    # Tokenize the dataset
    return TextDataset(
        tokenizer=tokenizer,
        file_path='processed_dataset.txt',
        block_size=128
    )

# Inisialisasi model dan tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Persiapkan dataset
train_dataset = prepare_dataset_from_json("model/tegaltourism.json")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Argumen training
training_args = TrainingArguments(
    output_dir="./tegal_tourism_chatbot",
    overwrite_output_dir=True,
    num_train_epochs=10,  # Meningkatkan jumlah epoch
    per_device_train_batch_size=8,  # Meningkatkan batch size
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=2e-5,  # Mengatur learning rate yang lebih rendah
    warmup_steps=500,  # Penambahan warmup steps
    weight_decay=0.01,  # Menambahkan weight decay untuk regularisasi
    logging_dir='./logs',  # Direktori untuk menyimpan log
    logging_steps=200,
)

# Inisialisasi trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune model
trainer.train()

# Simpan model
trainer.save_model()
print("Model telah disimpan di ./tegal_tourism_chatbot")
