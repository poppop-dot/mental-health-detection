import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, recall_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# =================配置区域=================
MODEL_PATH = "./mentalbert"           # 本地模型路径
DATA_PATH = "./data"                  # 本地数据路径
OUTPUT_DIR = "./mentalbert_finetuned_final" # 输出路径

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =================1. 加载数据=================
print(f"Loading data from {DATA_PATH}...")
data_files = {
    "train": os.path.join(DATA_PATH, "train-00000-of-00001.parquet"),
    "test": os.path.join(DATA_PATH, "test-00000-of-00001.parquet")
}
dataset = load_dataset("parquet", data_files=data_files)
# =================2. 数据预处理 =================
print(f"Loading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# 强制指定实际列名
text_col = "text"   
label_col = "label" 
print(f"使用文本列: '{text_col}', 标签列: '{label_col}'")
def preprocess_function(examples):
    # MentalBERT处理: 截断长度512
    return tokenizer(examples[text_col], truncation=True, padding=False, max_length=512)
# 进行分词
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 移除不需要的辅助列 (LIWC特征等)，只保留模型训练需要的列
# 这一步非常重要，因为您的原始数据有100多列，不清理会报错或导致训练极慢
columns_to_keep = ['input_ids', 'attention_mask', 'label'] 
# 如果有 token_type_ids 也保留
if 'token_type_ids' in tokenized_datasets['train'].features:
    columns_to_keep.append('token_type_ids')

# 获取所有列名
all_cols = tokenized_datasets['train'].column_names
# 计算需要删除的列
columns_to_remove = [col for col in all_cols if col not in columns_to_keep]

print(f"正在移除 {len(columns_to_remove)} 个多余的特征列 (LIWC等)...")
tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

# 将 label 重命名为 labels (HuggingFace 标准)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 设置格式为 torch，确保数据是 Tensor 类型
tokenized_datasets.set_format("torch")

# =================3. 初始化模型=================
print("Initializing model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)

# =================4. 评估指标=================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "recall": recall_score(labels, predictions, average='binary'),
        "f1": f1_score(labels, predictions, average='binary')
    }

# =================5. 训练参数=================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,   # 保持4以防爆显存
    gradient_accumulation_steps=4,   # [核心] 累积4步，等效 Batch Size = 16，训练更稳！
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # 自动检测 GPU 加速
    dataloader_num_workers=0         # 避免多进程死锁
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# =================6. 开始训练=================
if __name__ == "__main__":
    print(">>> 开始训练...")
    trainer.train()
    
    # 保存最终模型
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f">>> 训练完成！模型已保存至: {final_path}")