import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import os

# =================配置=================
# 指向您刚刚训练好的最佳模型
MODEL_PATH = "./mentalbert_finetuned_final/final_model" 
DATA_PATH = "./data"
TEST_FILE = os.path.join(DATA_PATH, "test-00000-of-00001.parquet")

# =================1. 加载模型与数据=================
print(f"正在加载最佳模型: {MODEL_PATH} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

print("正在加载测试数据...")
dataset = load_dataset("parquet", data_files={'test': TEST_FILE})['test']

# =================2. 批量推理=================
print("正在进行批量预测 (这可能需要几十秒)...")
true_labels = []
pred_labels = []

# 为了速度，我们不使用 Trainer，直接用简单的循环
batch_size = 16
for i in range(0, len(dataset), batch_size):
    batch = dataset[i : i + batch_size]
    texts = batch['text']
    labels = batch['label']
    
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
    
    true_labels.extend(labels)
    pred_labels.extend(preds)

# =================3. 绘制混淆矩阵=================
cm = confusion_matrix(true_labels, pred_labels)
# 计算百分比形式
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy (0)', 'Risk (1)'], 
            yticklabels=['Healthy (0)', 'Risk (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('MentalBERT Confusion Matrix')

# 保存图片
save_path = "confusion_matrix.png"
plt.savefig(save_path)
print(f"\n混淆矩阵图已保存为: {save_path}")

# =================4. 打印详细文本报告=================
tn, fp, fn, tp = cm.ravel()
print("\n" + "="*30)
print("详细诊断报告")
print("="*30)
print(f"正确识别健康 (TN): {tn} (特异度: {tn/(tn+fp):.2%})")
print(f"正确识别风险 (TP): {tp} (敏感度/Recall: {tp/(tp+fn):.2%})")
print(f"误报 (FP - 把健康说成有病): {fp}")
print(f"漏报 (FN - 把有病说成健康): {fn}  <-- 这个数字越小越好")
print("="*30)