import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# 1. 指向您刚刚训练完成的模型路径
MODEL_PATH = "./mentalbert_finetuned_final/final_model"

# 2. 定义标签 (根据您的数据集，0通常是正常/Negative，1是风险/Positive)
# 如果您发现预测结果反了，只需要把这两个标签对调即可
ID2LABEL = {
    0: "心理健康 (Normal/Control)", 
    1: "存在风险 (Risk/Depression)"
}

def predict(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n加载模型中... (Device: {device})")
    try:
        # 加载微调后的模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 预处理
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    ).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1) # 转为概率
        
        # 获取结果
        pred_idx = torch.argmax(probs, dim=-1).item()
        score = probs[0][pred_idx].item()

    # 打印结果
    print("=" * 40)
    print(f"输入文本: \"{text}\"")
    print(f"诊断结果: {ID2LABEL[pred_idx]}")
    print(f"置信度: {score:.2%}")
    print(f"   (健康概率: {probs[0][0]:.2%}, 风险概率: {probs[0][1]:.2%})")
    print("=" * 40)

if __name__ == "__main__":
    # === 在这里修改您想测试的句子 ===
    test_cases = [
        "I had a wonderful dinner with my family today.", 
        "I feel empty inside and I don't see a future for myself.",
        "The anxiety is getting worse, I can't sleep at night."
    ]
    
    for sentence in test_cases:
        predict(sentence)