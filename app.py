import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# =================配置区域=================
# 指向您上一轮训练出的最佳模型路径
MODEL_PATH = "./mentalbert_finetuned_final/final_model"

# 定义标签映射 (UI显示用)
LABELS = {
    0: "✅ 心理健康 (Healthy)",
    1: "⚠️ 存在风险 (Risk)"
}

# =================1. 加载模型 (启动时运行一次)=================
print(f"正在加载模型: {MODEL_PATH} ...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败，请检查路径。错误: {e}")
    exit()

# =================2. 定义预测函数 (核心逻辑)=================
def predict(text):
    if not text:
        return None
    
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
        # 使用 Softmax 将 logits 转为概率
        probs = F.softmax(outputs.logits, dim=-1)[0]
    
    # Gradio 要求返回一个字典: {类别名: 概率值}
    return {
        LABELS[0]: float(probs[0]),
        LABELS[1]: float(probs[1])
    }

# =================3. 构建 Gradio 界面=================
# 自定义 CSS 美化界面 (可选)
custom_css = """
#component-0 {max-width: 800px; margin: auto;}
"""

with gr.Blocks(css=custom_css, title="MentalBERT 心理健康检测") as demo:
    gr.Markdown(
        """
        # 🧠 MentalBERT 心理健康风险检测系统
        
        基于 **MentalBERT** 微调的深度学习模型，用于识别文本中的**抑郁倾向**或**心理健康风险**。
        *(仅供研究演示，不构成医疗诊断建议)*
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                lines=5, 
                placeholder="请输入患者的主诉、日记或社交媒体文本...", 
                label="输入文本 (Input Text)"
            )
            submit_btn = gr.Button("开始分析 (Analyze)", variant="primary")
            
        with gr.Column():
            output_label = gr.Label(num_top_classes=2, label="分析结果 (Prediction)")
    
    # 添加一些示例，方便点击测试
    gr.Examples(
        examples=[
            ["I had a great time with my friends today, the food was delicious!"],
            ["I feel so empty and hopeless. I don't know if I can go on."],
            ["The anxiety is keeping me up all night, my chest hurts."],
            ["I'm looking for a job, it's a bit stressful but I'm hopeful."]
        ],
        inputs=input_text,
        outputs=output_label,
        fn=predict,
        cache_examples=False,
    )

    # 绑定按钮事件
    submit_btn.click(fn=predict, inputs=input_text, outputs=output_label)

# =================4. 启动服务=================
if __name__ == "__main__":
    print("正在启动 Web 服务...")
    # server_name="0.0.0.0" 允许局域网访问
    # share=True 会生成一个免费的公网链接 (类似 xxxx.gradio.live)
    demo.launch(server_name="0.0.0.0", share=True)