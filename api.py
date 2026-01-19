import os
import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================= 配置区域 =================
# 指定显卡，保持单卡运行
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = "./mentalbert_finetuned_final/final_model"

# 定义输入数据格式 (Request Body)
class SentimentRequest(BaseModel):
    text: str

# 定义输出数据格式 (Response Body)
class SentimentResponse(BaseModel):
    label: str
    risk_score: float
    probabilities: dict

# ================= 全局变量 =================
# 用于存储加载后的模型，避免反复加载
ml_models = {}

# ================= 1. 生命周期管理 (Lifespan) =================
# 这是 FastAPI 推荐的现代写法：在服务启动前加载模型，服务关闭后清理
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 [Startup] 正在加载模型: {MODEL_PATH} ...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval() # 切换到评估模式
        
        # 将模型和配置存入全局字典
        ml_models["tokenizer"] = tokenizer
        ml_models["model"] = model
        ml_models["device"] = device
        ml_models["labels"] = {0: "Healthy", 1: "Risk"}
        
        print(f"[Startup] 模型加载完成！运行设备: {device}")
        yield
        
    except Exception as e:
        print(f"[Error] 模型加载失败: {e}")
        # 这里可以让程序退出，或者记录日志
    finally:
        print("[Shutdown] 服务正在关闭，清理资源...")
        ml_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ================= 2. 初始化 App =================
app = FastAPI(
    title="MentalBERT API Service",
    description="基于微调 MentalBERT 的心理健康风险检测 API",
    version="1.0.0",
    lifespan=lifespan
)

# ================= 3. 核心预测接口 =================
@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    # 1. 获取模型资源
    if "model" not in ml_models:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    tokenizer = ml_models["tokenizer"]
    model = ml_models["model"]
    device = ml_models["device"]
    id2label = ml_models["labels"]

    # 2. 文本预处理
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    inputs = tokenizer(
        request.text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    ).to(device)

    # 3. 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
        # 转为概率
        probs = F.softmax(outputs.logits, dim=-1)[0]
        
        # 获取最大概率的标签
        pred_idx = torch.argmax(probs).item()
        pred_label = id2label[pred_idx]
        
        # 提取风险概率 (Label 1 的概率)
        risk_score = float(probs[1])

    # 4. 构造返回结果
    return SentimentResponse(
        label=pred_label,
        risk_score=round(risk_score, 4),
        probabilities={
            "Healthy": float(probs[0]),
            "Risk": float(probs[1])
        }
    )

# ================= 4. 健康检查接口 =================
@app.get("/health")
async def health_check():
    return {"status": "ok", "device": ml_models.get("device", "unknown")}

# ================= 启动入口 =================
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" 允许局域网访问
    # port=8000 是标准端口
    uvicorn.run(app, host="0.0.0.0", port=8060)