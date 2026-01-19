# 🧠 Mental Health Risk Detection System

> 基于 MentalBERT 微调的心理健康风险智能检测系统

## 项目简介 (Introduction)

本项目旨在利用自然语言处理（NLP）技术，从非结构化文本中自动识别潜在的心理健康风险（如抑郁倾向）。核心模型基于 **MentalBERT** 进行微调，针对医疗健康领域的数据特点进行了深度优化。

系统具备从数据处理、模型训练、效果评估到服务部署的全流程能力，旨在为早期心理干预提供技术辅助。

## 核心特性 (Key Features)

* **领域专用模型**: 采用在心理健康语料上预训练的 `MentalBERT` 作为基座，语义理解更精准。
* **单卡训练优化**: 实现了 **梯度累积 (Gradient Accumulation)** 和 **混合精度训练 (FP16)**，在有限显存（<8G）下实现了大 Batch Size 的训练效果。
* **训练策略增强**: 引入 **早停机制 (Early Stopping)** 防止过拟合，自动保存最佳权重。
* **高召回率**: 针对医疗场景优化，实现了 **87.3% 的召回率 (Recall)**，最大程度降低漏报风险。
* **多端部署**:
* **Web UI**: 基于 Gradio 的交互式演示界面。
* **REST API**: 基于 FastAPI 的高性能推理接口。



## 性能表现 (Performance)

模型在测试集上的表现如下（基于 `mentalbert_finetuned_final`）：

| 指标 (Metric) | 得分 (Score) | 说明 |
| --- | --- | --- |
| **Recall (召回率)** | **87.3%** | 关键指标：成功识别出绝大多数高风险样本 |
| **F1-Score** | **83.6%** | 精确率与召回率的平衡 |
| **Accuracy (准确率)** | 82.4% | 整体分类准确度 |

*注：模型采用“宁可误报，不可漏报”的策略，误报（False Positive）主要集中在模糊语义样本，符合筛查工具的设计原则。*

## 🛠️ 安装指南 (Installation)

1. **克隆仓库**
```bash
git clone https://github.com/poppop-dot/mental-health-detection.git
cd mental-health-detection

```


2. **创建虚拟环境 (推荐)**
```bash
conda create -n inspeech python=3.10
conda activate inspeech

```


3. **安装依赖**
```bash
pip install torch torchvision torchaudio
pip install transformers datasets scikit-learn pandas matplotlib seaborn
pip install gradio fastapi uvicorn

```



## 快速开始 (Quick Start)

### 1. 数据准备

请确保 `data/` 目录下包含 `train-*.parquet` 和 `test-*.parquet` 数据文件。
运行数据探查脚本，检查数据分布：

```bash
python inspect_data.py

```

### 2. 模型训练

启动微调训练（包含自动早停和梯度累积）：

```bash
python train_final.py

```

*训练完成后，最佳模型将保存在 `./mentalbert_finetuned_final/final_model`。*

### 3. 模型评估

生成混淆矩阵图表 `confusion_matrix.png` 和详细评估报告：

```bash
python plot_confusion_matrix.py

```

### 4. 启动 Web 演示 (Gradio)

在本地启动可视化的网页界面：

```bash
python app.py

```

*访问地址: http://localhost:7860*

### 5. 启动 API 服务 (FastAPI)

启动生产级 API 服务：

```bash
python serve_model.py

```

*API 文档地址: http://localhost:8000/docs*

## 项目结构 (Structure)

```text
mental-health-detection/
├── data/                       # 数据集目录 (Parquet格式)
├── mentalbert/                 # 原始预训练模型权重
├── mentalbert_finetuned_final/ # 训练输出目录 (保存最佳模型)
│   └── final_model/
├── train_final.py              # [核心] 训练脚本 (含显存优化)
├── inspect_data.py             # 数据探查与分布统计工具
├── inference.py                # 单条文本推理测试脚本
├── plot_confusion_matrix.py    # 批量评估与混淆矩阵绘制
├── app.py                      # Gradio Web 演示应用
├── serve_model.py              # FastAPI 后端服务接口
├── requirements.txt            # 项目依赖列表
└── README.md                   # 项目说明文档

```

## 贡献 (Contributing)

欢迎提交 Issue 或 Pull Request 来改进本项目！

## 版权说明 (License)

本项目采用 MIT 开源协议。
模型仅供学术研究和技术演示使用，不构成专业医疗诊断建议。
