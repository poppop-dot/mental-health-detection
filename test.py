import os
from datasets import load_dataset
import pandas as pd

# =================配置=================
DATA_PATH = "./data"
TRAIN_FILE = os.path.join(DATA_PATH, "train-00000-of-00001.parquet")

# =================检查流程=================
def inspect():
    print(f"正在读取文件: {TRAIN_FILE} ...")
    
    # 1. 加载数据
    try:
        # 使用 datasets 库加载
        dataset = load_dataset("parquet", data_files={'train': TRAIN_FILE}, split='train')
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 2. 获取基本信息
    print(f"\n数据加载成功! 总行数: {len(dataset)}")
    print("-" * 40)
    
    # 3. 检查列名
    print(f"列名 (Columns): {dataset.column_names}")
    
    # 4. 检查数据类型
    print(f"TYPES 数据类型: {dataset.features}")
    
    # 5. 打印前 3 行样本
    print("-" * 40)
    print("样本预览 (前3行):")
    df = pd.DataFrame(dataset.select(range(3))) 
    print(df)
    print("-" * 40)

    # ================= [新增功能] 检查类别分布 =================
    print("\n标签分布统计 (Class Distribution):")
    
    if 'label' in dataset.column_names:
        # 获取所有标签
        labels = dataset['label']
        # 使用 pandas 快速统计
        label_counts = pd.Series(labels).value_counts().sort_index()
        print(label_counts)
        
        # 获取具体数量 (防止某个类别完全不存在的情况)
        count_1 = label_counts.get(1, 0)
        count_0 = label_counts.get(0, 0)
        
        print(f"\n   -> Label 0 (健康/负例) 数量: {count_0}")
        print(f"   -> Label 1 (风险/正例) 数量: {count_1}")
        
        total = count_0 + count_1
        ratio_1 = (count_1 / total) * 100 if total > 0 else 0
        
        print(f"   -> Label 1 占比: {ratio_1:.2f}%")

        # 回答您的问题：Label 1 是否更多？
        print(f"\n   结论: ", end="")
        if count_1 > count_0:
            print(f"是的，Label 1 (风险样本) 更多。 (多 {count_1 - count_0} 条)")
            print("   建议: 数据主要偏向正样本，如果不做加权，模型可能会倾向于把所有人都判为‘有风险’。")
        elif count_0 > count_1:
            print(f"不是，Label 0 (健康样本) 更多。 (多 {count_0 - count_1} 条)")
            print("   建议: 这是医疗数据的常见情况（健康人更多）。在微调代码中建议保留 'Weighted Loss' (加权损失)。")
        else:
            print("两者数量完全相等 (完美平衡)。")
            
    else:
        print("   无法统计：未找到 'label' 列")
    
    print("-" * 40)

    # =================适配性诊断=================
    print("\nMentalBERT 适配性诊断报告:")
    
    # 诊断 1: 文本列是否存在
    text_cols = [col for col in dataset.column_names if 'text' in col or 'content' in col or 'sentence' in col]
    if text_cols:
        print(f"  [通过] 找到文本列: '{text_cols[0]}'")
    else:
        print(f"  [警告] 未找到常见文本列名 (text/content/sentence)")

    # 诊断 2: 标签格式
    if 'label' in dataset.column_names:
        sample_label = dataset[0]['label']
        if isinstance(sample_label, int):
             print(f"  [通过] 标签格式为整数 (int)，可以直接训练。")
        elif isinstance(sample_label, str):
             print(f"  [注意] 标签格式为字符串 (str)，需要转换。")
    else:
        print(f"  [警告] 未找到 'label' 列。")

if __name__ == "__main__":
    inspect()