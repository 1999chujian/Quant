import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 假设你的DataFrame名为df，包含"Factors"和"Label"两列数据

# 计算每个"Factors"列与"Label"列的相关性系数和信息系数
ic_dict = {}

for column in df.columns:
    if column != "Label":
        correlation, _ = pearsonr(df[column], df["Label"])
        ic_dict[column] = correlation

# 按照IC值的绝对值大小排序
sorted_ic = sorted(ic_dict.items(), key=lambda x: abs(x[1]), reverse=True)

# 打印排序后的结果
for column, ic in sorted_ic:
    print(f"Column: {column}, IC: {ic}")
