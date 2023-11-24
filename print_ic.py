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


###################################
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 假设你的DataFrame名为df，列是各个股票ukey的因子
# good_cols是包含20个因子列的列表

# 创建一个空的DataFrame来存储相似度矩阵
similarity_matrix = pd.DataFrame()

# 按时间点进行分组
grouped = df.groupby('time')

# 遍历每个时间点的分组
for time, group in grouped:
    # 获取该时间点下的股票因子数据
    factor_data = group[good_cols]
    
    # 计算余弦相似度
    cosine_sim = cosine_similarity(factor_data)
    
    # 将相似度矩阵转换为DataFrame
    sim_matrix = pd.DataFrame(
        cosine_sim, 
        columns=factor_data.index, 
        index=factor_data.index
    )
    
    # 将该时间点下的相似度矩阵添加到总的相似度矩阵中
    similarity_matrix = pd.concat([similarity_matrix, sim_matrix], axis=0)

# 打印相似度矩阵
print(similarity_matrix)
