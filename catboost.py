import pandas as pd
import pickle
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import zscore





with open('/home/chujian/yuanlan_code/bs/train.pkl', 'rb') as f:
    train_df = pickle.load(f)

with open('/home/chujian/yuanlan_code/bs/test.pkl', 'rb') as f:
    test_df = pickle.load(f)

test_label = pd.read_csv('/home/chujian/yuanlan_code/bs/test_label.csv')
test_df = pd.merge(test_df, test_label, on = ['time_id','stock_id'],how='inner')

#train_df.dropna(inplace=True)


def train_val_oos_cut1(cut_date_list, RAW_DATA):
    train_data = RAW_DATA[(RAW_DATA["time_id"] >= cut_date_list[0]) & (RAW_DATA["time_id"] <= cut_date_list[1])]
    val_data = RAW_DATA[(RAW_DATA["time_id"] >= cut_date_list[2]) & (RAW_DATA["time_id"] <= cut_date_list[3])]
    print(
        f"Training data range: [{train_data['time_id'].min()}, {train_data['time_id'].max()}], count: {len(train_data['time_id'].unique())}")
    print(
        f"Validation data range: [{val_data['time_id'].min()}, {val_data['time_id'].max()}], count: {len(val_data['time_id'].unique())}")
    return train_data, val_data


cut_date_list = [0, 660,661, 727]
train_df,val_df = train_val_oos_cut1(cut_date_list, train_df)



# # 填充缺失数据
# train_df.fillna(0, inplace=True)
# val_df.fillna(0, inplace=True)
# test_df.fillna(0, inplace=True)

# train_df['target_prev']=train_df.groupby('stock_id')['label'].shift(1)
# train_df = train_df.dropna(axis=0)
#
# val_df['target_prev']=val_df.groupby('stock_id')['label'].shift(1)
# val_df = val_df.dropna(axis=0)
#
# test_df['target_prev']=test_df.groupby('stock_id')['label'].shift(1)
# test_df.fillna(0, inplace=True)

train_df = train_df.set_index(['time_id', 'stock_id'])
#val_df = val_df.set_index(['time_id', 'stock_id'])
test_df = test_df.set_index(['time_id', 'stock_id'])

col = train_df.columns[:-1]

X = train_df[col].values
y = train_df['label'].values.reshape(-1, 1)
y = np.log(y + 0.7)



X_test = test_df[col].values




# 构建CatBoost模型
params = {
    'iterations': 2000,
    'loss_function': 'Quantile',
    'depth': 5,
    'learning_rate': 0.05,
    'subsample': 1,
    'colsample_bylevel': 1,
    'thread_count': -1,
    'random_state':1403,
}

model = CatBoostRegressor(**params)
#model.fit(X, y.ravel(), eval_set=(X_val, y_val),verbose=True, use_best_model=True)
model.fit(X, y.ravel(), verbose=1000,)

y_pred = model.predict(X_test)

with open('/home/chujian/yuanlan_code/bs/test.pkl', 'rb') as f:
    test_df = pickle.load(f)
test_df = test_df.set_index(['time_id', 'stock_id'])

test_label = pd.read_csv('/home/chujian/yuanlan_code/bs/test_label.csv')
test_label = test_label.set_index(['time_id', 'stock_id'])


result = pd.DataFrame(y_pred, index=test_df.index, columns=['pred'])
test_df['pred'] = result

def rank_ic(test_df, test_label):
    result = pd.concat([test_df, test_label], axis=1)
    #result = result.reset_index()  # 将 time_id 转换为列
    rank_ic = result.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()
    return rank_ic

rank_ic = rank_ic(test_df, test_label)

print(cut_date_list)

importances = model.feature_importances_
feature_names = train_df[col].columns
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
features_list =[]
for i, idx in enumerate(indices):
    if i >= 1=300:
        break
    feature_name = f'{feature_names[idx]}'
    features_list.append(feature_name)
    print(f"{i + 1}. {feature_names[idx]} ({importances[idx]:.4f})")
print(features_list)

print('rank_ic: {:.4f}'.format(rank_ic))


import pandas as pd
import pickle
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import zscore




with open('/home/chujian/yuanlan_code/bs/train.pkl', 'rb') as f:
    train_df = pickle.load(f)

with open('/home/chujian/yuanlan_code/bs/test.pkl', 'rb') as f:
    test_df = pickle.load(f)

test_label = pd.read_csv('/home/chujian/yuanlan_code/bs/test_label.csv')
test_df = pd.merge(test_df, test_label, on = ['time_id','stock_id'],how='inner')

#train_df.dropna(inplace=True)


def train_val_oos_cut1(cut_date_list, RAW_DATA):
    train_data = RAW_DATA[(RAW_DATA["time_id"] >= cut_date_list[0]) & (RAW_DATA["time_id"] <= cut_date_list[1])]
    val_data = RAW_DATA[(RAW_DATA["time_id"] >= cut_date_list[2]) & (RAW_DATA["time_id"] <= cut_date_list[3])]
    print(
        f"Training data range: [{train_data['time_id'].min()}, {train_data['time_id'].max()}], count: {len(train_data['time_id'].unique())}")
    print(
        f"Validation data range: [{val_data['time_id'].min()}, {val_data['time_id'].max()}], count: {len(val_data['time_id'].unique())}")
    return train_data, val_data


cut_date_list = [0, 660,661, 727]
train_df,val_df = train_val_oos_cut1(cut_date_list, train_df)


train_df = train_df.set_index(['time_id', 'stock_id'])
#val_df = val_df.set_index(['time_id', 'stock_id'])
test_df = test_df.set_index(['time_id', 'stock_id'])

col = train_df.columns[:-1]


## 使用log1p变换将特征基本拉到一个尺度进行建模

train_df[col] = np.log1p(train_df[col])

test_df[col] = np.log1p(train_df[col])

X = train_df[col].values
y = train_df['label'].values.reshape(-1, 1)
y = np.log(y + 0.7)

# 构建CatBoost模型
params = {
    'iterations': 2000,
    'loss_function': 'Quantile',
    'depth': 5,
    'learning_rate': 0.05,
    'subsample': 1,
    'colsample_bylevel': 1,
    'thread_count': -1,
    'random_state':42,
}

model = CatBoostRegressor(**params)
#model.fit(X, y.ravel(), eval_set=(X_val, y_val),verbose=True, use_best_model=True)
model.fit(X, y.ravel(), verbose=True,)

y_pred = model.predict(X_test)

with open('/home/chujian/yuanlan_code/bs/test.pkl', 'rb') as f:
    test_df = pickle.load(f)
test_df = test_df.set_index(['time_id', 'stock_id'])

test_label = pd.read_csv('/home/chujian/yuanlan_code/bs/test_label.csv')
test_label = test_label.set_index(['time_id', 'stock_id'])

result = pd.DataFrame(y_pred, index=test_df.index, columns=['pred'])
test_df['pred'] = result

def rank_ic(test_df, test_label):
    result = pd.concat([test_df, test_label], axis=1)
    #result = result.reset_index()  # 将 time_id 转换为列
    rank_ic = result.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()
    return rank_ic

rank_ic = rank_ic(test_df, test_label)

print(cut_date_list)

importances = model.feature_importances_
feature_names = train_df[col].columns
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
features_list =[]
for i, idx in enumerate(indices):
    if i >= 100:
        break
    feature_name = f'{feature_names[idx]}'
    features_list.append(feature_name)
    print(f"{i + 1}. {feature_names[idx]} ({importances[idx]:.4f})")
print(features_list)

print('rank_ic: {:.4f}'.format(rank_ic))
