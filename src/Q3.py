#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入必要的库
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value

# 设置matplotlib参数以支持中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ### 由数据处理部分的相关性分析：
# 1.种植成本与销售单价之间的相关性:种植成本与销售单价之间的相关性系数高达0.76，证明种植成本越高，销售单价越高，介于这种情况，调整部分参数。
# 2.同类的作物可以相互替代。

# ## 定义农作物数据

# ### 与第一问相似 先处理平旱地、梯田和水浇地

# In[2]:


# 初始化随机数种子
np.random.seed(10)

# 定义农作物的基本信息
duration = 7
crops_count = 15
fields_count = 26
data = {
    '作物': [f'作物{i+1}' for i in range(crops_count)],
    '亩产量': [400,500,400,350,415,800,1000,400,630,525,110,3000,2200,420,525],  
    '种植成本': [400, 400, 350,350,350,450,500,360,400,360,350,1000,2000,400,350], 
    '销售价格': [3.25, 7.5, 8.25, 7, 6.75, 3.5, 3, 6.75, 6, 7.5, 40, 1.5, 3.25, 5.5, 3.5], 
    '预期销售量': [57000, 21850, 22400, 33040, 9875, 170840, 132750, 71400, 3000,12500,1500,35100,36000,14000,10000],  
}

yield_data = np.array([
    [400,500,400,350,415,800,1000,400,630,525,110,3000,2200,420,525],
    [380,475,380,330,395,760,950,380,600,500,105,2850,2100,400,500],
    [360,450,360,315,375,720,900,360,570,475,100,2700,2000,380,475]
    ])

forbid_arrays = [6,7,7,1,4,8,6,2,3,4,5,8,6,8,9,10,1,7,14,15,11,12,1,13,6,3]
land_areas = [80,55,35,72,68,55,60,46,40,28,25,86,55,44,50,25,60,45,35,20,15,13,15,18,27,20]
index_array = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2]

# 创建 DataFrame
df = [pd.DataFrame(data)] * duration
yield_data = [yield_data] * duration


# ## 为特定作物的预期销售量&产量&成本变化创建随机因子

# In[3]:


adjustment_factor_1 = np.random.uniform(1.05,1.10,1)
adjustment_factor_2 = np.random.uniform(1.05,1.10,1)
    
df[0].loc[df[0]["作物"] == "作物6","预期销售量"] *= adjustment_factor_1
df[0].loc[df[0]["作物"] == "作物7","预期销售量"] *= adjustment_factor_2


f2 = np.random.uniform(0.95,1.05,1)
df[0].loc[(df[0]["作物"] != "作物6") & (df[0]["作物"] != "作物7") ,"预期销售量"] *= f2
  
f3 = np.random.uniform(0.9,1.1,yield_data[0].shape)
yield_data[0] = yield_data[0]*f3

df[0]["种植成本"] *= 1.05
    
for i in range(1,duration):
    # 作物6和作物7的预期销售量调整
    adjustment_factor_1 = np.random.uniform(1.05,1.10,1)
    adjustment_factor_2 = np.random.uniform(1.05,1.10,1)
    df[i].loc[df[i]["作物"] == "作物6","预期销售量"] = df[i-1].loc[df[i-1]["作物"] == "作物6","预期销售量"] * adjustment_factor_1
    df[i].loc[df[i]["作物"] == "作物7","预期销售量"] = df[i-1].loc[df[i-1]["作物"] == "作物7","预期销售量"] * adjustment_factor_2
    
    # 其他作物的预期销售量调整
    f2 = np.random.uniform(0.95,1.05,1)
    df[i].loc[(df[i]["作物"] != "作物6") & (df[i]["作物"] != "作物7") ,"预期销售量"] = df[i-1].loc[(df[i-1]["作物"] != "作物6") & (df[i-1]["作物"] != "作物7") ,"预期销售量"] * f2
    
    # 种植成本调整
    f3 = np.random.uniform(0.9,1.1,yield_data[0].shape)
    yield_data[i] = yield_data[i-1]*f3
    
    #产量数据调整
    df[i]["种植成本"] = df[i-1]["种植成本"] * 1.05


# In[4]:


# 创建线性规划模型，目标是最大化总利润
A_model = LpProblem("最优种植方案", LpMaximize)

# 初始化决策变量字典，用于存储不同年份、地块和作物的种植面积
x = {}
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            # 定义决策变量 x[year][i][j]，表示第 year 年地块 i 上种植作物 j 的面积，面积值必须大于等于0
            x[(year, i, j)] = LpVariable(f"x_{year}_{i}_{j}", lowBound=0.0)

# 构建目标函数，即总利润的计算
total_profit = 0

for year in range(duration):
    f3 = np.random.uniform(-0.1,0.1,crops_count)
    for j in range(crops_count):
        # 计算第 year 年作物 j 在所有地块上的预期总产量
        total_production = lpSum(yield_data[year][index_array[i],j] * x[(year, i, j)] for i in range(fields_count))
        expected_sales = df[year].loc[j, '预期销售量']  # 获取作物 j 在第 year 年的预期销售量
        
        # 根据总产量与预期销售量的关系计算利润
        if total_production <= expected_sales:
            # 如果总产量不超过预期销售量，则按正常价格计算利润
            profit = total_production * df[year].loc[j, '销售价格'] - lpSum(df[year].loc[j, '种植成本'] * x[(year, i, j)] for i in range(fields_count))
        else:
            # 如果总产量超过预期销售量，则超出部分按半价计算利润
            normal_sales = expected_sales
            excess_sales = total_production - expected_sales
            profit = (normal_sales * df[year].loc[j, '销售价格'] + 
                      excess_sales * (0.5 * df[year].loc[j, '销售价格'])) - lpSum(df[year].loc[j, '种植成本'] * x[(year, i, j)] for i in range(fields_count))
        
        # 将计算出的利润累加到总利润中
        total_profit += profit

# 将构建好的总利润作为目标函数添加到模型中
A_model += total_profit, "Total_Profit"

# 添加2024年作物种类的约束条件
for i in range(fields_count):
    A_model += lpSum(x[(0, i, j)] for j in range(crops_count) if j == forbid_arrays[i]) == 0, f"Different_Crop_Constraint_{i}"
    
# 添加销售量限制条件
for year in range(duration):
    for j in range(crops_count):
        production = 0
        for i in range(fields_count):
            production += yield_data[year][index_array[i],j] * x[(year, i, j)]  # 计算第 year 年作物 j 的总生产量
        expected_sales = df[year].loc[j, '预期销售量']  # 获取作物 j 在第 year 年的预期销售量
            # 设置生产量的下限和上限，确保生产量在预期销售量的80%到120%之间
        A_model += production >= 0.5 * expected_sales, f"Min_Sales_{year}_{j}"
        A_model += production <= 1.1 * expected_sales, f"Max_Sales_{year}_{j}"

# 添加每块地的面积限制条件
for year in range(duration):
    for i in range(fields_count):
        A_model += lpSum(x[(year, i, j)] for j in range(crops_count)) <= land_areas[i], f"Area_limit_{year}_{i}"

# 初始化布尔变量字典，用于指示某年某地块是否种植了某种作物
y = {}
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            y[(year, i, j)] = LpVariable(f"y_{year}_{i}_{j}", cat='Binary')

for year in range(1,duration):  # 从第 2 年开始
    for i in range(fields_count):
        for j in range(crops_count):
            # 添加作物轮作约束，确保同一地块不会连续两年种植同一作物
            A_model += x[(year - 1, i, j)] + x[(year, i, j)] <= land_areas[i], f"rotation_constraint_{year}_{i}_{j}"

# 求解模型并输出求解状态
A_model.solve()
print("求解状态:", LpStatus[A_model.status])

# 初始化数组用于存储输出结果
df_dst_1 = np.zeros((duration,fields_count,crops_count))

# 遍历决策变量，输出种植方案和面积
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            if value(x[(year, i, j)]) > 0:
                df_dst_1[year,i,j] = value(x[(year, i, j)])
                print(f"第 {year + 1} 年 地块 {i + 1} 种植 {df[year].loc[j, '作物']} 面积: {value(x[(year, i, j)])} 亩")


# ## 水浇地、普通大棚和智慧大棚
# ### 先处理水稻

# In[5]:


duration = 7
crops_count = 1
fields_count = 8
data_crop = {
    '作物': [f'作物{i+16}' for i in range(crops_count)],
    '亩产量': [500],  
    '种植成本': [680], 
    '销售价格': [7],
    '预期销售量':[21000]
}

forbid_arrays = [[20,36],[28,35],[21,35],[22,35],[17,36],[18,37],[16],[16]]# 定义禁种作物数组
land_areas = [15,10,14,6,10,12,22,20]# 定义每个地块的面积
df = [pd.DataFrame(data_crop)]*duration# 创建数据框架列表，每个元素都是相同的数据框架，表示不同时间点的数据

f2 = np.random.uniform(0.95,1.05,1) # 随机调整预期销售量，使用一个介于0.95和1.05之间的随机因子
df[0]["预期销售量"] *= f2
     
f3 = np.random.uniform(0.9,1.1,1) # 随机调整亩产量，使用一个介于0.9和1.1之间的随机因子
df[0]["亩产量"] *= f3

df[0]["种植成本"] *= 1.05 # 种植成本增加5%

for i in range(1,duration):
    # 随机调整预期销售量，使用一个介于0.95和1.05之间的随机因子
    f2 = np.random.uniform(0.95,1.05,1)
    df[i]["预期销售量"] = df[i]["预期销售量"] * f2
     
    # 随机调整亩产量，使用一个介于0.9和1.1之间的随机因子
    f3 = np.random.uniform(0.9,1.1,1)
    df[i]["亩产量"] = df[i]["亩产量"] * f3
    
    # 种植成本按照前一时间点的1.05倍增加
    df[i]["种植成本"] = df[i-1]["种植成本"] * 1.05


# In[6]:


# 创建线性规划模型，用于最大化水稻种植的总利润
crop_model = LpProblem("水稻种植方案", LpMaximize)

# 初始化决策变量字典，用于存储不同年份、地块和作物的种植面积
x_crop = {}
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            # 定义决策变量 x_crop[year][i][j]，表示第 year 年地块 i 上种植作物 j 的面积，面积值必须大于等于0
            x_crop[(year, i, j)] = LpVariable(f"x_crop_{year}_{i}_{j}", lowBound=0.0)

# 构建目标函数，即总利润的计算
total_profit_first = 0

for year in range(duration):
    for j in range(crops_count):
        # 计算第 year 年作物 j 在所有地块上的预期总产量
        total_production = lpSum(df[year].loc[j, '亩产量'] * x_crop[(year, i, j)] for i in range(fields_count))
        expected_sales = df[year].loc[j, '预期销售量']  # 获取作物 j 在第 year 年的预期销售量
        
        # 根据总产量与预期销售量的关系计算利润
        if total_production <= expected_sales:
            # 如果总产量不超过预期销售量，则按正常价格计算利润
            profit = total_production * df[year].loc[j, '销售价格'] - lpSum(df[year].loc[j, '种植成本'] * x_crop[(year, i, j)] for i in range(fields_count))
        else:
            # 如果总产量超过预期销售量，则超出部分按半价计算利润
            normal_sales = expected_sales
            excess_sales = total_production - expected_sales
            profit = (normal_sales * df[year].loc[j, '销售价格'] + 
                      excess_sales * (0.3 * df[year].loc[j, '销售价格'])) - lpSum(df[year].loc[j, '种植成本'] * x_crop[(year, i, j)] for i in range(fields_count))
        
        # 将计算出的利润累加到总利润中
        total_profit_first += profit

# 将构建好的总利润作为目标函数添加到模型中
crop_model += total_profit_first, "Total_Profit"

# 添加销售量限制条件
for year in range(duration):
    for j in range(crops_count):
        production = 0
        for i in range(fields_count):
            production += df[year].loc[j, '亩产量'] * x_crop[(year, i, j)]
        expected_sales = df[year].loc[j, '预期销售量']
        # 设置生产量的下限和上限，确保生产量在预期销售量的50%到110%之间
        crop_model += production >= 0.5 * expected_sales, f"Min_Sales_{year}_{j}"
        crop_model += production <= 1.1 * expected_sales, f"Max_Sales_{year}_{j}"

# 添加每块地的面积限制条件
for year in range(duration):
    for i in range(fields_count):
        crop_model += lpSum(x_crop[(year, i, j)] for j in range(crops_count)) <= land_areas[i], f"Area_limit_{year}_{i}"

# 添加禁种作物约束
for i in range(fields_count):
    for year in range(duration):
        # 对于每个地块，确保不会种植任何禁种作物
        for forbidden_crop in forbid_arrays[i]:
            # 如果禁种作物列表不为空，则添加约束
            if isinstance(forbidden_crop, list) and forbidden_crop:
                crop_model += lpSum(x_crop[(year, i, j)] for j in forbidden_crop) == 0, f"Forbidden_Crop_{year}_{i}"

# 初始化布尔变量字典，用于指示某年某地块是否种植了某种作物
y = {}
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            y[(year, i, j)] = LpVariable(f"y_{year}_{i}_{j}", cat='Binary')

# 更新决策变量与布尔变量之间的关系，确保种植面积与地块面积和种植状态相匹配
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            crop_model += x_crop[(year, i, j)] <= land_areas[i] * y[(year, i, j)], f"Link_{year}_{i}_{j}"

# 添加作物轮作约束，确保同一地块不会连续两年种植同一作物
for year in range(1, duration):
    for i in range(fields_count):
        for j in range(crops_count):
            crop_model += y[(year-1, i, j)] + y[(year, i, j)] <= 1, f"Crop_rotation_{year}_{i}_{j}"

# 求解模型并输出求解状态
crop_model.solve()
print("求解状态:", LpStatus[crop_model.status])

# 输出种植方案的结果
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            if value(x_crop[(year, i, j)]) > 0:
                print(f"第 {year + 1} 年 地块 D{i + 1} 种植 {df[year].loc[j, '作物']} 面积: {value(x_crop[(year, i, j)])} 亩")

use_land_areas = np.zeros((duration,fields_count))

for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            if value(x_crop[(year, i, j)]) > 0:
                use_land_areas[year,i] = value(x_crop[(year, i, j)])


# ## 水浇地第二季部分(大白菜, 白萝卜, 红萝卜)

# In[7]:


duration = 7
crops_count = 3
fields_count = 8
data22 = {
    '作物': [f'作物{i+35}' for i in range(crops_count)],
    '亩产量': [5000,4000,3000],  
    '种植成本': [2000,500,500], 
    '销售价格': [2.5,2.5,3.25],
    '预期销售量':[150000,100000,36000]
}

forbid_arrays = [[20,36],[28,35],[21,35],[22,35],[17,36],[18,37],[16],[16]]
land_areas = [15,10,14,6,10,12,22,20]

df = [pd.DataFrame(data22)]*duration

f2 = np.random.uniform(0.95,1.05,crops_count)# 随机调整预期销售量，使用一个介于0.95和1.05之间的随机因子数组
df[0]["预期销售量"] *= f2
     
f3 = np.random.uniform(0.9,1.1,crops_count)# 随机调整亩产量，使用一个介于0.9和1.1之间的随机因子数组
df[0]["亩产量"] *= f3

df[0]["种植成本"] *= 1.05

df[0]["销售价格"] *= 1.05

for i in range(1,duration):
    # 随机调整预期销售量，使用一个介于0.95和1.05之间的随机因子数组
    f2 = np.random.uniform(0.95,1.05,crops_count)
    df[i]["预期销售量"] = df[i]["预期销售量"] * f2
     
    # 随机调整亩产量，使用一个介于0.9和1.1之间的随机因子数组
    f3 = np.random.uniform(0.9,1.1,crops_count)
    df[i]["亩产量"] = df[i]["亩产量"] * f3
    
    #种植成本按照前一时间点的1.05倍增加
    df[i]["种植成本"] = df[i-1]["种植成本"] * 1.05
    
    #销售价格按照前一时间点的1.05倍增加
    df[i]["销售价格"] = df[i-1]["销售价格"] * 1.05


# In[8]:


# 创建线性规划模型
irrigated_model = LpProblem("水浇地种植方案", LpMaximize)
# 决策变量
x_irrigated = {}

for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            x_irrigated[(year, i, j)] = LpVariable(f"x_irrigated_{year}_{i}_{j}", lowBound=0.0)

# 目标函数：最大化总利润
total_profit_first = 0

for year in range(duration):
    for j in range(crops_count):
        # 计算某种作物在所有地块上的总产量
        total_production = lpSum( df[year].loc[j, '亩产量'] * x_irrigated[(year, i, j)] for i in range(fields_count))
        
        expected_sales = df[year].loc[j, '预期销售量']  # 预期销售量
        
        # 判断总产量与预期销售量的关系
        if total_production <= expected_sales:
            # 正常售卖
            profit = total_production * df[year].loc[j, '销售价格'] - lpSum(df[year].loc[j, '种植成本'] * x_irrigated[(year, i, j)] for i in range(fields_count))
        else:
            # 超出部分售价为50%
            normal_sales = expected_sales
            excess_sales = total_production - expected_sales
            profit = (normal_sales * df[year].loc[j, '销售价格'] + 
                      excess_sales * (0.3 * df[year].loc[j, '销售价格'])) - lpSum(df[year].loc[j, '种植成本'] * x_irrigated[(year, i, j)] for i in range(fields_count))
        
        # 累加利润
        total_profit_first += profit

# 添加目标函数
irrigated_model += total_profit_first, "Total_Profit"


# 添加销售量限制
for year in range(duration):
    for j in range(crops_count):
        production = 0
        for i in range(fields_count):
            production += df[year].loc[j, '亩产量'] * x_irrigated[(year, i, j)]
        expected_sales = df[year].loc[j, '预期销售量']
        irrigated_model += production >= 0.5 * expected_sales, f"Min_Sales_{year}_{j}"
        irrigated_model += production <= 1.1 * expected_sales, f"Max_Sales_{year}_{j}"

# 每块地的面积限制
for year in range(duration):
    for i in range(fields_count):
        irrigated_model += lpSum(x_irrigated[(year, i, j)] for j in range(crops_count)) <= land_areas[i] - use_land_areas[year,i], f"Area_limit_{year}_{i}"

# 添加禁种作物约束
for i in range(fields_count):
    for year in range(duration):
        # 处理可能的多个禁种作物
        for forbidden_crop in forbid_arrays[i]:
            # 检查 forbidden_crop 是否是列表且有值
            if isinstance(forbidden_crop, list) and forbidden_crop:
                irrigated_model += lpSum(x_irrigated[(year, i, j)] for j in forbidden_crop) == 0, f"Forbidden_Crop_{year}_{i}"

# 添加约束 ：不能重茬种植         
y = {}
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            y[(year, i, j)] = LpVariable(f"y_{year}_{i}_{j}", cat='Binary')

# 更新决策变量与布尔变量之间的关系
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            irrigated_model += x_irrigated[(year, i, j)] <= land_areas[i] * y[(year, i, j)], f"Link_{year}_{i}_{j}"

# 添加约束条件：如果去年种植了某种作物，下一年就不能种植
for year in range(1, duration):
    for i in range(fields_count):
        for j in range(crops_count):
            irrigated_model += y[(year-1, i, j)] + y[(year, i, j)] <= 1, f"Crop_rotation_{year}_{i}_{j}"
        
        
# 求解模型
irrigated_model.solve()
df_dst_3 = np.zeros((duration,fields_count,crops_count))
# 输出结果
print("求解状态:", LpStatus[irrigated_model.status])
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            if value(x_irrigated[(year, i, j)]) > 0:
                df_dst_3[year,i,j] = value(x_irrigated[(year, i, j)])
                print(f"第 {year + 1} 年 地块 D{i + 1} 种植 {df[year].loc[j, '作物']} 面积: {value(x_irrigated[(year, i, j)])} 亩")


# ## 普通大棚部分（第二季）
# ### 即榆黄菇,香菇,白灵菇,羊肚菌

# In[9]:


duration = 7
crops_count = 4
fields_count = 16
data23 = {
    '作物': [f'作物{i+38}' for i in range(crops_count)],
    '亩产量': [5000,4000,10000,1000],  
    '种植成本': [3000,2000,10000,10000], 
    '销售价格': [57.5,19,16,100],
    '预期销售量':[9000,7200,18000,4200]
}

forbid_arrays = [[18,38],[24,38],[25,38],[26,39],[28,39],[27,39],[19,40],[19,40],[18,40],[17,41],[17,41],[22,41],[21,41],[29,41],[30,27,41],[31,41]]
land_areas = [0.6]*fields_count

df = [pd.DataFrame(data23)]*duration


# In[10]:


f6 = np.random.uniform(1.05,1.10,1)
f7 = np.random.uniform(1.05,1.10,3)
    
df[0].loc[df[0]["作物"] == "作物41","预期销售量"] *= f6
df[0].loc[df[0]["作物"] != "作物41","预期销售量"] *= f7

f2 = np.random.uniform(0.95,1.05,crops_count)
df[0]["预期销售量"] *= f2

f3 = np.random.uniform(0.9,1.1,yield_data[0].shape)
yield_data[0] = yield_data[0]*f3

df[0]["种植成本"] *= 1.05
    
for i in range(1,duration):

    f2 = np.random.uniform(0.95,1.05,1)
    df[i].loc[(df[i]["作物"] != "作物6") & (df[i]["作物"] != "作物7") ,"预期销售量"] = df[i-1].loc[(df[i-1]["作物"] != "作物6") & (df[i-1]["作物"] != "作物7") ,"预期销售量"] * f2
    
    f3 = np.random.uniform(0.9,1.1,yield_data[0].shape)
    yield_data[i] = yield_data[i-1]*f3
 
    df[i]["种植成本"] = df[i-1]["种植成本"] * 1.05
    
    f6 = np.random.uniform(1.05,1.10,1)
    df[i].loc[df[i]["作物"] == "作物41","预期销售量"] = df[i-1].loc[df[i-1]["作物"] == "作物41","预期销售量"] * f6
    
    f7 = np.random.uniform(1.05,1.10,3)
    df[i].loc[df[i]["作物"] != "作物41","预期销售量"] = df[i-1].loc[df[i-1]["作物"] != "作物41","预期销售量"] * f7


# In[11]:


# 创建线性规划模型
greenhouse_model = LpProblem("普通大棚种植方案", LpMaximize)
# 决策变量
x_greenhouse = {}

for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            x_greenhouse[(year, i, j)] = LpVariable(f"x_greenhouse_{year}_{i}_{j}", lowBound=0.0)

# 目标函数：最大化总利润
total_profit_first = 0

for year in range(duration):
    for j in range(crops_count):
        # 计算某种作物在所有地块上的总产量
        total_production = lpSum(df[year].loc[j, '亩产量'] * x_greenhouse[(year, i, j)] for i in range(fields_count))
        
        expected_sales = df[year].loc[j, '预期销售量']  # 预期销售量
        
        # 判断总产量与预期销售量的关系
        if total_production <= expected_sales:
            # 正常售卖
            profit = total_production * df[year].loc[j, '销售价格'] - lpSum(df[year].loc[j, '种植成本'] * x_greenhouse[(year, i, j)] for i in range(fields_count))
        else:
            # 超出部分售价为50%
            normal_sales = expected_sales
            excess_sales = total_production - expected_sales
            profit = (normal_sales * df[year].loc[j, '销售价格'] + 
                      excess_sales * (0.3 * df[year].loc[j, '销售价格'])) - lpSum(df[year].loc[j, '种植成本'] * x_greenhouse[(year, i, j)] for i in range(fields_count))
        
        # 累加利润
        total_profit_first += profit
greenhouse_model += total_profit_first, "Total_Profit"

# 添加销售量限制
for year in range(duration):
    for j in range(crops_count):
        production = 0
        for i in range(fields_count):
            production += df[year].loc[j, '亩产量'] * x_greenhouse[(year, i, j)]
        expected_sales = df[year].loc[j, '预期销售量']
        greenhouse_model += production >= 0.5 * expected_sales, f"Min_Sales_{year}_{j}"
        greenhouse_model += production <= 1.1 * expected_sales, f"Max_Sales_{year}_{j}"

# 每块地的面积限制
for year in range(duration):
    for i in range(fields_count):
        greenhouse_model += lpSum(x_greenhouse[(year, i, j)] for j in range(crops_count)) <= land_areas[i], f"Area_limit_{year}_{i}"

# 添加禁种作物约束
for i in range(fields_count):
    for year in range(duration):
        # 处理可能的多个禁种作物
        for forbidden_crop in forbid_arrays[i]:
            # 检查 forbidden_crop 是否是列表且有值
            if isinstance(forbidden_crop, list) and forbidden_crop:
                greenhouse_model += lpSum(x_greenhouse[(year, i, j)] for j in forbidden_crop) == 0, f"Forbidden_Crop_{year}_{i}"

# 添加约束 ：不能重茬种植          
y = {}
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            y[(year, i, j)] = LpVariable(f"y_{year}_{i}_{j}", cat='Binary')

# 更新决策变量与布尔变量之间的关系
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            greenhouse_model += x_greenhouse[(year, i, j)] <= land_areas[i] * y[(year, i, j)], f"Link_{year}_{i}_{j}"

# 添加约束条件：如果去年种植了某种作物，下一年就不能种植
for year in range(1, duration):
    for i in range(fields_count):
        for j in range(crops_count):
            greenhouse_model += y[(year-1, i, j)] + y[(year, i, j)] <= 1, f"Crop_rotation_{year}_{i}_{j}"
        
        
# 求解模型
greenhouse_model.solve()
df_dst_4 = np.zeros((duration,fields_count,crops_count))
# 输出结果
print("求解状态:", LpStatus[greenhouse_model.status])
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            if value(x_greenhouse[(year, i, j)]) > 0:
                df_dst_4[year,i,j] = value(x_greenhouse[(year, i, j)])
                print(f"第 {year + 1} 年 第二季 地块 E{i + 1} 种植 {df[year].loc[j, '作物']} 面积: {value(x_greenhouse[(year, i, j)])} 亩")


# ## 其他所有部分（水浇地第一季，普通大棚第一季，智慧大棚第一季，智慧大棚第二季）
# ### 该部分均计算两季作物

# In[12]:


duration = 7
crops_count = 18*4
fields_count = 8+16+4+4
data24 = {
    '作物': [f'作物{i+17}' for i in range(18)]*4,
    '亩产量': [3000,2000,3000,2000,2400,6400,2700,2400,3300,3700,4100,3200,12000,4100,1600,10000,5000,5500,
            3600,2400,3600,2400,3000,8000,3300,3000,4000,4500,5000,4000,15000,5000,2000,12000,6000,6600,
            3600,2400,3600,2400,3000,8000,3300,3000,4000,4500,5000,4000,15000,5000,2000,12000,6000,6600,
            3200,2200,3200,2200,2700,7200,3000,2700,3600,4100,4500,3600,13500,4500,1800,11000,5400,6000,
           ],  
    '种植成本': [2000,1000,2000,2000,2000,2000,2300,1600,2400,2900,1600,1600,2900,1600,1000,4100,2000,900,
            2400,1200,2400,2400,2400,2400,2700,2000,3000,3500,2000,2000,3500,2000,1200,5000,2500,1100,
            2400,1200,2400,2400,2400,2400,2700,2000,3000,3500,2000,2000,3500,2000,1200,5000,2500,1100,
            2640,1320,2640,2640,2640,2640,3000,2200,3300,3850,2200,2200,3850,2200,1300,5500,2750,1200 
            ], 
    '销售价格': [8,6.75,6.5,3.75,6.25,5.5,5.75,5.25,5.5,6.5,5,5.75,7,5.25,7.25,4.5,4.5,4,
             8,6.75,6.5,3.75,6.25,5.5,5.75,5.25,5.5,6.5,5,5.75,7,5.25,7.25,4.5,4.5,4,
             8,6.75,6.5,3.75,6.25,5.5,5.75,5.25,5.5,6.5,5,5.75,7,5.25,7.25,4.5,4.5,4,
             9.6,8.1,7.8,4.5,7.5,6.6,6.9,6.8,6.6,7.8,6,6.9,8.4,6.3,8.7,5.4,5.4,4.8
            ]
}

df = [pd.DataFrame(data24)]*duration
forbid_arrays = [[20,36],[28,35],[21,35],[22,35],[17,36],[18,37],[16],[16],
     [18,38],[24,38],[25,38],[26,39],[28,39],[27,39],[19,40],[19,40],[18,40],[17,41],[17,41],[22,41],[21,41],[29,41],[30,27,41],[31,41],
    [32,33],[25,26],[17],[19],[24,21],[22,29],[28,30],[34,23]]      
                  
     #[32,33,24,21],[25,26,22,29],[17,28,30],[19,34,23]
                  
land_areas = [15,10,14,6,10,12,22,20] + [0.6]*16 + [0.6] * 4 + [0.6]*4
sell_amount = [36240,26880,6240,30000,36210,45360,900,2610,3480,3930,4500,35480,13050,2850,1200,3300,1620,1800]*duration
use_land_areas_plus = np.zeros((duration,fields_count))
use_land_areas_plus[:duration,:8] = use_land_areas


# In[13]:


f2 = np.random.uniform(0.95,1.05,crops_count)
sell_amount[0] *= f2
     
f3 = np.random.uniform(0.9,1.1,crops_count)
df[0]["亩产量"] *= f3

df[0]["种植成本"] *= 1.05

df[0]["销售价格"] *= 1.05
for i in range(1,duration):#根据题设在范围内随机调整变量波动
    f2 = np.random.uniform(0.95,1.05,crops_count)
    sell_amount[i] = sell_amount[i] * f2
     
    f3 = np.random.uniform(0.9,1.1,crops_count)
    df[i]["亩产量"] = df[i]["亩产量"] * f3

    df[i]["种植成本"] = df[i-1]["种植成本"] * 1.05
    
    df[i]["销售价格"] = df[i-1]["销售价格"] * 1.05


# In[14]:


# 创建线性规划模型
cultivation_model = LpProblem("其他种植方案", LpMaximize)
# 决策变量
x_cultivation = {}

for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            x_cultivation[(year, i, j)] = LpVariable(f"x_cultivation_{year}_{i}_{j}", lowBound=0.0)

# 目标函数：最大化总利润
total_profit_first = 0

for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count):
            production = df[year].loc[j, '亩产量'] * x_cultivation[(year, i, j)]
            #expected_sales = df[year].loc[j, '预期销售量']
            expected_sales = sell_amount[year][j%18]
            
            if production <= expected_sales:
                profit = production * df[year].loc[j, '销售价格'] - df[year].loc[j, '种植成本'] * x_cultivation[(year, i, j)]
            else:
                normal_sales = expected_sales
                excess_sales = production - expected_sales
                profit = (normal_sales * df[year].loc[j, '销售价格'] +
                          excess_sales * (0.3 * df[year].loc[j, '销售价格'])) - df[year].loc[j, '种植成本'] * x_cultivation[(year, i, j)]

            total_profit_first += profit  # 累加利润

# 添加目标函数
cultivation_model += total_profit_first, "Total_Profit"

# 添加销售量限制
for year in range(duration):
    for j in range(18):
        production = 0
        for i in range(fields_count):
            production += df[year].loc[j, '亩产量'] * x_cultivation[(year, i, j)]
            production += df[year].loc[j+18*1, '亩产量'] * x_cultivation[(year, i, j+18*1)]
            production += df[year].loc[j+18*2, '亩产量'] * x_cultivation[(year, i, j+18*2)]
            production += df[year].loc[j+18*3, '亩产量'] * x_cultivation[(year, i, j+18*3)]
        expected_sales = sell_amount[year][j]
        cultivation_model += production >= 0.2 * expected_sales, f"Min_Sales_{year}_{j}"
        # 松弛系数 0.5 -> 0.4
        cultivation_model += production <= 1.1 * expected_sales, f"Max_Sales_{year}_{j}"

# 每块地的面积限制
for year in range(duration):
    for i in range(fields_count):
        cultivation_model += lpSum(x_cultivation[(year, i, j)] for j in range(crops_count)) <= land_areas[i] - use_land_areas_plus[year,i], f"Area_limit_{year}_{i}"

# 添加禁种作物约束
for i in range(fields_count):
    for year in range(duration):
        # 处理可能的多个禁种作物
        for forbidden_crop in forbid_arrays[i]:
            # 检查 forbidden_crop 是否是列表且有值
            if isinstance(forbidden_crop, list) and forbidden_crop:
                cultivation_model += lpSum(x_cultivation[(year, i, j)] for j in forbidden_crop) == 0, f"Forbidden_Crop_{year}_{i}"

            # 添加约束条件：确保每个地块在过去三年内至少种植一次豆类作物      
bean_crops = range(3)  
for year in range(3, duration):  # 从2023年开始
    for i in range(fields_count):
        cultivation_model += lpSum(x_cultivation[(y, i, j)] for y in range(year - 3, year) for j in bean_crops) >= 1, f"Bean_Crop_Constraint_{year}_{i}"

for year in range(1,duration):  # 从第 2 年开始
    for i in range(fields_count):
        for j in range(crops_count):
            # 如果某年地块 i 种植了作物 j，则下一年该地块不能再种植作物 j
            cultivation_model += x_cultivation[(year - 1, i, j)] + x_cultivation[(year, i, j)] <= land_areas[i], f"rotation_constraint_{year}_{i}_{j}"


# 求解模型
print(111)
cultivation_model.solve()
df_dst_5 = np.zeros((duration,fields_count,crops_count//4))
# 输出结果
print("求解状态:", LpStatus[cultivation_model.status])
for year in range(duration):
    for i in range(fields_count):
        for j in range(crops_count): 
            if value(x_cultivation[(year, i, j)]) > 0:
                df_dst_5[year,i,j%18] = value(x_cultivation[(year, i, j)])
                print(f"第 {year + 1} 年 地块 {i + 1} 种植 {df[year].loc[j, '作物']} 面积: {value(x_cultivation[(year, i, j)])} 亩")


# ## 24-30每年总利润统计

# In[15]:


#表示以上五个作物类别所占种植面积&七年的总利润
profit_2 = [18290507.81351242,930032.0797854386,7090650.322774629,10531798.73119229,18856415.743954714]
area_2 = [1092, 42, 67, 9.6, 81.4]
# 定义作物类别的标签，为表示方便，此处用'类别1'等代替。
crop_categories = ['类别1', '类别2', '类别3', '类别4', '类别5']


# In[16]:


# 绘制每年所有作物的总利润统计图
# 定义颜色
colors = ['#d8bfd8','#66cdaa','#98fb98','#90ee90','#add8e6','#b0c4de']
# 绘制饼图
plt.figure(figsize=(10, 6))
plt.pie(profit_2, labels=crop_categories, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('作物类别七年总利润')
plt.axis('equal')
plt.show()


# In[17]:


# 绘制作物类别所占种植面积的饼图
plt.figure(figsize=(10, 6))
plt.pie(area_2, labels=crop_categories, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('作物类别所占种植面积')
plt.axis('equal')
plt.show()


# ## 调整后的利润如下

# In[19]:


profit_3 = [27385247.28428268,1284339.0298090328,5648323.953177902,13797066.81244492,14654627.156877685]
# 绘制饼图
plt.figure(figsize=(10, 6))
plt.pie(profit_3, labels=crop_categories, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('调整后作物类别七年总利润')
plt.axis('equal')
plt.show()


# In[21]:


# 计算两个列表的总和
sum_profit_2 = sum(profit_2)
sum_profit_3 = sum(profit_3)

# 计算profit_3相较于profit_2的增长量/增长率
growth = sum_profit_3 - sum_profit_2
rate=growth/sum_profit_2

# 打印结果
print("Total profit for profit_2:", sum_profit_2)
print("Total profit for profit_3:", sum_profit_3)
print("Growth of profit_3 compared to profit_2:", growth)
print("Growth Rate of profit_3 compared to profit_2:", rate)


# In[ ]:





# In[ ]:




