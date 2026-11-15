#!/usr/bin/env python
# coding: utf-8

# In[25]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value


# ## 平旱地、梯田和水浇地部分

# In[26]:


duration = 7
crops_count = 15
fields_count = 26
info = {
    'crops': [f'crop_{i+1}' for i in range(crops_count)],
    'yield_per_mu': [400,500,400,350,415,800,1000,400,630,525,110,3000,2200,420,525],  
    'cultivation_cost': [400, 400, 350,350,350,450,500,360,400,360,350,1000,2000,400,350], 
    'selling_price': [3.25, 7.5, 8.25, 7, 6.75, 3.5, 3, 6.75, 6, 7.5, 40, 1.5, 3.25, 5.5, 3.5], 
    'expected_sales': [57000, 21850, 22400, 33040, 9875, 170840, 132750, 71400, 3000,12500,1500,35100,36000,14000,10000],  
}

yield_matrix = np.array([
    [400,500,400,350,415,800,1000,400,630,525,110,3000,2200,420,525],
    [380,475,380,330,395,760,950,380,600,500,105,2850,2100,400,500],
    [360,450,360,315,375,720,900,360,570,475,100,2700,2000,380,475]
    ])

restricted_indices = [6,7,7,1,4,8,6,2,3,4,5,8,6,8,9,10,1,7,14,15,11,12,1,13,6,3]
area_limits = [80,55,35,72,68,55,60,46,40,28,25,86,55,44,50,25,60,45,35,20,15,13,15,18,27,20]
index_list = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2]

# 创建 DataFrame
data_frame = pd.DataFrame(info)

# 创建线性规划模型
lp_model = LpProblem("Optimal_Crop_Planting", LpMaximize)

# 决策变量
decision_vars = {}
for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            # 决策变量 decision_vars[yr][fld][crp] 表示在第 yr 年地块 fld 上种植作物 crp 的面积
            decision_vars[(yr, fld, crp)] = LpVariable(f"area_{yr}_{fld}_{crp}", lowBound=0.0)

# 目标函数：最大化总利润
total_revenue = 0

for yr in range(duration):
    for crp in range(crops_count):
        # 计算某种作物在所有地块上的总产量
        total_yield = lpSum( data_frame.loc[crp, 'yield_per_mu'] * decision_vars[(yr, fld, crp)] for fld in range(fields_count))
        
        expected_volume = data_frame.loc[crp, 'expected_sales']  # 预期销售量
        
        # 判断总产量与预期销售量的关系
        if total_yield <= expected_volume:
            # 正常售卖
            revenue = total_yield * data_frame.loc[crp, 'selling_price'] - lpSum(data_frame.loc[crp, 'cultivation_cost'] * decision_vars[(yr, fld, crp)] for fld in range(fields_count))
        else:
            # 超出部分售价为50%
            normal_volume = expected_volume
            excess_volume = total_yield - expected_volume
            revenue = (normal_volume * data_frame.loc[crp, 'selling_price'] + 
                      excess_volume * (0.5 * data_frame.loc[crp, 'selling_price'])) - lpSum(data_frame.loc[crp, 'cultivation_cost'] * decision_vars[(yr, fld, crp)] for fld in range(fields_count))
        
        # 累加利润
        total_revenue += revenue

# 添加目标函数
lp_model += total_revenue, "Total_Revenue"

# 添加作物种类约束
for fld in range(fields_count):
    lp_model += lpSum(decision_vars[(0, fld, crp)] for crp in range(crops_count) if crp == restricted_indices[fld]) == 0, f"Crop_Constraint_{fld}"
    
# 添加销售量限制
for yr in range(duration):
    for crp in range(crops_count):
        yield_total = 0
        for fld in range(fields_count):
            yield_total += yield_matrix[index_list[fld],crp] * decision_vars[(yr, fld, crp)]  # 计算生产量
        expected_volume = data_frame.loc[crp, 'expected_sales']  # 预期销售量
            # 下限约束：生产量 >= 预期销售量 * 0.8
        lp_model += yield_total >= 0.8 * expected_volume, f"Min_Volume_{yr}_{crp}"
            # 上限约束：生产量 <= 预期销售量 * 1.2
        lp_model += yield_total <= 1.2 * expected_volume, f"Max_Volume_{yr}_{crp}"

# 每块地的面积限制
for yr in range(duration):
    for fld in range(fields_count):
        lp_model += lpSum(decision_vars[(yr, fld, crp)] for crp in range(crops_count)) <= area_limits[fld], f"Area_Limit_{yr}_{fld}"

# 创建布尔变量，指示某年某地块是否种植某种作物
binary_vars = {}
for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            binary_vars[(yr, fld, crp)] = LpVariable(f"binary_{yr}_{fld}_{crp}", cat='Binary')

# 更新决策变量与布尔变量之间的关系
for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            lp_model += decision_vars[(yr, fld, crp)] <= area_limits[fld] * binary_vars[(yr, fld, crp)], f"Link_{yr}_{fld}_{crp}"

# 添加约束条件：如果去年种植了某种作物，下一年就不能种植
for yr in range(1, duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            lp_model += binary_vars[(yr-1, fld, crp)] + binary_vars[(yr, fld, crp)] <= 1, f"Crop_Rotation_{yr}_{fld}_{crp}"
        
# 添加约束条件：确保每个地块在过去三年内至少种植一次豆类作物      
bean_crop_indices = range(5)  # 假设前五种作物是豆类作物
for yr in range(3, duration):  # 从第3年开始
    for fld in range(fields_count):
        lp_model += lpSum(decision_vars[(y, fld, crp)] for y in range(yr - 3, yr) for crp in bean_crop_indices) >= 1, f"Bean_Crop_Constraint_{yr}_{fld}"

# 求解模型
lp_model.solve()

result_matrix = np.zeros((duration,fields_count,crops_count))
# 输出结果
areas_lp=0 #定义种植面积
#获取目标函数的值
objective_value = value(lp_model.objective)
# 输出目标函数的值
print("目标函数的值为:", objective_value)
print("Solving status:", LpStatus[lp_model.status])
for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            if value(decision_vars[(yr, fld, crp)]) > 0:
                result_matrix[yr,fld,crp] = value(decision_vars[(yr, fld, crp)])
                areas_lp+=value(decision_vars[(yr, fld, crp)])
                print(f"Year {yr + 1} Field {fld + 1} Planting {data_frame.loc[crp, 'crops']} Area: {value(decision_vars[(yr, fld, crp)])} mu")
print("这一部分总种植面积为:", areas_lp)


# ## 水浇地、普通大棚和智慧大棚部分
# ### 先处理水稻

# In[27]:


duration = 7
crops = 1
fields = 8
rice_data = {
    'crop': [f'rice_{i+16}' for i in range(crops)],
    'yield_per_mu': [500],  
    'cultivation_cost': [680], 
    'sell_price': [7],
    'expected_sales':[21000]
}

forbidden = [[20,36],[28,35],[21,35],[22,35],[17,36],[18,37],[16],[16]]
limits = [15,10,14,6,10,12,22,20]

rice_df = pd.DataFrame(rice_data)

# 创建线性规划模型
rice_model = LpProblem("Rice_Farming_Plan", LpMaximize)
# 决策变量
rice_vars = {}

for yr in range(duration):
    for fld in range(fields):
        for crp in range(crops):
            rice_vars[(yr, fld, crp)] = LpVariable(f"rice_var_{yr}_{fld}_{crp}", lowBound=0.0)

# 目标函数：最大化总利润
total_revenue = 0

for yr in range(duration):
    for crp in range(crops):
        # 计算某种作物在所有地块上的总产量
        total_yield = lpSum( rice_df.loc[crp, 'yield_per_mu'] * rice_vars[(yr, fld, crp)] for fld in range(fields))
        
        expected_volume = rice_df.loc[crp, 'expected_sales']  # 预期销售量
        
        # 判断总产量与预期销售量的关系
        if total_yield <= expected_volume:
            # 正常售卖
            revenue = total_yield * rice_df.loc[crp, 'sell_price'] - lpSum(rice_df.loc[crp, 'cultivation_cost'] * rice_vars[(yr, fld, crp)] for fld in range(fields))
        else:
            # 超出部分售价为50%
            normal_volume = expected_volume
            excess_volume = total_yield - expected_volume
            revenue = (normal_volume * rice_df.loc[crp, 'sell_price'] + 
                      excess_volume * (0.5 * rice_df.loc[crp, 'sell_price'])) - lpSum(rice_df.loc[crp, 'cultivation_cost'] * rice_vars[(yr, fld, crp)] for fld in range(fields))
        
        # 累加利润
        total_revenue += revenue

rice_model += total_revenue, "Total_Revenue"

# 添加销售量限制
for yr in range(duration):
    for crp in range(crops):
        yield_total = 0
        for fld in range(fields):
            yield_total += rice_df.loc[crp, 'yield_per_mu'] * rice_vars[(yr, fld, crp)]
        expected_volume = rice_df.loc[crp, 'expected_sales']
        rice_model += yield_total >= 0.8 * expected_volume, f"Min_Volume_{yr}_{crp}"
        rice_model += yield_total <= 1.2 * expected_volume, f"Max_Volume_{yr}_{crp}"

# 每块地的面积限制
for yr in range(duration):
    for fld in range(fields):
        rice_model += lpSum(rice_vars[(yr, fld, crp)] for crp in range(crops)) <= limits[fld], f"Area_Limit_{yr}_{fld}"

# 添加禁种作物约束
for fld in range(fields):
    for yr in range(duration):
        # 处理可能的多个禁种作物
        for forbidden_crop in forbidden[fld]:
            # 检查 forbidden_crop 是否是列表且有值
            if isinstance(forbidden_crop, list) and forbidden_crop:
                rice_model += lpSum(rice_vars[(yr, fld, crp)] for crp in forbidden_crop) == 0, f"Forbidden_Crop_{yr}_{fld}"

# 不能重复种植            
binary_vars = {}
for yr in range(duration):
    for fld in range(fields):
        for crp in range(crops):
            binary_vars[(yr, fld, crp)] = LpVariable(f"binary_var_{yr}_{fld}_{crp}", cat='Binary')

# 更新决策变量与布尔变量之间的关系
for yr in range(duration):
    for fld in range(fields):
        for crp in range(crops):
            rice_model += rice_vars[(yr, fld, crp)] <= limits[fld] * binary_vars[(yr, fld, crp)], f"Link_{yr}_{fld}_{crp}"

# 添加约束条件：如果去年种植了某种作物，下一年就不能种植
for yr in range(1, duration):
    for fld in range(fields):
        for crp in range(crops):
            rice_model += binary_vars[(yr-1, fld, crp)] + binary_vars[(yr, fld, crp)] <= 1, f"Crop_Rotation_{yr}_{fld}_{crp}"
        

# 求解模型
rice_model.solve()

# 输出结果
areas_rice=0 #定义种植面积
#获取目标函数的值
objective_value = value(rice_model.objective)
# 输出目标函数的值
print("目标函数的值为:", objective_value)
print("Solving status:", LpStatus[rice_model.status])
for yr in range(duration):
    for fld in range(fields):
        for crp in range(crops):
            if value(rice_vars[(yr, fld, crp)]) > 0:
                areas_rice+=value(rice_vars[(yr, fld, crp)])
                print(f"Year {yr + 1} Field D{fld + 1} Planting {rice_df.loc[crp, 'crop']} Area: {value(rice_vars[(yr, fld, crp)])} mu")
print("这一部分总种植面积为:", areas_rice)
used_land_areas = np.zeros((duration,fields))            
for yr in range(duration):
    for fld in range(fields):
        for crp in range(crops):
            if value(rice_vars[(yr, fld, crp)]) > 0:
                used_land_areas[yr,fld] = value(rice_vars[(yr, fld, crp)])


# ## 水浇地第二季部分(大白菜, 白萝卜, 红萝卜)

# In[28]:


duration = 7
crops_count = 3
fields_count = 8
irrigated_data = {
    'crop': [f'crop_{i+35}' for i in range(crops_count)],
    'yield_per_mu': [5000, 4000, 3000],
    'cultivation_cost': [2000, 500, 500],
    'sell_price': [2.5, 2.5, 3.25],
    'expected_sales': [150000, 100000, 36000]
}

forbidden_crops = [[20, 36], [28, 35], [21, 35], [22, 35], [17, 36], [18, 37], [16], [16]]
field_limits = [15, 10, 14, 6, 10, 12, 22, 20]

irrigated_df = pd.DataFrame(irrigated_data)

# 创建线性规划模型
irrigated_model = LpProblem("Irrigated_Farming_Plan", LpMaximize)
# 决策变量
irrigated_vars = {}

for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            irrigated_vars[(yr, fld, crp)] = LpVariable(f"irrigated_var_{yr}_{fld}_{crp}", lowBound=0.0)

# 目标函数：最大化总利润
total_revenue = 0

for yr in range(duration):
    for crp in range(crops_count):
        # 计算某种作物在所有地块上的总产量
        total_yield = lpSum(irrigated_df.loc[crp, 'yield_per_mu'] * irrigated_vars[(yr, fld, crp)] for fld in range(fields_count))
        
        expected_volume = irrigated_df.loc[crp, 'expected_sales']  # 预期销售量
        
        # 判断总产量与预期销售量的关系
        if total_yield <= expected_volume:
            # 正常售卖
            revenue = total_yield * irrigated_df.loc[crp, 'sell_price'] - lpSum(irrigated_df.loc[crp, 'cultivation_cost'] * irrigated_vars[(yr, fld, crp)] for fld in range(fields_count))
        else:
            # 超出部分售价为50%
            normal_volume = expected_volume
            excess_volume = total_yield - expected_volume
            revenue = (normal_volume * irrigated_df.loc[crp, 'sell_price'] + 
                      excess_volume * (0.5 * irrigated_df.loc[crp, 'sell_price'])) - lpSum(irrigated_df.loc[crp, 'cultivation_cost'] * irrigated_vars[(yr, fld, crp)] for fld in range(fields_count))
        
        # 累加利润
        total_revenue += revenue

irrigated_model += total_revenue, "Total_Revenue"

# 添加销售量限制
for yr in range(duration):
    for crp in range(crops_count):
        production = 0
        for fld in range(fields_count):
            production += irrigated_df.loc[crp, 'yield_per_mu'] * irrigated_vars[(yr, fld, crp)]
        expected_volume = irrigated_df.loc[crp, 'expected_sales']
        irrigated_model += production >= 0.8 * expected_volume, f"Min_Volume_{yr}_{crp}"
        irrigated_model += production <= 1.2 * expected_volume, f"Max_Volume_{yr}_{crp}"

# 每块地的面积限制
for yr in range(duration):
    for fld in range(fields_count):
        irrigated_model += lpSum(irrigated_vars[(yr, fld, crp)] for crp in range(crops_count)) <= field_limits[fld] - used_land_areas[yr, fld], f"Area_Limit_{yr}_{fld}"

# 添加禁种作物约束
for fld in range(fields_count):
    for yr in range(duration):
        # 处理可能的多个禁种作物
        for forbidden_crop in forbidden_crops[fld]:
            # 检查 forbidden_crop 是否是列表且有值
            if isinstance(forbidden_crop, list) and forbidden_crop:
                irrigated_model += lpSum(irrigated_vars[(yr, fld, crp)] for crp in forbidden_crop) == 0, f"Forbidden_Crop_{yr}_{fld}"

# 不能重复种植            
binary_vars = {}
for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            binary_vars[(yr, fld, crp)] = LpVariable(f"binary_var_{yr}_{fld}_{crp}", cat='Binary')

# 更新决策变量与布尔变量之间的关系
for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            irrigated_model += irrigated_vars[(yr, fld, crp)] <= field_limits[fld] * binary_vars[(yr, fld, crp)], f"Link_{yr}_{fld}_{crp}"

# 添加约束条件：如果去年种植了某种作物，下一年就不能种植
for yr in range(1, duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            irrigated_model += binary_vars[(yr-1, fld, crp)] + binary_vars[(yr, fld, crp)] <= 1, f"Crop_Rotation_{yr}_{fld}_{crp}"
        

# 求解模型
irrigated_model.solve()
result_matrix = np.zeros((duration, fields_count, crops_count))
# 输出结果
areas_irrigated=0 #定义种植面积
print("Solving status:", LpStatus[irrigated_model.status])
#获取目标函数的值
objective_value = value(irrigated_model.objective)
# 输出目标函数的值
print("目标函数的值为:", objective_value)
for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            if value(irrigated_vars[(yr, fld, crp)]) > 0:
                result_matrix[yr, fld, crp] = value(irrigated_vars[(yr, fld, crp)])
                areas_irrigated+=value(irrigated_vars[(yr, fld, crp)])
                print(f"Year {yr + 1} Field D{fld + 1} Planting {irrigated_df.loc[crp, 'crop']} Area: {value(irrigated_vars[(yr, fld, crp)])} mu")
print("这一部分总种植面积为:", areas_irrigated)  


# ## 普通大棚部分（第二季）
# ### 即榆黄菇,香菇,白灵菇,羊肚菌

# In[29]:


duration = 7
crops_count = 4
fields_count = 16
greenhouse_data = {
    'crop': [f'crop_{i+38}' for i in range(crops_count)],
    'yield_per_mu': [5000, 4000, 10000, 1000],
    'cultivation_cost': [3000, 2000, 10000, 10000],
    'sell_price': [57.5, 19, 16, 100],
    'expected_sales': [9000, 7200, 18000, 4200]
}

forbidden_crops = [[18,38],[24,38],[25,38],[26,39],[28,39],[27,39],[19,40],[19,40],[18,40],[17,41],[17,41],[22,41],[21,41],[29,41],[30,27,41],[31,41]]
field_sizes = [0.6] * fields_count

greenhouse_df = pd.DataFrame(greenhouse_data)

# 创建线性规划模型
greenhouse_model = LpProblem("Greenhouse_Farming_Plan", LpMaximize)
# 决策变量
greenhouse_vars = {}

for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            greenhouse_vars[(yr, fld, crp)] = LpVariable(f"greenhouse_var_{yr}_{fld}_{crp}", lowBound=0.0)

# 目标函数：最大化总利润
total_revenue = 0

for yr in range(duration):
    for crp in range(crops_count):
        # 计算某种作物在所有地块上的总产量
        total_yield = lpSum(greenhouse_df.loc[crp, 'yield_per_mu'] * greenhouse_vars[(yr, fld, crp)] for fld in range(fields_count))
        
        expected_volume = greenhouse_df.loc[crp, 'expected_sales']  # 预期销售量
        
        # 判断总产量与预期销售量的关系
        if total_yield <= expected_volume:
            # 正常售卖
            revenue = total_yield * greenhouse_df.loc[crp, 'sell_price'] - lpSum(greenhouse_df.loc[crp, 'cultivation_cost'] * greenhouse_vars[(yr, fld, crp)] for fld in range(fields_count))
        else:
            # 超出部分售价为50%
            normal_volume = expected_volume
            excess_volume = total_yield - expected_volume
            revenue = (normal_volume * greenhouse_df.loc[crp, 'sell_price'] + 
                      excess_volume * (0.5 * greenhouse_df.loc[crp, 'sell_price'])) - lpSum(greenhouse_df.loc[crp, 'cultivation_cost'] * greenhouse_vars[(yr, fld, crp)] for fld in range(fields_count))
        
        # 累加利润
        total_revenue += revenue

greenhouse_model += total_revenue, "Total_Revenue"

# 添加销售量限制
for yr in range(duration):
    for crp in range(crops_count):
        production = 0
        for fld in range(fields_count):
            production += greenhouse_df.loc[crp, 'yield_per_mu'] * greenhouse_vars[(yr, fld, crp)]
        expected_volume = greenhouse_df.loc[crp, 'expected_sales']
        greenhouse_model += production >= 0.8 * expected_volume, f"Min_Volume_{yr}_{crp}"
        greenhouse_model += production <= 1.2 * expected_volume, f"Max_Volume_{yr}_{crp}"

# 每块地的面积限制
for yr in range(duration):
    for fld in range(fields_count):
        greenhouse_model += lpSum(greenhouse_vars[(yr, fld, crp)] for crp in range(crops_count)) <= field_sizes[fld], f"Area_Limit_{yr}_{fld}"

# 添加禁种作物约束
for i in range(fields_count):
    for year in range(duration):
        # 处理可能的多个禁种作物
        for forbidden_crop in forbidden_crops[i]:
            # 检查 forbidden_crop 是否是列表且有值
            if isinstance(forbidden_crop, list) and forbidden_crop:
                greenhouse_model += lpSum(greenhouse_vars[(year, i, j)] for j in forbidden_crop) == 0, f"Forbidden_Crop_{year}_{i}"
                
# 不能重复种植            
binary_vars = {}
for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            binary_vars[(yr, fld, crp)] = LpVariable(f"binary_var_{yr}_{fld}_{crp}", cat='Binary')

# 更新决策变量与布尔变量之间的关系
for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            greenhouse_model += greenhouse_vars[(yr, fld, crp)] <= field_sizes[fld] * binary_vars[(yr, fld, crp)], f"Link_{yr}_{fld}_{crp}"

# 添加约束条件：如果去年种植了某种作物，下一年就不能种植
for yr in range(1, duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            greenhouse_model += binary_vars[(yr-1, fld, crp)] + binary_vars[(yr, fld, crp)] <= 1, f"Crop_Rotation_{yr}_{fld}_{crp}"
        

# 求解模型
greenhouse_model.solve()
result_matrix = np.zeros((duration, fields_count, crops_count))
# 输出结果
areas_greenhouse=0 #定义种植面积
#获取目标函数的值
objective_value = value(greenhouse_model.objective)
# 输出目标函数的值
print("目标函数的值为:", objective_value)
print("Solving status:", LpStatus[greenhouse_model.status])
for yr in range(duration):
    for fld in range(fields_count):
        for crp in range(crops_count):
            if value(greenhouse_vars[(yr, fld, crp)]) > 0:
                result_matrix[yr, fld, crp] = value(greenhouse_vars[(yr, fld, crp)])
                areas_greenhouse+=value(greenhouse_vars[(yr, fld, crp)])
                print(f"Year {yr + 1} Second Season Field E{fld + 1} Planting {greenhouse_df.loc[crp, 'crop']} Area: {value(greenhouse_vars[(yr, fld, crp)])} mu")
print("这一部分总种植面积为:", areas_greenhouse)


# ## 其他所有部分（水浇地第一季，普通大棚第一季，智慧大棚第一季，智慧大棚第二季）
# ### 该部分均计算两季作物

# In[30]:


duration = 7
crops_count = 18 * 4
fields_count = 8 + 16 + 4 + 4
cultivation_data = {
    'crop': [f'crop_{i+17}' for i in range(18)] * 4,
    'yield_per_mu': [
        3000, 2000, 3000, 2000, 2400, 6400, 2700, 2400, 3300, 3700, 4100, 3200, 12000, 4100, 1600, 10000, 5000, 5500,
        3600, 2400, 3600, 2400, 3000, 8000, 3300, 3000, 4000, 4500, 5000, 4000, 15000, 5000, 2000, 12000, 6000, 6600,
        3600, 2400, 3600, 2400, 3000, 8000, 3300, 3000, 4000, 4500, 5000, 4000, 15000, 5000, 2000, 12000, 6000, 6600,
        3200, 2200, 3200, 2200, 2700, 7200, 3000, 2700, 3600, 4100, 4500, 3600, 13500, 4500, 1800, 11000, 5400, 6000,
    ],  
    'cultivation_cost': [
        2000, 1000, 2000, 2000, 2000, 2000, 2300, 1600, 2400, 2900, 1600, 1600, 2900, 1600, 1000, 4100, 2000, 900,
        2400, 1200, 2400, 2400, 2400, 2400, 2700, 2000, 3000, 3500, 2000, 2000, 3500, 2000, 1200, 5000, 2500, 1100,
        2400, 1200, 2400, 2400, 2400, 2400, 2700, 2000, 3000, 3500, 2000, 2000, 3500, 2000, 1200, 5000, 2500, 1100,
        2640, 1320, 2640, 2640, 2640, 2640, 3000, 2200, 3300, 3850, 2200, 2200, 3850, 2200, 1300, 5500, 2750, 1200 
    ], 
    'selling_price': [
        8, 6.75, 6.5, 3.75, 6.25, 5.5, 5.75, 5.25, 5.5, 6.5, 5, 5.75, 7, 5.25, 7.25, 4.5, 4.5, 4,
        8, 6.75, 6.5, 3.75, 6.25, 5.5, 5.75, 5.25, 5.5, 6.5, 5, 5.75, 7, 5.25, 7.25, 4.5, 4.5, 4,
        8, 6.75, 6.5, 3.75, 6.25, 5.5, 5.75, 5.25, 5.5, 6.5, 5, 5.75, 7, 5.25, 7.25, 4.5, 4.5, 4,
        9.6, 8.1, 7.8, 4.5, 7.5, 6.6, 6.9, 6.8, 6.6, 7.8, 6, 6.9, 8.4, 6.3, 8.7, 5.4, 5.4, 4.8
    ]
}

cultivation_df = pd.DataFrame(cultivation_data)
forbidden_crops = [
    [20, 36], [28, 35], [21, 35], [22, 35], [17, 36], [18, 37], [16], [16],
    [18, 38], [24, 38], [25, 38], [26, 39], [28, 39], [27, 39], [19, 40], [19, 40], [18, 40], [17, 41], [17, 41], [22, 41], [21, 41], [29, 41], [30, 27, 41], [31, 41],
    [32, 33], [25, 26], [17], [19], [24, 21], [22, 29], [28, 30], [34, 23]
]
field_sizes = [15, 10, 14, 6, 10, 12, 22, 20] + [0.6] * 16 + [0.6] * 4 + [0.6] * 4
sales_amount = [36240, 26880, 6240, 30000, 36210, 45360, 900, 2610, 3480, 3930, 4500, 35480, 13050, 2850, 1200, 3300, 1620, 1800]
additional_land_use = np.zeros((duration, fields_count))
additional_land_use[:duration, :8] = used_land_areas


# In[31]:


# 创建线性规划模型
cultivation_model = LpProblem("Diverse_Cultivation_Plan", LpMaximize)
# 决策变量
cultivation_vars = {}

for year in range(duration):
    for field in range(fields_count):
        for crop in range(crops_count):
            cultivation_vars[(year, field, crop)] = LpVariable(f"cultivation_var_{year}_{field}_{crop}", lowBound=0.0)

# 目标函数：最大化总利润
total_revenue = 0

for year in range(duration):
    for field in range(fields_count):
        for crop in range(crops_count):
            production = cultivation_df.loc[crop, 'yield_per_mu'] * cultivation_vars[(year, field, crop)]
            expected_sales = sales_amount[crop % 18]
            
            if production <= expected_sales:
                profit = production * cultivation_df.loc[crop, 'selling_price'] - cultivation_df.loc[crop, 'cultivation_cost'] * cultivation_vars[(year, field, crop)]
            else:
                normal_sales = expected_sales
                excess_sales = production - expected_sales
                profit = (normal_sales * cultivation_df.loc[crop, 'selling_price'] +
                          excess_sales * (0.5 * cultivation_df.loc[crop, 'selling_price'])) - cultivation_df.loc[crop, 'cultivation_cost'] * cultivation_vars[(year, field, crop)]

            total_revenue += profit  # 累加利润

# 添加目标函数
cultivation_model += total_revenue, "Total_Revenue"

# 添加销售量限制
for year in range(duration):
    for crop in range(18):
        production = 0
        for field in range(fields_count):
            production += cultivation_df.loc[crop, 'yield_per_mu'] * cultivation_vars[(year, field, crop)]
            production += cultivation_df.loc[crop + 18 * 1, 'yield_per_mu'] * cultivation_vars[(year, field, crop + 18 * 1)]
            production += cultivation_df.loc[crop + 18 * 2, 'yield_per_mu'] * cultivation_vars[(year, field, crop + 18 * 2)]
            production += cultivation_df.loc[crop + 18 * 3, 'yield_per_mu'] * cultivation_vars[(year, field, crop + 18 * 3)]
        expected_sales = sales_amount[crop]
        cultivation_model += production >= 0.8 * expected_sales, f"Min_Sales_{year}_{crop}"
        cultivation_model += production <= 1.2 * expected_sales, f"Max_Sales_{year}_{crop}"

#添加每块地的面积约束
for year in range(duration):
    for field in range(fields_count):
        cultivation_model += lpSum(cultivation_vars[(year, field, crop)] for crop in range(crops_count)) <= field_sizes[field] - additional_land_use[year, field], f"Area_Limit_{year}_{field}"

#添加禁种作物约束
for field in range(fields_count):
    for year in range(duration):
        # 处理可能的多个禁种作物
        for forbidden_crop in forbidden_crops[field]:
            # 检查 forbidden_crop 是否是列表且有值
            if isinstance(forbidden_crop, list) and forbidden_crop:
                cultivation_model += lpSum(cultivation_vars[(year, field, crop)] for crop in forbidden_crop) == 0, f"Forbidden_Crop_{year}_{field}"

#添加约束条件：确保每个地块在过去三年内至少种植一次豆类作物
bean_crops = range(3)
for year in range(3, duration):  # 从第 3 年开始
    for field in range(fields_count):
        cultivation_model += lpSum(cultivation_vars[(y, field, crop)] for y in range(year - 3, year) for crop in bean_crops) >= 1, f"Bean_Crop_Constraint_{year}_{field}"

for year in range(1,duration):  # 从第 2 年开始
    for i in range(fields_count):
        for j in range(crops_count):
            # 如果某年地块 i 种植了作物 j，则下一年该地块不能再种植作物 j
            cultivation_model += cultivation_vars[(year - 1, i, j)] + cultivation_vars[(year, i, j)] <= field_sizes[i], f"rotation_constraint_{year}_{i}_{j}"
            
#求解模型
cultivation_model.solve()
result_matrix = np.zeros((duration, fields_count, crops_count // 4))

#输出结果
areas_cultivation=0 #定义种植面积
#获取目标函数的值
objective_value = value(cultivation_model.objective)
# 输出目标函数的值
print("目标函数的值为:", objective_value)
print("Solving status:", LpStatus[cultivation_model.status])
for year in range(duration):
    for field in range(fields_count):
        for crop in range(crops_count):
            if value(cultivation_vars[(year, field, crop)]) > 0:
                result_matrix[year, field, crop % 18] = value(cultivation_vars[(year, field, crop)])
                areas_cultivation+=value(cultivation_vars[(year, field, crop)])
                print(f"Year {year + 1} Field {field + 1} Planting {cultivation_df.loc[crop, 'crop']} Area: {value(cultivation_vars[(year, field, crop)])} mu")
print("这一部分总种植面积为:", areas_cultivation)


# In[45]:


#表示以上五个作物类别所占种植面积&七年的总利润
total_profits = np.array([18454260.948635638, 994896.0, 4083100.0, 9734760.0, 11847564.792054156])
total_areas=[7644.000000564998,352.79999999999995,410.20000000000005,67.20000000000005,510.9999990620006]
# 定义作物类别的标签，为表示方便，此处用'类别1'等代替。
crop_categories = ['类别1', '类别2', '类别3', '类别4', '类别5']
# 定义颜色
colors = ['#d8bfd8','#66cdaa','#98fb98','#90ee90','#add8e6','#b0c4de']
# 绘制饼图
plt.figure(figsize=(10, 6))
plt.pie(total_profits, labels=crop_categories, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('作物类别七年总利润')
plt.axis('equal')
plt.show()


# In[47]:


# 绘制作物类别所占种植面积的饼图
plt.figure(figsize=(10, 6))
plt.pie(total_areas, labels=crop_categories, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('作物类别所占种植面积')
plt.axis('equal')
plt.show()


# In[ ]:




