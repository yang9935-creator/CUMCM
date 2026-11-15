#!/usr/bin/env python
# coding: utf-8

# ## 一、导入必要的库

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#options
warnings.filterwarnings('ignore')


# ## 二、读取数据并查看

# In[2]:


# 加载数据
file1_path = '附件1(原始数据).xlsx'
file2_path = '附件2(原始数据).xlsx'

data_land = pd.read_excel(file1_path, sheet_name=0) #乡村的现有耕地
data_crops = pd.read_excel(file1_path, sheet_name=1) #乡村种植的农作物
data_plant_2023 = pd.read_excel(file2_path, sheet_name=0) #2023年的农作物种植情况
data_2023 =pd.read_excel(file2_path, sheet_name=1) #2023年统计的相关数据


# In[3]:


data_land.head()


# In[4]:


data_crops.head()


# In[5]:


data_plant_2023.tail()


# In[6]:


data_2023.head()


# ## 三、数据清洗

# ### 3.1缺失值填充

# In[7]:


#根据原始数据对data_crops和data_plant_2023进行缺失值的向下填充
data_crops.fillna(method='ffill',inplace=True)


# In[8]:


data_plant_2023.fillna(method='ffill',inplace=True)


# 再次查看数据，此时数据缺失值已被填充

# In[9]:


data_crops.head()


# In[10]:


data_plant_2023.tail()


# ### 3.2 重复值检测

# In[11]:


data_land.duplicated().sum()


# In[12]:


data_crops.duplicated().sum()


# In[13]:


data_plant_2023.duplicated().sum()


# In[14]:


data_2023.duplicated().sum()


# In[15]:


data_crops.duplicated()


# In[16]:


data_crops.tail()


# In[17]:


#发现查看后分析此处有重复值是由于闲钱缺失值的填充，直接删除即可
data_crops.drop_duplicates(inplace=True)
data_crops.index = range(data_crops.shape[0])#恢复索引


# ### 3.3 异常值检测

# In[18]:


# 针对数值型变量进行异常值检测
# 将所有DataFrame的数值列合并到一个DataFrame中
all_data = pd.concat([data_land.select_dtypes(include=['float', 'int']),
                       data_crops.select_dtypes(include=['float', 'int']),
                       data_plant_2023.select_dtypes(include=['float', 'int']),
                       data_2023.select_dtypes(include=['float', 'int'])])

all_data.describe()


# 初步判断无明显异常值，后续通过分布图进一步确定。

# In[19]:


# 保存计算结果
data_land.to_csv('乡村的现有耕地.csv', index=False)
data_crops.to_csv('乡村种植的农作物.csv', index=False)
data_plant_2023.to_csv('2023年的农作物种植情况.csv', index=False)
data_2023.to_csv('2023年统计的相关数据.csv', index=False)


# ## 四、计算产量

# In[20]:


import pandas as pd
# 合并耕地信息和种植情况
data_combined = pd.merge(data_land, data_plant_2023, how='left', left_on='地块名称', right_on='种植地块')

# 接着，合并农作物种植情况和2023年的统计数据
merged_data_2023 = pd.merge(data_combined, data_2023, how='left', on=['作物编号', '地块类型'])


# In[21]:


merged_data_2023


# In[22]:


# 删除'序号'、‘说明’和’种植地块’（重复）列
merged_data_2023 = merged_data_2023.drop(columns=['序号'])
merged_data_2023 = merged_data_2023.drop(columns=['说明 '])
merged_data_2023 = merged_data_2023.drop(columns=['种植地块'])
# 决定保留哪一列，这里我们保留data_plant_2023中的'种植季次'和'作物名称'，_2023中的对应列
merged_data_2023 = merged_data_2023.drop(columns=['种植季次_y', '作物名称_y'])
merged_data_2023.rename(columns=lambda x: x[:-2] if x.endswith('_x') else x, inplace=True)


# In[23]:


# 显示合并后的前几行数据
merged_data_2023.head()


# In[24]:


# 计算产量
merged_data_2023['产量/斤']= merged_data_2023['种植面积/亩']* merged_data_2023['亩产量/斤']


# In[25]:


merged_data_2023.head()


# ## 五、数据类型转换

# In[26]:


# 拆分销售单价区间
def split_range(range_str):
    low, high = range_str.split('-')
    return float(low), float(high)

# 应用拆分函数
merged_data_2023[['销售单价最小值/(元/斤)', '销售单价最大值/(元/斤)']] = merged_data_2023['销售单价/(元/斤)'].apply(lambda x: pd.Series(split_range(x)))

# 检查结果
print(merged_data_2023[['销售单价/(元/斤)', '销售单价最小值/(元/斤)', '销售单价最大值/(元/斤)']])


# In[27]:


# 保存计算结果
merged_data_2023.to_csv('产量计算后的聚合数据.csv', index=False)


# ## 六、数据探索性分析

# ### 6.1 产量相关

# #### 6.1.1 产量分布直方图

# In[28]:


plt.rcParams['font.sans-serif']=['Simhei'] #用于正常显示中文


# In[29]:


# 绘制产量的直方图
plt.figure(figsize=(10, 6))
sns.histplot(merged_data_2023['产量/斤'], bins=30, kde=True)
plt.title('产量分布图')
plt.xlabel('产量/斤')
plt.ylabel('频数')
plt.show()


# #### 6.1.2 不同地块类型的作物总产量&平均亩产量&平均种植成本对比

# In[30]:


# 计算每个地块类型的总产量
total_yield_by_land_type = merged_data_2023.groupby('地块类型')['产量/斤'].sum()

# 计算每个地块类型的平均亩产量
average_yield_by_land_type = merged_data_2023.groupby('地块类型')['亩产量/斤'].mean()

# 计算每个地块类型的平均种植成本
average_cost_by_land_type = merged_data_2023.groupby('地块类型')['种植成本/(元/亩)'].mean()

# 准备数据用于绘图
land_types = total_yield_by_land_type.index
total_yield = total_yield_by_land_type.values
average_yield = average_yield_by_land_type.values
average_cost = average_cost_by_land_type.values

# 设置柱状图的宽度
barWidth = 0.25

# 设置柱状图的位置
r1 = range(len(land_types))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# 绘制柱状图
plt.figure(figsize=(12, 8))

# 绘制总产量柱状图
plt.bar(r1, total_yield, color='b', width=barWidth, edgecolor='grey', label='总产量 (斤)')

# 绘制平均亩产量柱状图
plt.bar(r2, average_yield, color='r', width=barWidth, edgecolor='grey', label='平均亩产量 (斤/亩)')

# 绘制平均种植成本柱状图
plt.bar(r3, average_cost, color='g', width=barWidth, edgecolor='grey', label='平均种植成本 (元/亩)')

# 添加图例
plt.legend()

# 添加网格线
plt.grid(True)

# 添加 x 轴的标签
plt.xlabel('地块类型', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(land_types))], land_types)

# 添加 y 轴的标签
plt.ylabel('数值', fontweight='bold')

# 添加标题
plt.title('不同地块类型的作物总产量、亩产量和种植成本对比')

# 展示图表
plt.show()


# 由于总产量相较于其他两个指标过大，影响分析，因此单独将亩产量与种植成本列出比较，并计算它们的标准差，用误差线表示标准差。

# In[31]:


# 计算每个地块类型的平均亩产量
average_yield_by_land_type = merged_data_2023.groupby('地块类型')['亩产量/斤'].mean()

# 计算每个地块类型的平均种植成本
average_cost_by_land_type = merged_data_2023.groupby('地块类型')['种植成本/(元/亩)'].mean()

# 计算标准差以用于误差线
std_yield_by_land_type = merged_data_2023.groupby('地块类型')['亩产量/斤'].std()
std_cost_by_land_type = merged_data_2023.groupby('地块类型')['种植成本/(元/亩)'].std()

# 准备数据用于绘图
land_types = average_yield_by_land_type.index
average_yield = average_yield_by_land_type.values
average_cost = average_cost_by_land_type.values
std_yield = std_yield_by_land_type.values
std_cost = std_cost_by_land_type.values

# 设置柱状图的宽度
barWidth = 0.3

# 设置柱状图的位置
r1 = range(len(land_types))
r2 = [x + barWidth for x in r1]

# 绘制柱状图
plt.figure(figsize=(12, 8))

# 绘制平均亩产量柱状图
plt.bar(r1, average_yield, color='b', width=barWidth, edgecolor='grey', label='平均亩产量 (斤/亩)', yerr=std_yield, capsize=5)

# 绘制平均种植成本柱状图
plt.bar(r2, average_cost, color='r', width=barWidth, edgecolor='grey', label='平均种植成本 (元/亩)', yerr=std_cost, capsize=5)

# 添加图例
plt.legend()

# 添加网格线
plt.grid(True)

# 添加 x 轴的标签
plt.xlabel('地块类型', fontweight='bold')
plt.xticks([r + barWidth / 2 for r in range(len(land_types))], land_types)

# 添加 y 轴的标签
plt.ylabel('数值', fontweight='bold')

# 添加标题
plt.title('不同地块类型的作物平均亩产量和平均种植成本对比')

# 展示图表
plt.show()


# #### 6.1.3 不同类型作物总产量&平均亩产量&平均种植成本对比

# In[32]:


# 计算每个作物类型的总产量、平均亩产量和平均种植成本
summary_stats = merged_data_2023.groupby('作物类型').agg(
    Total_Yield=('产量/斤', 'sum'),
    Average_Yield=('亩产量/斤', 'mean'),
    Average_Cost=('种植成本/(元/亩)', 'mean')
).reset_index()

# 准备数据用于绘图
crop_types = summary_stats['作物类型']
total_yield = summary_stats['Total_Yield']
average_yield = summary_stats['Average_Yield']
average_cost = summary_stats['Average_Cost']

# 设置柱状图的宽度
barWidth = 0.25

# 设置柱状图的位置
r1 = range(len(crop_types))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# 绘制柱状图
plt.figure(figsize=(12, 8))

# 绘制总产量柱状图
plt.bar(r1, total_yield, color='#1f77b4', width=barWidth, edgecolor='grey', label='总产量 (斤)')

# 绘制平均亩产量柱状图
plt.bar(r2, average_yield, color='#ff7f0e', width=barWidth, edgecolor='grey', label='平均亩产量 (斤/亩)')

# 绘制平均种植成本柱状图
plt.bar(r3, average_cost, color='#2ca02c', width=barWidth, edgecolor='grey', label='平均种植成本 (元/亩)')

# 添加图例
plt.legend()

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 添加 x 轴的标签
plt.xlabel('作物类型', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(crop_types))], crop_types)

# 添加 y 轴的标签
plt.ylabel('数值', fontweight='bold')

# 添加标题
plt.title('不同类型作物的总产量、平均亩产量和平均种植成本对比')

# 展示图表
plt.show()


# 同理，由于总产量相较于其他两个指标过大，影响分析，因此单独将亩产量与种植成本列出比较，并计算它们的标准差，用误差线表示标准差。

# In[33]:


# 计算每个作物类型的平均亩产量和平均种植成本，以及它们的标准差
summary_stats = merged_data_2023.groupby('作物类型').agg(
    Average_Yield=('亩产量/斤', 'mean'),
    Average_Cost=('种植成本/(元/亩)', 'mean'),
    Std_Yield=('亩产量/斤', 'std'),
    Std_Cost=('种植成本/(元/亩)', 'std')
).reset_index()

# 准备数据用于绘图
crop_types = summary_stats['作物类型']
average_yield = summary_stats['Average_Yield']
average_cost = summary_stats['Average_Cost']
std_yield = summary_stats['Std_Yield']
std_cost = summary_stats['Std_Cost']

# 设置柱状图的宽度
barWidth = 0.3

# 设置柱状图的位置
r1 = range(len(crop_types))
r2 = [x + barWidth for x in r1]

# 绘制柱状图
plt.figure(figsize=(12, 8))

# 绘制平均亩产量柱状图
plt.bar(r1, average_yield, color='#1f77b4', yerr=std_yield, width=barWidth, edgecolor='grey', label='平均亩产量 (斤/亩)', capsize=5)

# 绘制平均种植成本柱状图
plt.bar(r2, average_cost, color='#8B0000', yerr=std_cost, width=barWidth, edgecolor='grey', label='平均种植成本 (元/亩)', capsize=5)

# 添加图例
plt.legend()

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 添加 x 轴的标签
plt.xlabel('作物类型', fontweight='bold')
plt.xticks([r + barWidth / 2 for r in range(len(crop_types))], crop_types)

# 添加 y 轴的标签
plt.ylabel('数值', fontweight='bold')

# 添加标题
plt.title('不同类型作物的平均亩产量与平均种植成本对比')

# 展示图表
plt.show()


# In[34]:


#计算每个种植季次的总产量、平均亩产量和平均种植成本
summary_stats = merged_data_2023.groupby('种植季次').agg(
   Total_Yield=('产量/斤', 'sum'),
   Average_Yield=('亩产量/斤', 'mean'),
   Average_Cost=('种植成本/(元/亩)', 'mean')
).reset_index()

# 准备数据用于绘图
labels = summary_stats['种植季次']
colors = ['lightblue', 'lightcoral', 'lightgreen']

# 绘制总产量饼图
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.pie(summary_stats['Total_Yield'], labels=labels,colors=colors,autopct='%1.1f%%', startangle=140)
plt.title('总产量分布')

# 绘制平均亩产量饼图
plt.subplot(1, 3, 2)
plt.pie(summary_stats['Average_Yield'], labels=labels,colors=colors,autopct='%1.1f%%', startangle=140)
plt.title('平均亩产量分布')

# 绘制平均种植成本饼图
plt.subplot(1, 3, 3)
plt.pie(summary_stats['Average_Cost'], labels=labels,colors=colors,autopct='%1.1f%%', startangle=140)
plt.title('平均种植成本分布')

# 展示图表
plt.tight_layout()
plt.show()


# ### 6.2 销售单价相关

# ### 6.2.1 不同作物类型的销售单价区间

# In[35]:


# 设置图形大小
plt.figure(figsize=(12, 8))

# 创建一个条形图，显示销售单价最大值
sns.barplot(x='作物类型', y='销售单价最大值/(元/斤)', data=merged_data_2023, color='lightblue', label='最大值')

# 在同一图中添加销售单价最小值的条形图
sns.barplot(x='作物类型', y='销售单价最小值/(元/斤)', data=merged_data_2023, color='pink', label='最小值', alpha=0.6)

plt.title('不同作物类型的销售单价区间')
plt.xlabel('作物类型')
plt.ylabel('销售单价 (元/斤)')
plt.legend()
plt.tight_layout()  # 调整布局
plt.show()


# ### 6.2.2 不同作物销售单价趋势

# In[36]:


plt.figure(figsize=(20, 10))
sns.lineplot(data=merged_data_2023, x='作物名称', y='销售单价最小值/(元/斤)', label='最低销售单价')
sns.lineplot(data=merged_data_2023, x='作物名称', y='销售单价最大值/(元/斤)', label='最高销售单价')
plt.title('作物销售价格趋势')
plt.xlabel('作物名称')
plt.xticks(rotation=45)  # 旋转x轴标签以更好地显示
plt.ylabel('销售价格（元/斤）')
plt.legend()
plt.show()


# ### 6.3 地块面积分布情况

# In[37]:


# 绘制
plt.figure(figsize=(10, 6))
sns.histplot(data_land['地块面积/亩'], bins=20, kde=True)
plt.title('地块面积分布图')
plt.xlabel('地块面积 (亩)')
plt.ylabel('频数')
plt.show()


# In[38]:


# 计算每个地块类型的总面积
total_area = data_land.groupby('地块类型')['地块面积/亩'].sum()

# 计算平均种植面积，这里假设每个地块类型的种植面积是均匀分布的
average_area = merged_data_2023.groupby('地块类型')['种植面积/亩'].mean()

# 定义颜色
colors = ['#66cdaa','#d8bfd8','#90ee90','#98fb98','#add8e6','#b0c4de']

# 设置图表大小
plt.figure(figsize=(14, 7))

# 绘制总面积饼图
plt.subplot(1, 2, 1)
wedges, texts, autotexts = plt.pie(total_area, startangle=10, colors=colors,
                                   autopct='%1.1f%%', pctdistance=0.85)
plt.title('不同类型地块的总面积分布')

# 绘制平均种植面积饼图
plt.subplot(1, 2, 2)
wedges, texts, autotexts = plt.pie(average_area, startangle=140, colors=colors,
                                   autopct='%1.1f%%', pctdistance=0.85)
plt.title('不同类型地块的平均种植面积分布')

# 使用图例代替标签
plt.subplot(1, 2, 1)
plt.legend(wedges, total_area.index, title="地块类型", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.subplot(1, 2, 2)
plt.legend(wedges, average_area.index, title="地块类型", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# 展示图表
plt.tight_layout()
plt.show()


# ### 6.4 数据检查
# 

# In[39]:


# 提取作物名称列
plants_2023 = data_plant_2023['作物名称'].drop_duplicates()  # 删除重复项
sales_2023 = data_2023['作物名称'].drop_duplicates()  # 删除重复项

# 将列表转换为集合
set_plants = set(plants_2023)
set_sales = set(sales_2023)

# 找出共同的作物名称
common_crops = set_plants.intersection(set_sales)

# 找出仅在种植数据中出现的作物名称
only_planted = set_plants.difference(set_sales)

# 找出仅在销售数据中出现的作物名称
only_sold = set_sales.difference(set_plants)

# 打印结果，nan表示无
print("共同种植和销售的作物:", common_crops)
print("仅种植未在销售数据中出现的作物:", only_planted)
print("未种植但在销售数据中出现的作物:", only_sold)


# ### 6.5 所有数值型变量的相关性分析（初步）

# In[40]:


merged_data_2023


# In[41]:


# 选择数值型变量
numeric_data = merged_data_2023.select_dtypes(include=['float64', 'int64'])


# In[42]:


# 计算相关性矩阵
corr_matrix = numeric_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('数值型变量相关性热图')
plt.xlabel('变量')
plt.ylabel('变量')
plt.show()


# In[ ]:




