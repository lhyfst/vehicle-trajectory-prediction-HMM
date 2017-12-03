# -*- coding: utf-8 -*-
# data_pretreatment

# 原本想给数据预处理写一个大函数的，但是考虑到仅使用两次，于是作罢，只是如果需要有任何修改的话，必须更改两次
# 我懒得给关于测试集的数据中的变量再起个名字，所以测试集中的变量沿用了训练集中的变量，导致变量非常混乱
# 为此，我索性在该main文件此部分后直接删除了命名空间的所有变量，把得到的数据全部打包在middata中，需要再次加载使用
# 该文件执行后一定有命令%reset -f

# readme:
# 该文件解析了原数据，生成了测试集的df3文件，用于训练的cdf3文件，测试集的tdf3文件，用于训练的ctdf3文件

import pandas as pd
from numpy import *
from funcs import *
from pandas import Series,DataFrame

##########################################################对训练集进行数据预处理####################################################
print 'loading train_set...'
# train_set = loadtxt('testdata.txt')
train_set = loadtxt('/home/liheyuan/Data/hmm/beijing/ly_2kid_1k6ts_train.txt')
df = train_set_to_df(train_set)

# 经纬度转换为坐标-----------------------------------------------------------------------------------------------------------
print 'transforming lon and lat to x and y...'
tmpx = []
tmpy = []
for point in df.itertuples():
    tmpx.append(lon_to_x(point[3]))
    tmpy.append(lat_to_y(point[2]))

df['x'] = tmpx
df['y'] = tmpy


## 1、求出每个点的速度、方向、速度较上一点的改变量、方向较上一点的改变量(v,theta,u,diata)-------------------------------------
print 'computing v and u...'
# 求出每个点的速度、方向(v,theta)-----------------------------------------

tmp = df
vtmp = [0]
thetatmp = [0]
vxtmp = [0]
vytmp = [0]
thetatmp = [0]

for i in range(1,len(tmp)):
    vx = tmp['x'][i] - tmp['x'][i-1]
    vxtmp.append(vx)
    vy = tmp['y'][i] - tmp['y'][i-1]    
    vytmp.append(vy)
    v = xy_to_len(vx,vy)
    vtmp.append(v)
    theta = tan_to_deg(vx,vy)
    thetatmp.append(theta)

df['vx'] = Series(vxtmp)
df['vy'] = Series(vytmp)
df['v'] = Series(vtmp)
df['theta'] = Series(thetatmp)

# 加速度 --速度较上一点的改变量、方向较上一点的改变量(u,diata)--------------
tmp = df
utmp = [0]
diatatmp = [0]
uxtmp = [0]
uytmp = [0]
diatatmp = [0]

for i in range(1,len(tmp)):
    ux = tmp['vx'][i] - tmp['vx'][i-1]
    uxtmp.append(ux)
    uy = tmp['vy'][i] - tmp['vy'][i-1]    
    uytmp.append(uy)
    u = xy_to_len(ux,uy)
    utmp.append(u)
    diata = tan_to_deg(ux,uy)
    diatatmp.append(diata)

df['ux'] = Series(uxtmp)
df['uy'] = Series(uytmp)
df['u'] = Series(utmp)
df['diata'] = Series(diatatmp)


print 'computing stops_time and restart'
# 遍历df，生成对应的stops_time，加做一列------------------------------------------------------------------------------------------------
stops=[]
for car in range(len(df)/1600):
    stops.append(find_thepoint_stops(car,df))

stops = array(stops).flatten()
df['stops_time'] = DataFrame(stops)

# 把stops_time大于40的挑出来，加作一列--------------------------------------------------------------------------------------------------
df['restart']=DataFrame(df['stops_time'] > 40)

# 为了避免可能出现的问题，现在把diata的范围归到[-180,180)-------------------------------------------------------------------------------
df['diata'] = df['diata'] - 180

print 'fixing the error number...'
# 修复漏洞------------------------------------------------------------------------------------------------------------------------------
# 发现一个重大漏洞：每次计算没有把车辆分开计算，导致在index=1600x处一系列数据异常，在index=1600x+1处的加速度异常

# 为了保险起见，将df中的值传入dftest，如果成功，再把dftest传回df
dftest = df
# 每条轨迹的初始点
carnum = int(max(dftest['id'])+1)
tmp1 = dftest.loc[[car*1600 for car in range(carnum)]]
# 每条轨迹的第二个点
tmp2 = dftest.loc[[car*1600+1 for car in range(carnum)]]

# 删除对应的错误行
dftest.drop([car*1600 for car in range(carnum)],inplace=True)
dftest.drop([car*1600+1 for car in range(carnum)],inplace=True)

# 批量修改
tmp1['vx'] = 0.
tmp1['vy'] = 0.
tmp1['v'] = 0.
tmp1['ux'] = 0.
tmp1['uy'] = 0.
tmp1['u'] = 0.
tmp1['theta'] = 0.
tmp1['diata'] = 0.

tmp2['ux'] = tmp2['vx']
tmp2['uy'] = tmp2['vy']
tmp2['u'] = tmp2['v']
tmp2['diata'] = tmp2['theta']

df = dftest.append(tmp1).append(tmp2).sort_index()

# 保存修复后的df
df.to_csv('../middata/df4.csv')
print 'df has been output'

# 提出用于参加训练隐马尔可夫模型的DataFrame：cdf-------------------------------------------------------------------------------------------------
cdf = DataFrame([df['id'],df['x'],df['y'],df['v'],df['theta'],df['u'],df['diata'],df['stops_time']])
cdf = cdf.transpose()
# fillna操作
cdf['theta'] = cdf['theta'].fillna(method='ffill')
cdf['diata'] = cdf['diata'].fillna(0)
# 保存
cdf.to_csv('../middata/cdf2.csv')  
print 'cdf has been ouput'




##########################################################对测试集进行数据预处理####################################################

print 'loading test_set...'
# test_set = loadtxt('testdata2.txt')
test_set = loadtxt('/home/liheyuan/Data/hmm/beijing/ly_1wid_2k1ts_test.txt')
df = train_set_to_df(test_set)

# 经纬度转换为坐标-----------------------------------------------------------------------------------------------------------------------------
print 'transforming lon and lat to x and y...'
tmpx = []
tmpy = []
for point in df.itertuples():
    tmpx.append(lon_to_x(point[3]))
    tmpy.append(lat_to_y(point[2]))

df['x'] = tmpx
df['y'] = tmpy

## 1、求出每个点的速度、方向、速度较上一点的改变量、方向较上一点的改变量(v,theta,u,diata)-------------------------------------
print 'computing v and u...'
# 求出每个点的速度、方向(v,theta)-----------------------------------------

tmp = df
vtmp = [0]
thetatmp = [0]
vxtmp = [0]
vytmp = [0]
thetatmp = [0]

for i in range(1,len(tmp)):
    vx = tmp['x'][i] - tmp['x'][i-1]
    vxtmp.append(vx)
    vy = tmp['y'][i] - tmp['y'][i-1]    
    vytmp.append(vy)
    v = xy_to_len(vx,vy)
    vtmp.append(v)
    theta = tan_to_deg(vx,vy)
    thetatmp.append(theta)

df['vx'] = Series(vxtmp)
df['vy'] = Series(vytmp)
df['v'] = Series(vtmp)
df['theta'] = Series(thetatmp)

# 加速度 --速度较上一点的改变量、方向较上一点的改变量(u,diata)--------------
tmp = df
utmp = [0]
diatatmp = [0]
uxtmp = [0]
uytmp = [0]
diatatmp = [0]

for i in range(1,len(tmp)):
    ux = tmp['vx'][i] - tmp['vx'][i-1]
    uxtmp.append(ux)
    uy = tmp['vy'][i] - tmp['vy'][i-1]    
    uytmp.append(uy)
    u = xy_to_len(ux,uy)
    utmp.append(u)
    diata = tan_to_deg(ux,uy)
    diatatmp.append(diata)

df['ux'] = Series(uxtmp)
df['uy'] = Series(uytmp)
df['u'] = Series(utmp)
df['diata'] = Series(diatatmp)

# -------------------------------------------------此处有bug，隔离观察-----************************************************
print 'computing stops_time and restart'
# 遍历df，生成对应的stops_time，加做一列------------------------------------------------------------------------------------------------
stops=[]
for car in range(len(df)/2100):
    stops.append(find_thepoint_stops(car,df))

stops = array(stops).flatten()
df['stops_time'] = DataFrame(stops)

# 把stops_time大于40的挑出来，加作一列--------------------------------------------------------------------------------------------------
df['restart']=DataFrame(df['stops_time'] > 40)

# -------------------------------------------------上面有bug，隔离观察-----************************************************


# 为了避免可能出现的问题，现在把diata的范围归到[-180,180)-------------------------------------------------------------------------------
df['diata'] = df['diata'] - 180

print 'fixing the error number...'
# 修复漏洞------------------------------------------------------------------------------------------------------------------------------
# 发现一个重大漏洞：每次计算没有把车辆分开计算，导致在index=1600x处一系列数据异常，在index=1600x+1处的加速度异常

# 为了保险起见，将df中的值传入dftest，如果成功，再把dftest传回df
dftest = df
# 每条轨迹的初始点
carnum = int(max(dftest['id'])+1)
tmp1 = dftest.loc[[car*2100 for car in range(carnum)]]
# 每条轨迹的第二个点
tmp2 = dftest.loc[[car*2100+1 for car in range(carnum)]]

# 删除对应的错误行
dftest.drop([car*2100 for car in range(carnum)],inplace=True)
dftest.drop([car*2100+1 for car in range(carnum)],inplace=True)

# 批量修改
tmp1['vx'] = 0.
tmp1['vy'] = 0.
tmp1['v'] = 0.
tmp1['ux'] = 0.
tmp1['uy'] = 0.
tmp1['u'] = 0.
tmp1['theta'] = 0.
tmp1['diata'] = 0.

tmp2['ux'] = tmp2['vx']
tmp2['uy'] = tmp2['vy']
tmp2['u'] = tmp2['v']
tmp2['diata'] = tmp2['theta']

df = dftest.append(tmp1).append(tmp2).sort_index()

# 保存修复后的df
df.to_csv('../middata/tdf4.csv')
print 'tdf has been output'

# 提出用于参加训练隐马尔可夫模型的DataFrame：cdf-------------------------------------------------------------------------------------------------
cdf = DataFrame([df['id'],df['x'],df['y'],df['v'],df['theta'],df['u'],df['diata'],df['stops_time']])
cdf = cdf.transpose()
# fillna操作
cdf['theta'] = cdf['theta'].fillna(method='ffill')
cdf['diata'] = cdf['diata'].fillna(0)
# 保存
cdf.to_csv('../middata/ctdf2.csv')  
print 'ctdf has been ouput'


print 'data_pretreatment has been done'