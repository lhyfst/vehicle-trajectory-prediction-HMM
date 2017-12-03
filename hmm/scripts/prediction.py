#以下可以考虑拉出来当做函数，反正基本上已经封装完了

# 传入由于测试的数据
test_df = ctdf
# 传入观测点(基准点)
base_point = 1000
# 传入答案点
anwser_point = 1001

# 指定用于训练和预测的列
traincol = array(test_df['v']).reshape(-1,1)

Z = model.decode(traincol)
test_df['label'] = DataFrame(Z[1])

state_num = len(model.startprob_)

print '---------*观察报告*---------'
print '每种状态对应的轨迹数'
for i in range(state_num):
    tmp = test_df[test_df['label']==i]
    print '状态',i,'有',len(tmp),'条轨迹'

# 从均值的角度观察一下数据，发现确实是有效果的，并将状态对应的v储存
tmp = 0
state_v=[]
print '每种状态对应的速度'
for tmp in range(state_num):
    state_v.append(mean(test_df[test_df['label']==tmp].v))
print state_v

# 每条轨迹长度
length = len(test_df[test_df['id']==0])
# 轨迹数
car_num = int(max(array([test_df['id']]).flatten())-min(array([test_df['id']]).flatten())+1)
print '每条轨迹长度:',length
print '轨迹数:',car_num

# 设置基准点和答案点
base_points = test_df.iloc[[i*length+base_point for i in range(car_num)]]                     #
anwser_points = test_df.iloc[[i*length+anwser_point for i in range(car_num)]]

base_points_v = array(base_points['v'])
base_points_v = base_points_v.reshape(-1,1)

# 以下2条注释代码在求以概率最大的state作为预测state时会用到
# last_state = model.predict(base_points_v)
# state_v_col_vec = array(state_v).reshape(-1,1)

# 上一点状态的概率矩阵
last_states_proba = model.predict_proba(base_points_v) #这里没有设置length，但是有可能需要，否则有可能被当做是同一个序列
# 下一点状态的概率矩阵
predict_proba_next_points = dot(last_states_proba,model.transmat_)
# 通过下一点状态的概率矩阵进行对应速度均值，得出的预测速度
predict_next_points_v = dot(predict_proba_next_points,state_v_col_vec)

predict_next_position = []
tmp=0
predict_next_points_v = predict_next_points_v.flatten()
for base_point in base_points.itertuples():
    predict_next_position.append(next_v_to_predict_next_position(predict_next_points_v[tmp],base_point))
    tmp+=1
    
predict_next_position = array(predict_next_position)
print '取前5个预测点观察'
print predict_next_position[:5]