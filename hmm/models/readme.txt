在刚开始的时候没有注意保存的模型命名的问题，尤其是模型对应的traincol。

以下是整理后的各模型参数：

model                                 |trainset| traincol    |n_components | n_iter          |create_time
---------------------------------------------------------------------------------------------------------------------------------
model_2000_car_300_iter_v_u_diata_12m2d21h31min.pkl  | cdf     |v u diata    |16        | 300           |12m2d21h31min
model_test_v_u_diata_stops_12m2d19h48min.pkl       |cdf[:100*1600]|v u diata |12        | 50            |12m2d19h48min
model_2000_car_200_iter_12m1d7h47min.pkl         | cdf     |v         |16        | 200           |12m1d7h47min
model1.pkl                             | cdf     |v         |8         |100           |-