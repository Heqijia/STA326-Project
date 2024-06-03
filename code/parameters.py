stations_N = 25
skipped_saved = 1.2
station_interval = 1.5
# round_trip_interval = 2.7 #这个没用
train_N = 27
passenger_max = 75
# para = 1.5  
para_list=[2.5, 3, 3.5]  # 变成网格参数
reduce_list = {0.3, 0.4, 0.5, 0.6, 1, 5, 10, 20} # algorithm‘3 搜索范围


import numpy as np
column_names = np.arange(stations_N*2)
column_names_with_stop = np.array([f"{col}_stop" for col in column_names])