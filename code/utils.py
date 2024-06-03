import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from parameters import *


"""
在已知发车时间和停靠站时间的情况下，直接计算下一班车的整个运行时间表
"""
def next_train(stop_stations, StartTrain_time, column_names=column_names, column_names_with_stop=column_names_with_stop):
    time_arrived=[]
    for j in range(stations_N*2):
        if stop_stations[j]==0:
            time_arrived.append(StartTrain_time)   # 如果不经过该站，就纪录经过该站的瞬时时间
            StartTrain_time+=station_interval
        else:
            time_arrived.append(StartTrain_time)  # 否则加上经停时间
            StartTrain_time+=station_interval+skipped_saved

    schedule_temp=pd.DataFrame(time_arrived+stop_stations).T
    schedule_temp.columns=np.concatenate([column_names, column_names_with_stop])
    schedule_temp['restart_time'] = schedule_temp[str(stations_N*2-1)] + station_interval + skipped_saved*schedule_temp[f"{stations_N-1}_stop"]
    return schedule_temp

"""
获取当前时间内，正在轨道上运行的车辆位置和schedule
"""
def get_ActiveSchedule(schedule, time):
    return schedule[(schedule['restart_time']>time)&
                     (schedule['0']<=time)] # 已经发车且未到终点站的车

"""
初始化地铁系统，每个车辆都具有相同的时间间距，返回一个初始的schedule
"""
def initialization():
    column_names = np.arange(stations_N*2)
    column_names_with_stop = np.array([f"{col}_stop" for col in column_names])
    df_schedule = pd.DataFrame(columns=np.concatenate([column_names, column_names_with_stop, ['restart_time']]))


    for i in range(train_N):  # 先把所有的车全部送出去
        stop_stations=list(np.ones(stations_N*2).astype(int))  # 初始化：所有的站都停
        StartTrain_time=i*(stations_N*2*(skipped_saved+station_interval)/train_N)    #在早高峰之前已经完成车辆全局部署, StartTrian_time是每辆车的发车时间
        
        schedule_temp=next_train(stop_stations, StartTrain_time, column_names, column_names_with_stop)
        df_schedule=pd.concat([df_schedule,schedule_temp],ignore_index=True)
    return df_schedule, StartTrain_time

"""
拿到下一班车的最早发车时间
"""
def get_next_starting_Time(active_schedule):
    return active_schedule['restart_time'].min()



"""
对于一个车站，已知此前最后一班车离开的时间和按照计划表开入的时间，计算在该站会有多少人上车
"""
def my_passengers_InStation(arrived_time, station_set, station, df_):
    return df_[(df_['start_']==station)&
                (df_['time']<=arrived_time)&
                (df_['end_'].isin(station_set))].reset_index(drop=True)


"""
根据当前的停靠站和历史数据，得到该列车能承载的全部乘客
"""
def get_df_ofPassengers(next_schedule_Ref, stop_stations,df_changing, passenger_max=passenger_max):
    station_set=[]
    for i in range(stations_N*2):
        if stop_stations[i]==1:
            station_set.append(i)

    df_temp=pd.DataFrame(columns=df_changing.columns)
    passengers=[]
    for i in range(stations_N*2):
        if stop_stations[i]==1:  # 表明经过这一站，
            new_passengers_temp=my_passengers_InStation(
                                                arrived_time=next_schedule_Ref[str(i)].values[0], 
                                                station_set=station_set, 
                                                station=i, 
                                                df_=df_changing.copy()).reset_index(drop=True)
            # 限制人数的三行代码
            passsengers_on_train=len(df_temp[(df_temp['start_']<i)&(df_temp['end_']>i)]) #end=i的要在这里下车
            # if flag==1:
                # print('potenial_passengers_to_train:',len(new_passengers_temp), 'vacant_space:',passenger_max-passsengers_on_train)
            new_passengers_temp=new_passengers_temp.iloc[:passenger_max-passsengers_on_train]  # 保证不超载
            
            df_temp=pd.concat([df_temp,new_passengers_temp])
            passengers.append(len(new_passengers_temp))
        else:
            passengers.append(0)
    return df_temp, passengers

        

"""
这个输入删除前的接待人次和删除后的接待人次，计算节约的总通勤时间成本
"""
# def calculate_TotalTimeSaved(df_before, df_after, drop_station,arrival_time_bef,para,drop_N):
#     # 浪费的时间只有：1.在该站没坐下车的； 2. 在该站没能上车的
    
#     df_before_sub=df_before[(df_before['end_']==drop_station)|(df_before['start_']==drop_station)]
#     cost=df_before_sub.shape[0]*(stations_N*2*(skipped_saved+station_interval)/train_N)   
#     gain=df_after[(df_after['end_']>drop_station)&(df_after['start_']<drop_station)].shape[0]*skipped_saved
#     # print(len(df_after[df_after['end_']==drop_station]),len(df_after[df_after['start_']==drop_station]))
#     max_waiting_time=(-df_before[df_before['start_']==drop_station]['time']+arrival_time_bef).max()
#     # if len(df_before)==0:
#     #     df_ratio=1/(stations_N)
#     # else:
#     #     df_ratio=len(df_before_sub)/len(df_before)  # 双向＋往返，自动算成每站的比例重要性
#     # # print(df_before)
#     # # print(df_after)
#     # # print('are you here?')
#     # if df_ratio>2/(stations_N):  # 有进站的，出站的，表明这站的重要性很大
#     #     print('big_station', gain-cost*para<0)
#     #     return -1  # 不跳站
#     # elif df_ratio<0.05/(stations_N):  # 这站不重要，阈值先设低点
#     #     print('small_station', gain-cost*para>0)
#     #     return 1  # 跳站
#     return gain-cost*para  # 这玩意大于0就跳站

# def calculate_TotalTimeSaved(df_before, df_after, drop_station,arrival_time_bef,para, drop_N):
#     before_eff=(df_before['end_']-df_before['start_']).sum()/((skipped_saved+station_interval)*stations_N*2-(drop_N)*skipped_saved)
#     after_eff=(df_after['end_']-df_after['start_']).sum()/((skipped_saved+station_interval)*stations_N*2-(drop_N+1)*skipped_saved)
#     return after_eff-before_eff

def calculate_TotalTimeSaved_2(df_before, df_after, drop_station,arrival_time_bef,para, drop_N):
    if drop_station>=stations_N:
        df_before=df_before[(df_before['start_']>=stations_N)]
        df_after=df_after[(df_after['start_']>=stations_N)]
    else:
        df_before=df_before[(df_before['end_']<stations_N)]
        df_after=df_after[(df_after['end_']<stations_N)]
    before_eff=(df_before['end_']-df_before['start_']).sum()/((skipped_saved+station_interval)*stations_N-(drop_N)*skipped_saved)
    after_eff=(df_after['end_']-df_after['start_']).sum()/((skipped_saved+station_interval)*stations_N-(drop_N+1)*skipped_saved)
    return after_eff-before_eff


def calculate_TotalTimeSaved_1(df_before, df_after, drop_station,arrival_time_bef,para,drop_N):
    # 先限制半程范围
    if drop_station>=stations_N:
        df_before=df_before[(df_before['start_']>=stations_N)]
        df_after=df_after[(df_after['start_']>=stations_N)]
    else:
        df_before=df_before[(df_before['end_']<stations_N)]
        df_after=df_after[(df_after['end_']<stations_N)]
    
    cost=(df_before.shape[0]-df_after.shape[0])*stations_N*2*(skipped_saved+station_interval)/train_N
    # 加速了的人的增益
    gain=df_after[(df_after['end_']>drop_station)].shape[0]*skipped_saved
    return gain-cost*para   #这玩意大于0就跳站



def arrival_time(start_, end_, time,df_schedule):
    df_temp=df_schedule[(df_schedule[str(start_)]>=time)   # 只能做在进站之后的车
                        &(df_schedule[str(end_)+'_stop']==1)  # 在起点终点要停
                        &(df_schedule[str(start_)+'_stop']==1)]
    try:
        out_time=df_schedule.loc[df_temp[str(start_)].idxmin()][str(end_)]  # 选最早可以乘坐的车,并获取到站时间
        cost_time=out_time-time
        return out_time, cost_time  # 对于单个人次 通勤的时间
    except:
        print('no train is avaliable for this guy!')
        print('*'*50)
        return -1, -1  # 对于单个人次 通勤的时间
    

def next_train(stop_stations, StartTrain_time, column_names=column_names, column_names_with_stop=column_names_with_stop):
    time_arrived=[]
    for j in range(stations_N*2):
        if stop_stations[j]==0:
            time_arrived.append(StartTrain_time)   # 如果不经过该站，就纪录经过该站的瞬时时间
            StartTrain_time+=station_interval
        else:
            time_arrived.append(StartTrain_time)  # 否则加上经停时间
            StartTrain_time+=station_interval+skipped_saved


    schedule_temp=pd.DataFrame(time_arrived+stop_stations).T
    schedule_temp.columns=np.concatenate([column_names, column_names_with_stop])
    
    schedule_temp['restart_time'] =schedule_temp[str(stations_N*2-1)] + station_interval + skipped_saved*schedule_temp[f"{stations_N-1}_stop"]
    return schedule_temp


"""
根据贪婪算法 得到下一台车的运行计划
算法思路：
1. 遍历全部的站点，计算经过站点和不经过站点损失的时间和收益的时间
    收益时间： 在车上坐着的人数*节约时间（1.5 min）
    损失时间:  未能成功坐到该车的人数（即本来在该站下车和在该站上车的人数总和）*等待时间（5.6 min）
    （这是一种只考虑到当前状态的算法，司机在决策停车开车的时候并不知道（可能）在前面的站台会有更多/更少的人在等待）
    ，而且这是一个相对值，故无论人数多少，只要是均匀分布就基本上会在每站停车（极端情况：A-->B没人而B-->A一大堆人，
    正确的决策是跳过A-->B的所有站而去接B-->A的乘客，但这个算法做不到这样）
2. 收益时间-损失时间最大者，不经停该站点
3. 重复1-2，直到没有任何站点可以不经停，即利益最大化
"""
def newTrain(StartTrain_time,df_changing,para,algorithm):  # changing 里面是剩下的客人
    # print('start!')
    next_start_time=StartTrain_time
    stop_stations=list(np.ones(stations_N*2).astype(int))  # 初始化：所有的站都停   
    next_schedule_Ref=next_train(stop_stations=stop_stations.copy(), StartTrain_time=next_start_time)
    df_total_passengersInTrain_Ref, p_r=get_df_ofPassengers(next_schedule_Ref=next_schedule_Ref.copy(),
                        stop_stations=stop_stations.copy(),
                        df_changing=df_changing.copy())

    # 此时还没有客人出现，继续按照最原始的路线运行
    # print('checked_one_schedule, and Ref=',len(df_total_passengersInTrain_Ref))
    if (len(df_total_passengersInTrain_Ref)==0):
        # print('*'*50)
        return next_train(StartTrain_time=StartTrain_time, stop_stations=stop_stations.copy()), 0

    
    # 出现了客人！
    drop_N=0
    for i in range(stations_N*2):
        if (i+1)%stations_N==0:
            drop_N=0  # drop N 重置
        stop_stations_sub=stop_stations.copy()
        stop_stations_sub[i]=0 # 跳过第i个站点

        next_schedule_sub=next_train(stop_stations=stop_stations_sub.copy(), StartTrain_time=next_start_time)
        df_total_passengersInTrain_Sub,p_s=get_df_ofPassengers(next_schedule_Ref=next_schedule_sub.copy(),
                        stop_stations=stop_stations_sub.copy(),df_changing=df_changing.copy())

        if algorithm==1:
            saved_time=calculate_TotalTimeSaved_1(df_before=df_total_passengersInTrain_Ref.copy(),
                                                df_after=df_total_passengersInTrain_Sub.copy(),
                                                drop_station=i,
                                                arrival_time_bef=next_schedule_Ref[str(i)].values[0],
                                                para=para,
                                                drop_N=drop_N)
        else:
            saved_time=calculate_TotalTimeSaved_2(df_before=df_total_passengersInTrain_Ref.copy(),
                                                df_after=df_total_passengersInTrain_Sub.copy(),
                                                drop_station=i,
                                                arrival_time_bef=next_schedule_Ref[str(i)].values[0],
                                                para=para,
                                                drop_N=drop_N)

        if saved_time>0:
            df_total_passengersInTrain_Ref=df_total_passengersInTrain_Sub.copy()
            next_schedule_Ref=next_schedule_sub.copy()
            stop_stations=stop_stations_sub.copy()
            drop_N+=1

            
    # 每次选全局最优，但是时间复杂度是n^2
    # best_opt = -1
    # best_opt_time = 0
    # keep_optimizing = True
    # temp_passengers=[]

    # while keep_optimizing:
    #     keep_optimizing = False  # 设定一个标志位，如果内部循环有优化则置为True

    #     for i in range(stations_N * 2):
    #         if(stop_stations[i]==0):
    #             pass
    #         else:
    #             stop_stations_sub = stop_stations.copy()
    #             stop_stations_sub[i] = 0  # 跳过第i个站点
    #             next_schedule_sub = next_train(stop_stations=stop_stations_sub.copy(), StartTrain_time=next_start_time)
    #             df_total_passengersInTrain_Sub, p_s = get_df_ofPassengers(next_schedule_Ref=next_schedule_sub.copy(),
    #                                                                     stop_stations=stop_stations_sub.copy(),
    #                                                                     df_changing=df_changing.copy())
    #             temp_passengers.append(p_s)
    #             saved_time = calculate_TotalTimeSaved(df_before=df_total_passengersInTrain_Ref.copy(),  # Reference 都是和最原始的比
    #                                                 df_after=df_total_passengersInTrain_Sub.copy(),
    #                                                 drop_station=i)
    #             if saved_time >= 0 and saved_time >= best_opt_time:
    #                 best_opt = i
    #                 best_opt_time = saved_time
    #                 keep_optimizing = True  # 如果有优化，则置为True
    #                 temp_passengers=[temp_passengers[-1]]

    #     if best_opt != -1:
    #         stop_stations[best_opt] = 0
    #         # 在每轮结束后重置最优值，准备下一轮优化
    #         best_opt = -1
    #         best_opt_time = 0
        

    # print('stop_stations:',stop_stations)
    # print(StartTrain_time)
    return next_train(StartTrain_time=StartTrain_time, stop_stations=stop_stations.copy()),1

# 依据schedule 0101生成列车发车时间表
def get_train_schedule(schedule):
    train_schedule = []
    for i in range(train_N):
        train_schedule.append(schedule['restart_time'].values[i])
    return train_schedule   



# 计算通勤时间的upperbound（在列车超载的情况下，原来的方法不适用了）
def calculate_waitingTime_Up(df,df_schedule):
    df_changing=df.copy()
    print(df_changing.shape)
    waiting_time=0
    waiting_time_list=[]
    station_interval=[]
    check=0
    i=0
    while df_changing.shape[0]>0: # 如果
        i+=1
        temp_schedule=df_schedule.iloc[[i-1]]
        stop_stations=temp_schedule.T.reset_index()[stations_N*2:stations_N*4][i-1].values
        df_schedule=pd.concat([df_schedule, temp_schedule])
        df_drop, _=get_df_ofPassengers(next_schedule_Ref=temp_schedule, stop_stations=stop_stations.copy(), df_changing=df_changing.copy()) # 新车把旧车接走了
        check+=df_drop.shape[0]

        if len(df_drop)!=0:
            waiting_time_list.append(df_drop.apply(lambda row: arrival_time(row['start_'], row['end_'], row['time'],df_schedule=temp_schedule)[1], axis=1))
            waiting_time+=waiting_time_list[-1].sum()
            station_interval.append(list(df_drop['end_']-df_drop['start_']))

        _=df_changing.shape[0]
        df_changing=pd.merge(df_changing, df_drop, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

    return waiting_time


# make sure you have already add enough 'time' to col time 
def calculate_waitingTime_Algorithm_1(df,para):
        df_schedule, StartTrain_time=initialization()
        df_changing=df.copy()
        print(df_changing.shape)
        waiting_time=0
        waiting_time_list=[]
        station_interval_list=[]
        check=0
        while df_changing.shape[0]>0:
            print('df_changing.shape:',df_changing.shape[0])
            StartTrain_time=get_next_starting_Time(get_ActiveSchedule(df_schedule, StartTrain_time))
            temp_schedule, have_p=newTrain(StartTrain_time=StartTrain_time,df_changing=df_changing.copy(),para=para, algorithm=1)
            stop_stations=temp_schedule.T.reset_index()[stations_N*2:stations_N*4][0].values
            df_schedule=pd.concat([df_schedule, temp_schedule])
            df_drop, _=get_df_ofPassengers(next_schedule_Ref=temp_schedule, stop_stations=stop_stations.copy(), df_changing=df_changing.copy()) # 新车把旧车接走了
            check+=df_drop.shape[0]

            if len(df_drop)!=0:
                waiting_time_list.append(df_drop.apply(lambda row: arrival_time(row['start_'], row['end_'], row['time'],df_schedule=temp_schedule)[1], axis=1))
                waiting_time+=waiting_time_list[-1].sum()
                station_interval_list.append(list(df_drop['end_']-df_drop['start_']))
                # print(len(df_drop))
                # print('waiting_time:', waiting_time)
            _=df_changing.shape[0]
            df_changing=pd.merge(df_changing, df_drop, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        df_schedule.reset_index(drop=True, inplace=True)
        return waiting_time, df_schedule, station_interval_list, waiting_time_list


def calculate_waitingTime_Algorithm_2(df,para): 
        df_schedule, StartTrain_time=initialization()
        df_changing=df.copy()
        print(df_changing.shape)
        waiting_time=0
        waiting_time_list=[]
        station_interval_list=[]
        check=0
        while df_changing.shape[0]>0:
            print('df_changing.shape:',df_changing.shape[0])
            StartTrain_time=get_next_starting_Time(get_ActiveSchedule(df_schedule, StartTrain_time))
            temp_schedule, have_p=newTrain(StartTrain_time=StartTrain_time,df_changing=df_changing.copy(),para=para, algorithm=2)
            stop_stations=temp_schedule.T.reset_index()[stations_N*2:stations_N*4][0].values
            df_schedule=pd.concat([df_schedule, temp_schedule])
            df_drop, _=get_df_ofPassengers(next_schedule_Ref=temp_schedule, stop_stations=stop_stations.copy(), df_changing=df_changing.copy()) # 新车把旧车接走了
            check+=df_drop.shape[0]

            if len(df_drop)!=0:
                waiting_time_list.append(df_drop.apply(lambda row: arrival_time(row['start_'], row['end_'], row['time'],df_schedule=temp_schedule)[1], axis=1))
                waiting_time+=waiting_time_list[-1].sum()
                station_interval_list.append(list(df_drop['end_']-df_drop['start_']))
                # print(len(df_drop))
                # print('waiting_time:', waiting_time)
            _=df_changing.shape[0]
            df_changing=pd.merge(df_changing, df_drop, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        df_schedule.reset_index(drop=True, inplace=True)
        return waiting_time, df_schedule, station_interval_list, waiting_time_list


def algorithm_1(df,para_list, passenger_max=passenger_max, train_N=train_N,stations_N=stations_N):
    waiting_time_withPara=[]
    station_interval_list_withPara=[]
    waiting_time_list_withPara=[]
    df_schedule_withPara=[]

    for i in range(len(para_list)):
        df_schedule, StartTrain_time=initialization()
        df_changing=df.copy()
        print(df_changing.shape)
        waiting_time=0
        waiting_time_list=[]
        station_interval_list=[]
        check=0
        while df_changing.shape[0]>0:
            print('df_changing.shape:',df_changing.shape[0])
            StartTrain_time=get_next_starting_Time(get_ActiveSchedule(df_schedule, StartTrain_time))
            temp_schedule, have_p=newTrain(StartTrain_time=StartTrain_time,df_changing=df_changing.copy(),para=para_list[i],algorithm=1)
            stop_stations=temp_schedule.T.reset_index()[stations_N*2:stations_N*4][0].values
            df_schedule=pd.concat([df_schedule, temp_schedule])
            df_drop, _=get_df_ofPassengers(next_schedule_Ref=temp_schedule, stop_stations=stop_stations.copy(), df_changing=df_changing.copy()) # 新车把旧车接走了
            check+=df_drop.shape[0]

            if len(df_drop)!=0:
                waiting_time_list.append(df_drop.apply(lambda row: arrival_time(row['start_'], row['end_'], row['time'],df_schedule=temp_schedule)[1], axis=1))
                waiting_time+=waiting_time_list[-1].sum()
                station_interval_list.append(list(df_drop['end_']-df_drop['start_']))
                # print(len(df_drop))
                # print('waiting_time:', waiting_time)
            _=df_changing.shape[0]
            df_changing=pd.merge(df_changing, df_drop, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        df_schedule.reset_index(drop=True, inplace=True)
        waiting_time_withPara.append(waiting_time)
        station_interval_list_withPara.append(station_interval_list)
        waiting_time_list_withPara.append(waiting_time_list)
        df_schedule_withPara.append(df_schedule)
    
    
    best_para_index=np.argmin(waiting_time_withPara)

    """返回5个值： schedule,总通勤时间，站间距统计，个体的通勤时间（这两个用来画箱线图），最优参数"""
    return df_schedule_withPara[best_para_index], waiting_time_withPara[best_para_index], station_interval_list_withPara[best_para_index], waiting_time_list_withPara[best_para_index], para_list[best_para_index]

def algorithm_2(df, para_list, passenger_max=passenger_max, train_N=train_N,stations_N=stations_N):
    df_schedule, StartTrain_time=initialization()
    df_changing=df.copy()
    print(df_changing.shape)
    waiting_time=0
    waiting_time_list=[]
    station_interval_list=[]
    check=0
    while df_changing.shape[0]>0:
        print('df_changing.shape:',df_changing.shape[0])
        StartTrain_time=get_next_starting_Time(get_ActiveSchedule(df_schedule, StartTrain_time))
        temp_schedule, have_p=newTrain(StartTrain_time=StartTrain_time,df_changing=df_changing.copy(),para=0,algorithm=2)  # 这个地方不需要para,随便设成0
        stop_stations=temp_schedule.T.reset_index()[stations_N*2:stations_N*4][0].values
        df_schedule=pd.concat([df_schedule, temp_schedule])
        df_drop, _=get_df_ofPassengers(next_schedule_Ref=temp_schedule, stop_stations=stop_stations.copy(), df_changing=df_changing.copy()) # 新车把旧车接走了
        check+=df_drop.shape[0]

        if len(df_drop)!=0:
            waiting_time_list.append(df_drop.apply(lambda row: arrival_time(row['start_'], row['end_'], row['time'],df_schedule=temp_schedule)[1], axis=1))
            waiting_time+=waiting_time_list[-1].sum()
            station_interval_list.append(list(df_drop['end_']-df_drop['start_']))
            # print(len(df_drop))
            # print('waiting_time:', waiting_time)
        _=df_changing.shape[0]
        df_changing=pd.merge(df_changing, df_drop, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    df_schedule.reset_index(drop=True, inplace=True)
    """返回四个值：schedule,总通勤时间，站间距统计，个体的通勤时间（这两个用来画箱线图）"""
    return df_schedule, waiting_time, station_interval_list, waiting_time_list

# 点对点的运输
def cal_ranking(df):
    time=df['time'].max()-df['time'].min()
    combined=df['start_'].astype(str) + '-' + df['end_'].astype(str)
    station_pair=pd.DataFrame(combined.value_counts().reset_index().rename(columns={'index':'station_pair',0:'count'}))
#     print(station_pair['station_pair'])
#     station_pair['start_'],station_pair['end_']=station_pair['station_pair'].str.split('-').str
    
    # Split on the first occurrence of '-' and expand into two columns
    station_pair[['start_', 'end_']] = station_pair['station_pair'].str.split('-', n=1, expand=True)

    # Convert the start_ column to integers
#     station_pair['start_'] = station_pair['start_'].astype(int)
#     station_pair['start_'],station_pair['end_']=df['start_'], df['end_']
    station_pair['start_'] = station_pair['start_'].astype(int)
    station_pair['end_'] = station_pair['end_'].astype(int)
#     station_pair['arriving_speed']=station_pair['frequency']/time
    station_pair['arriving_speed']=station_pair['count']
    station_pair['arriving_speed_normalized']=station_pair['arriving_speed']/station_pair['arriving_speed'].max()
    station_pair['ralative_space']=station_pair['count']/passenger_max
    return station_pair


######################### Algorithm'3 #################################################
#######################################################################################
# def stop(df_pair,para_stop=para_stop):
def stop(df_pair, reduce):
    stop=[]
#     print(passenger_max/df_pair['count'].sum())
    for i in range(df_pair.shape[0]):
        ########################
        ## 直接用阈值筛选（阈值由人数排布方差决定，使用总人数进行标准化）,公式中除以3.3为优化后结果
        ratio = np.std(df_pair['count'])/reduce /df_pair['count'].sum()
#         ratio = 0.003
#         print(ratio)
        
        if (df_pair['count'][i]/df_pair['count'].sum()) < ratio:
#         if (df_pair['count'][i]/df_pair['count'].sum()) < 0.0011:
            stop.append(0)
        else:
            stop.append(1)
            
        #########################
    df_pair['stop']=stop
    return df_pair

def back_to_schedule(df_pair):
    df_pair_stop=df_pair[df_pair['stop']==1]
    stop_stations_temp=np.union1d(df_pair_stop['start_'].unique(),
                                  df_pair_stop['end_'].unique())
    stop_stations=np.zeros(stations_N*2).astype(int)
    for i in range(stations_N*2):
        stop_stations[i]=1 if i in stop_stations_temp else 0
    return stop_stations

def generate_schedule(df_pair, reduce):
    df_pair=stop(df_pair.copy(), reduce)
    return back_to_schedule(df_pair.copy())


def algorithm_3(reduce_list, df, para_list, passenger_max=passenger_max, train_N=train_N,stations_N=stations_N):
    waiting_time_min = -1
    best_para = 0
    for reduce in reduce_list:
#         print(reduce)
        df_schedule, waiting_time, station_interval_list, waiting_time_list=algorithm_3_iteration(reduce, df=df.copy(), para_list=para_list.copy(), passenger_max=passenger_max,train_N=train_N,stations_N=stations_N)
        if ((waiting_time_min == -1) | (waiting_time_min > waiting_time)):
            waiting_time_min = waiting_time
            best_para = reduce
    return df_schedule, waiting_time_min, station_interval_list, waiting_time_list, best_para
            
def algorithm_3_iteration(reduce, df, para_list, passenger_max=passenger_max, train_N=train_N,stations_N=stations_N):
    waiting_time=0
    waiting_time_list=[]
    station_interval_list=[]
    df_changing=df.copy()
    df_schedule, StartTrain_time=initialization()
    df_pair=cal_ranking(df.copy())
    slow_flag=0
    
    stop_stations_gen = generate_schedule(df_pair.copy(), reduce)
    
    while df_changing.shape[0]>0:
        StartTrain_time=get_next_starting_Time(get_ActiveSchedule(df_schedule, StartTrain_time)) 
        if slow_flag%5!=0:
#             stop_stations = generate_schedule(df_pair.copy(), reduce)
            stop_stations = stop_stations_gen.copy()
        else:
            stop_stations=list(np.ones(stations_N*2).astype(int))
        slow_flag+=1
        temp_schedule=next_train(stop_stations=list(stop_stations), StartTrain_time=StartTrain_time)
        df_schedule=pd.concat([df_schedule, temp_schedule])
        df_schedule.reset_index(drop=True, inplace=True)
        df_drop, _=get_df_ofPassengers(next_schedule_Ref=temp_schedule, stop_stations=stop_stations.copy(), df_changing=df_changing.copy()) # 新车把旧车接走了
        if len(df_drop)!=0:
            waiting_time_list.append(df_drop.apply(lambda row: arrival_time(row['start_'], row['end_'], row['time'],df_schedule=temp_schedule)[1], axis=1))
            waiting_time+=waiting_time_list[-1].sum()
            station_interval_list.append(list(df_drop['end_']-df_drop['start_']))

        df_changing=pd.merge(df_changing, df_drop, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
#         print("stop_stations:", stop_stations)

    df_schedule.reset_index(drop=True, inplace=True)
    """返回四个值：schedule,总通勤时间，站间距统计，个体的通勤时间（这两个用来画箱线图）"""
    return df_schedule, waiting_time, station_interval_list, waiting_time_list 
#######################################################################################################################################################################################################################################

def algorithm_Ref(df, para_list, passenger_max=passenger_max, train_N=train_N,stations_N=stations_N):
    df_schedule, StartTrain_time=initialization()
    waiting_time=0
    waiting_time_list=[]
    station_interval_list=[]
    df_changing=df.copy()
    check=0
    while len(df_changing)>0:
        StartTrain_time=get_next_starting_Time(get_ActiveSchedule(df_schedule, StartTrain_time)) 
        temp_schedule=next_train(stop_stations=list(np.ones(stations_N*2).astype(int)), StartTrain_time=StartTrain_time)
        stop_stations=temp_schedule.T.reset_index()[stations_N*2:stations_N*4][0].values
        df_schedule=pd.concat([df_schedule, temp_schedule])
        df_drop, _=get_df_ofPassengers(next_schedule_Ref=temp_schedule, stop_stations=stop_stations.copy(), df_changing=df_changing.copy()) # 新车把旧车接走了
        if len(df_drop)!=0:
            check+=df_drop.shape[0]
            waiting_time_list.append(df_drop.apply(lambda row: arrival_time(row['start_'], row['end_'], row['time'],df_schedule=temp_schedule)[1], axis=1))
            waiting_time+=waiting_time_list[-1].sum()
            station_interval_list.append(list(df_drop['end_']-df_drop['start_']))
            # print(len(df_drop))
            print('waiting_time:', waiting_time)
        df_changing=pd.merge(df_changing, df_drop, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        # print("stop_stations:", stop_stations)
    df_schedule.reset_index(drop=True, inplace=True)
    """返回四个值：schedule,总通勤时间，站间距统计，个体的通勤时间（这两个用来画箱线图）"""
    return df_schedule, waiting_time, station_interval_list, waiting_time_list 