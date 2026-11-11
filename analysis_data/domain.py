"""核对动作概率，计算观察到的折扣回报。"""
import math
from .io import finite,unique
def distribution(values):
    if not values or any(finite(x)<0 or x>1 for x in values.values()) or not math.isclose(sum(values.values()),1,abs_tol=1e-12,rel_tol=0):raise ValueError('概率分布错误')
def summarize(value):
    unique(value['episodes'],'episode_id');h=value['H'];gamma=finite(value['gamma'])
    if type(h) is not int or h<=0 or not 0<=gamma<=1:raise ValueError('长度或折扣错误')
    result=[]
    for ep in value['episodes']:
        steps=ep['steps'];total=0
        for t,row in enumerate(steps):
            if row['t']!=t or t>=h:raise ValueError('步骤编号错误')
            distribution(row['behavior']);distribution(row['target'])
            if row['behavior'].keys()!=row['target'].keys() or row['behavior'].get(row['action'],0)<=0:raise ValueError('动作空间或已选概率错误')
            total+=gamma**t*finite(row['reward'])
        result.append({'episode_id':ep['episode_id'],'site':ep['site'],'observed_return':total,'observed_steps':len(steps),'end':ep['end']})
    return result
