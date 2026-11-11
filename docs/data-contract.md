# 回合概率与离线回报

episode_id在文件内唯一，site是整群重抽单位。t从零连续，end=terminal表示已观察自然终止，end=horizon表示完整到达H，end=truncated表示资料被截断。terminal可在H前终止，后续只按吸收规则延展；horizon必须恰好H步。无效或truncated队列不输出正式完整队列结果，可单列观测描述。每个step为已发生动作、奖励及该历史下全部合法动作的行为/目标概率。

normal的每site都含二步完整树的四条等权行为路径，H=3所以第三步仅吸收，目标期望为1.2，行为期望为1.4；finite-tree-reference是独立列出的有限树数据。ESS=(sum rho)^2/sum rho^2，零权重和时该步估计与ESS不可估，不能定义为0收益。site bootstrap使用有放回抽取原site数量的副本，百分位使用排序样本的线性插值分位数；重复site必须给复制身份，不能被episode_id唯一检查去重。权重截断对每时点累积rho独立min(rho,c)，不递推截断值。

domain.summarize只核对概率与实际所选动作、计算原始回报，没有累积策略权重、支持域判断或bootstrap。
