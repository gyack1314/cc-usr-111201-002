"""有限状态树的完整路径枚举：独立精确期望，用于核对无截断PDIS/WPDIS估计。"""
from .domain import distribution
from .io import finite


def expectation(tree, policy):
    """枚举全部路径，按策略列('behavior'或'target')计算折扣期望收益。

    树节点给出每条历史下全部合法动作的概率与每动作奖励；深度为最长历史+1，
    到达深度的路径不再决策。树不完整或概率表不合法抛ValueError。
    """
    if policy not in ('behavior', 'target'):
        raise ValueError('未知策略列')
    nodes = {}
    for node in tree['nodes']:
        hist = tuple(node['history'])
        if hist in nodes:
            raise ValueError('树节点重复')
        distribution(node['behavior'])
        distribution(node['target'])
        if node['behavior'].keys() != node['target'].keys():
            raise ValueError('行为与目标动作空间不一致')
        if node['reward'].keys() != node['behavior'].keys():
            raise ValueError('奖励缺少合法动作')
        for r in node['reward'].values():
            finite(r)
        nodes[hist] = node
    if () not in nodes:
        raise ValueError('树缺少根节点')
    depth = max(len(h) for h in nodes) + 1
    gamma = finite(tree['discount'])
    if not 0 <= gamma <= 1:
        raise ValueError('折扣错误')

    def walk(hist):
        if len(hist) == depth:
            return 0.0
        node = nodes.get(hist)
        if node is None:
            raise ValueError('树不完整：缺少历史' + repr(list(hist)))
        return sum(p * (node['reward'][a] + gamma * walk(hist + (a,)))
                   for a, p in node[policy].items() if p > 0)

    return walk(())
