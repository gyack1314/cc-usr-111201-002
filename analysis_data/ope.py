"""离线策略价值分析：PDIS/WPDIS点估计、支持域诊断、截断敏感性与site整群bootstrap区间。

只评估给定概率表，不训练策略，不向设备发动作。数学口径见docs/data-contract.md：
累积权重rho_t为截至第t步目标/行为概率比值连乘，奖励按gamma^t折扣；提前结束的完整回合
按吸收态补齐到H（奖励0、比值1）且不从任何时点分母消失；截断或缺步回合单列并阻止正式估计。
"""
import hashlib
import json
import math
import random
from .domain import distribution
from .io import finite, unique

_EXP_LIMIT = 709.782712893384  # float64下math.exp的最大安全指数
SCOPE_NOTE = '支持域检查仅覆盖已观察历史；未到达历史的支持域与全局可识别性无法从本资料确认。'


def _log(p):
    """目标概率为0时权重为0，对数空间记为-inf。"""
    return math.log(p) if p > 0 else -math.inf


def _xexp(x):
    """不抛异常的exp：-inf→0，超出float64→inf（后续按不可表示诊断处理）。"""
    if x == -math.inf:
        return 0.0
    return math.exp(x) if x <= _EXP_LIMIT else math.inf


def _config(value):
    """读取并校验分析配置：bootstrap、截断阈值与诊断阈值，缺省给默认值。"""
    boot = value.get('bootstrap', {})
    rep = boot.get('replicates', 199)
    seed = boot.get('seed', 0)
    qs = boot.get('quantiles', [0.025, 0.975])
    if type(rep) is not int or rep < 1 or type(seed) is not int:
        raise ValueError('bootstrap配置错误')
    if not isinstance(qs, list) or not qs or any(finite(q) <= 0 or finite(q) >= 1 for q in qs):
        raise ValueError('分位数配置错误')
    clips = value.get('clipping', [])
    if any(finite(c) <= 0 for c in clips):
        raise ValueError('截断阈值必须为正数')
    ess_min = finite(value.get('ess_min', 2))
    min_sites = value.get('min_sites', 3)
    if ess_min < 0 or type(min_sites) is not int or min_sites < 1:
        raise ValueError('诊断阈值配置错误')
    return {'replicates': rep, 'seed': seed, 'quantiles': sorted(set(qs)),
            'clipping': list(clips), 'ess_min': ess_min, 'min_sites': min_sites}


def _cohort(value, h):
    """硬校验全部回合的概率表与所选动作，并把回合分为完整/截断/缺步三类。

    概率分布不合法、动作空间不一致、已选动作行为概率非正属于硬错误，直接抛ValueError；
    t不连续、步数与end标记矛盾等结构异常记为缺步（defective），单列并阻止正式估计。
    """
    complete, truncated, defective = [], [], []
    for ep in value['episodes']:
        eid = ep['episode_id']
        site = ep['site']
        if not isinstance(site, str) or not site:
            raise ValueError('site必须为非空字符串')
        steps = ep['steps']
        end = ep.get('end')
        if not isinstance(steps, list):
            raise ValueError('steps必须是数组')
        bad = None
        if end not in ('terminal', 'horizon', 'truncated'):
            bad = '未知结束标记'
        elif not steps:
            bad = '空步骤'
        elif len(steps) > h:
            bad = '步数超出H'
        for t, row in enumerate(steps):
            distribution(row['behavior'])
            distribution(row['target'])
            if row['behavior'].keys() != row['target'].keys():
                raise ValueError('行为与目标动作空间不一致')
            action = row.get('action')
            if action not in row['behavior'] or row['behavior'][action] <= 0:
                raise ValueError('已选动作缺失或行为概率非严格为正')
            finite(row['reward'])
            if bad is None and (type(row.get('t')) is not int or row['t'] != t):
                bad = '步骤编号不连续或缺失'
        if bad is None and end == 'horizon' and len(steps) != h:
            bad = 'horizon必须恰好H步'
        if bad:
            defective.append({'episode_id': eid, 'site': site, 'reason': bad})
        elif end == 'truncated':
            truncated.append({'episode_id': eid, 'site': site, 'observed_steps': len(steps)})
        else:
            complete.append(ep)
    return complete, truncated, defective


def _support(complete):
    """支持域检查：每个已观察历史的全部合法动作，目标正概率而行为零概率即违背。"""
    violations = []
    checked = 0
    for ep in complete:
        for row in ep['steps']:
            checked += 1
            for action, tp in row['target'].items():
                if tp > 0 and row['behavior'][action] == 0:
                    violations.append({'episode_id': ep['episode_id'], 'site': ep['site'],
                                       't': row['t'], 'action': action, 'target_probability': tp})
    return checked, violations


def _weights(complete, h):
    """完整回合的对数累积权重与奖励序列；提前结束按吸收态补齐：奖励0、比值1（对数0）。"""
    rows = []
    for ep in complete:
        logs, rewards = [], []
        acc = 0.0
        for t in range(h):
            if t < len(ep['steps']):
                row = ep['steps'][t]
                action = row['action']
                acc += _log(row['target'][action]) - _log(row['behavior'][action])
                r = row['reward']
            else:
                r = 0.0
            logs.append(acc)
            rewards.append(r)
        rows.append({'episode_id': ep['episode_id'], 'site': ep['site'],
                     'log_rho': logs, 'reward': rewards})
    return rows


def _step(logw, rewards, n):
    """单时点统计：缩放空间计算权重和、最大权重占比、ESS与两个估计的分子。

    零权重和时该步不可估（不能定义为0收益）。max权重占比=1/sum(缩放权重)，
    ESS=(sum w)^2/sum w^2，WPDIS分子分母同除exp(m)后不溢出，PDIS可能溢出为inf。
    """
    m = max(logw)
    if m == -math.inf:
        return {'estimable': False}
    sw = [_xexp(x - m) for x in logw]
    s1 = sum(sw)
    s2 = sum(w * w for w in sw)
    num = sum(w * r for w, r in zip(sw, rewards))
    return {'estimable': True,
            'weight_sum': _xexp(m) * s1,
            'max_weight_share': 1.0 / s1,
            'ess': s1 * s1 / s2,
            'pdis': _xexp(m) * num / n if num else 0.0,
            'wpdis': num / s1}


def _estimate(rows, idx, h, gamma, clip=None):
    """对回合索引idx（可重复，供bootstrap副本）计算PDIS/WPDIS与每时点诊断。

    截断为每时点累积rho独立min(rho,c)，不递推；在对数空间取min(log rho,log c)。
    """
    logc = _log(clip) if clip is not None else None
    n = len(idx)
    per_step = []
    pdis = wpdis = 0.0
    ok = True
    for t in range(h):
        lw = [rows[i]['log_rho'][t] for i in idx]
        if logc is not None:
            lw = [min(x, logc) for x in lw]
        rs = [rows[i]['reward'][t] for i in idx]
        st = _step(lw, rs, n)
        if not st['estimable']:
            ok = False
            per_step.append({'t': t, 'estimable': False, 'reason': '零权重和'})
            continue
        g = gamma ** t
        pdis += g * st['pdis']
        wpdis += g * st['wpdis']
        per_step.append({'t': t, 'estimable': True, 'n': n, 'discount': g,
                         'weight_sum': st['weight_sum'],
                         'max_weight_share': st['max_weight_share'], 'ess': st['ess'],
                         'pdis': st['pdis'], 'wpdis': st['wpdis']})
    return {'estimable': ok, 'per_step': per_step, 'pdis': pdis, 'wpdis': wpdis}


def _quantile(values, q):
    """排序样本的线性插值分位数。"""
    xs = sorted(values)
    m = len(xs)
    if m == 1:
        return xs[0]
    pos = q * (m - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, m - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _bootstrap(rows, h, gamma, cfg):
    """site整群bootstrap：有放回抽取原site数，重复site的整组回合重复计入副本。

    每个副本独立重算WPDIS分母；同一副本内原始与各截断变体共用同一抽取（配对敏感性）。
    返回(sites, {clip或None: {'pdis':[...], 'wpdis':[...], 'failed':int}})。
    """
    sites = sorted({r['site'] for r in rows})
    by_site = {s: [i for i, r in enumerate(rows) if r['site'] == s] for s in sites}
    rng = random.Random(cfg['seed'])
    variants = [None] + list(cfg['clipping'])
    draws = {v: {'pdis': [], 'wpdis': [], 'failed': 0} for v in variants}
    for _ in range(cfg['replicates']):
        picked = rng.choices(sites, k=len(sites))
        idx = [i for s in picked for i in by_site[s]]
        for v in variants:
            est = _estimate(rows, idx, h, gamma, clip=v)
            if est['estimable'] and math.isfinite(est['pdis']) and math.isfinite(est['wpdis']):
                draws[v]['pdis'].append(est['pdis'])
                draws[v]['wpdis'].append(est['wpdis'])
            else:
                draws[v]['failed'] += 1
    return sites, draws


def _scrub(node, diags, path):
    """递归把不可表示的浮点（inf）置为None并记录诊断，保证JSON可序列化。"""
    if isinstance(node, float) and not math.isfinite(node):
        diags.append('数值不可表示已置空:' + path)
        return None
    if isinstance(node, dict):
        return {k: _scrub(v, diags, path + '.' + str(k)) for k, v in node.items()}
    if isinstance(node, list):
        return [_scrub(v, diags, path + '[' + str(i) + ']') for i, v in enumerate(node)]
    return node


def report(value, digest=None, source=None):
    """生成完整分析报告；硬校验失败抛ValueError，可估性问题进blockers/diagnostics。"""
    cfg = _config(value)
    h = value['H']
    gamma = finite(value['gamma'])
    if type(h) is not int or h <= 0 or not 0 <= gamma <= 1:
        raise ValueError('长度或折扣错误')
    unique(value['episodes'], 'episode_id')
    complete, truncated, defective = _cohort(value, h)
    checked, violations = _support(complete)
    canonical = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False).encode('utf8')).hexdigest()
    diags = []
    blockers = []
    if defective:
        blockers.append({'reason': 'defective_episodes', 'count': len(defective),
                         'note': '缺步或结构异常回合不当作提前结束，单列并阻止完整队列估计'})
    if truncated:
        blockers.append({'reason': 'truncated_episodes', 'count': len(truncated),
                         'note': '截断回合不当作提前结束，单列并阻止完整队列估计'})
    if violations:
        blockers.append({'reason': 'support_violation', 'count': len(violations),
                         'note': '已观察历史存在目标正概率而行为零概率的动作，停止正式估计'})
    if not complete:
        blockers.append({'reason': 'no_complete_episodes', 'count': 0,
                         'note': '没有可用于估计的完整回合'})

    estimates = sensitivity = intervals = None
    if not blockers:
        rows = _weights(complete, h)
        idx_all = list(range(len(rows)))
        raw = _estimate(rows, idx_all, h, gamma)
        zero_steps = [s['t'] for s in raw['per_step'] if not s['estimable']]
        if zero_steps:
            blockers.append({'reason': 'zero_weight_sum', 'count': len(zero_steps),
                             'note': '时点' + str(zero_steps) + '权重和为零，该步估计与ESS不可估，不能定义为0收益'})
        low_ess = [{'t': s['t'], 'ess': s['ess'], 'ess_min': cfg['ess_min']}
                   for s in raw['per_step'] if s['estimable'] and s['ess'] < cfg['ess_min']]
        if low_ess:
            blockers.append({'reason': 'degenerate_weights', 'count': len(low_ess),
                             'note': '权重退化ESS低于阈值:' + json.dumps(low_ess, ensure_ascii=False)})
        if not (math.isfinite(raw['pdis']) and math.isfinite(raw['wpdis'])):
            blockers.append({'reason': 'non_finite_estimate', 'count': 1,
                             'note': '点估计数值超出float64可表示范围'})
        if not blockers:
            estimates = {'pdis': {'value': raw['pdis']}, 'wpdis': {'value': raw['wpdis']},
                         'per_step': raw['per_step']}
            sensitivity = []
            for c in cfg['clipping']:
                est = _estimate(rows, idx_all, h, gamma, clip=c)
                for s in est['per_step']:
                    if s['estimable'] and s['ess'] < cfg['ess_min']:
                        diags.append('截断阈值' + str(c) + '下t=' + str(s['t']) +
                                     '的ESS=' + format(s['ess'], '.6g') + '低于' +
                                     str(cfg['ess_min']) + '（有偏敏感性，ESS按截断后权重计算）')
                sensitivity.append({'threshold': c, 'biased': True,
                                    'estimable': est['estimable'],
                                    'pdis': {'value': est['pdis']},
                                    'wpdis': {'value': est['wpdis']},
                                    'per_step': est['per_step']})
            sites = sorted({r['site'] for r in rows})
            if len(sites) < cfg['min_sites']:
                intervals = {'estimable': False,
                             'reasons': ['site数' + str(len(sites)) + '少于min_sites=' +
                                         str(cfg['min_sites']) + '，整群bootstrap区间不可估'],
                             'sites': len(sites)}
                diags.append(intervals['reasons'][0])
            else:
                sites, draws = _bootstrap(rows, h, gamma, cfg)
                variants = []
                for v, d in draws.items():
                    used = len(d['pdis'])
                    if d['failed']:
                        diags.append('bootstrap变体clip=' + str(v) + '有' + str(d['failed']) +
                                     '/' + str(cfg['replicates']) + '个副本不可估已剔除')
                    if not used:
                        variants.append({'clip': v, 'biased': v is not None,
                                         'estimable': False, 'failed_replicates': d['failed']})
                        continue
                    variants.append({'clip': v, 'biased': v is not None, 'estimable': True,
                                     'failed_replicates': d['failed'],
                                     'pdis': {str(q): _quantile(d['pdis'], q) for q in cfg['quantiles']},
                                     'wpdis': {str(q): _quantile(d['wpdis'], q) for q in cfg['quantiles']}})
                intervals = {'estimable': True, 'method': 'site-cluster-bootstrap-percentile',
                             'sites': len(sites), 'replicates': cfg['replicates'],
                             'seed': cfg['seed'], 'variants': variants}

    out = {'report': 'offline-policy-evaluation',
           'method': 'PDIS逐决策重要性采样；WPDIS按每时点权重和归一；提前结束回合按吸收态补齐(奖励0、比值1)并保留在各时点分母中',
           'source': {'path': source, 'sha256': digest},
           'input': {'canonical_sha256': canonical, 'H': h, 'gamma': gamma,
                     'episodes_total': len(value['episodes']), 'complete': len(complete),
                     'sites': sorted({ep['site'] for ep in complete}),
                     'config': {'bootstrap': {'replicates': cfg['replicates'], 'seed': cfg['seed'],
                                              'quantiles': cfg['quantiles']},
                                'clipping': cfg['clipping'],
                                'ess_min': cfg['ess_min'], 'min_sites': cfg['min_sites']}},
           'scope_note': SCOPE_NOTE,
           'cohort': {'complete': len(complete), 'truncated': truncated, 'defective': defective},
           'support': {'histories_checked': checked, 'violations': violations},
           'status': 'ok' if estimates is not None else 'blocked',
           'blockers': blockers,
           'estimates': _scrub(estimates, diags, 'estimates') if estimates else None,
           'clipping_sensitivity': _scrub(sensitivity, diags, 'clipping') if sensitivity else None,
           'intervals': _scrub(intervals, diags, 'intervals') if intervals else None,
           'diagnostics': diags}
    return out
