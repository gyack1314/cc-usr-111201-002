"""核对离线策略价值分析：枚举对账、提前结束、长权重链、零支持、整群重抽与可复现性。"""
import copy
import math
import unittest
from analysis_data.io import load
from analysis_data.domain import summarize
from analysis_data.ope import report, _cohort, _weights, _estimate, _bootstrap, _config
from analysis_data.enumeration import expectation


def _rep(path):
    value, digest = load(path)
    return report(value, digest, source=path)


class EnumerationTests(unittest.TestCase):
    """小状态树完整路径枚举给出精确期望，独立核对无截断估计。"""

    def test_tree_exact_expectations(self):
        tree = load('data/finite-tree-reference.json')[0]
        self.assertAlmostEqual(expectation(tree, 'target'), tree['target_expected_return'], places=12)
        self.assertAlmostEqual(expectation(tree, 'target'), 1.2, places=12)
        self.assertAlmostEqual(expectation(tree, 'behavior'), 1.4, places=12)

    def test_incomplete_tree_rejected(self):
        tree = copy.deepcopy(load('data/finite-tree-reference.json')[0])
        tree['nodes'] = tree['nodes'][:2]
        with self.assertRaises(ValueError):
            expectation(tree, 'target')

    def test_normal_matches_enumeration(self):
        rep = _rep('data/normal.json')
        self.assertEqual(rep['status'], 'ok')
        exact = expectation(load('data/finite-tree-reference.json')[0], 'target')
        self.assertAlmostEqual(rep['estimates']['pdis']['value'], exact, places=12)
        self.assertAlmostEqual(rep['estimates']['wpdis']['value'], exact, places=12)
        rows = summarize(load('data/normal.json')[0])
        self.assertAlmostEqual(sum(r['observed_return'] for r in rows) / len(rows), 1.4, places=12)

    def test_normal_per_step_diagnostics(self):
        rep = _rep('data/normal.json')
        steps = {s['t']: s for s in rep['estimates']['per_step']}
        self.assertTrue(all(s['n'] == 12 for s in steps.values()))
        self.assertAlmostEqual(steps[0]['weight_sum'], 12.0, places=12)
        self.assertAlmostEqual(steps[0]['max_weight_share'], 0.125, places=12)
        self.assertAlmostEqual(steps[0]['ess'], 9.6, places=12)
        self.assertAlmostEqual(steps[1]['weight_sum'], 12.0, places=12)
        self.assertAlmostEqual(steps[1]['max_weight_share'], 2.25 / 12, places=12)
        self.assertAlmostEqual(steps[1]['ess'], 7.68, places=12)
        self.assertAlmostEqual(steps[2]['ess'], 7.68, places=12)  # 第三步仅吸收，权重同第二步


class EarlyTerminalTests(unittest.TestCase):
    """提前结束的完整回合按吸收态补齐到H：奖励0、比值1，且不从分母消失。"""

    def test_absorbing_padding(self):
        value, _ = load('data/early-terminal.json')
        complete, truncated, defective = _cohort(value, 3)
        self.assertEqual((len(complete), truncated, defective), (12, [], []))
        rows = {r['episode_id']: r for r in _weights(complete, 3)}
        one = rows['S1-left-left']  # 仅1步，t=1,2为吸收态
        self.assertEqual(one['reward'], [1, 0.0, 0.0])
        self.assertAlmostEqual(one['log_rho'][0], math.log(1.5), places=15)
        self.assertAlmostEqual(one['log_rho'][2], math.log(1.5), places=15)  # 后续比值1

    def test_padded_episode_stays_in_denominators(self):
        rep = _rep('data/early-terminal.json')
        self.assertEqual(rep['status'], 'ok')
        steps = {s['t']: s for s in rep['estimates']['per_step']}
        self.assertTrue(all(s['n'] == 12 for s in steps.values()))
        self.assertAlmostEqual(steps[1]['weight_sum'], 11.25, places=12)  # 含补齐回合的1.5
        self.assertAlmostEqual(steps[2]['weight_sum'], 11.25, places=12)
        self.assertAlmostEqual(rep['estimates']['pdis']['value'], 1.2, places=12)
        self.assertAlmostEqual(rep['estimates']['wpdis']['value'], 1.23, places=12)
        self.assertAlmostEqual(steps[1]['wpdis'], 6 / 11.25, places=12)


class BlockedCohortTests(unittest.TestCase):
    """截断、缺步、零支持与非法概率的处理。"""

    def test_truncated_blocks_and_listed(self):
        rep = _rep('data/truncated.json')
        self.assertEqual(rep['status'], 'blocked')
        self.assertIsNone(rep['estimates'])
        self.assertIn('truncated_episodes', [b['reason'] for b in rep['blockers']])
        self.assertEqual([x['episode_id'] for x in rep['cohort']['truncated']], ['S1-left-left'])

    def test_missing_steps_defective(self):
        value, _ = load('data/normal.json')
        gap = copy.deepcopy(value)
        gap['episodes'][1]['steps'][1]['t'] = 5  # t缺口：缺步而非提前结束
        rep = report(gap, 'x')
        self.assertEqual(rep['status'], 'blocked')
        self.assertEqual(rep['cohort']['defective'],
                         [{'episode_id': 'S1-left-right', 'site': 'S1', 'reason': '步骤编号不连续或缺失'}])
        bad_end = copy.deepcopy(value)
        bad_end['episodes'][0]['end'] = 'horizon'  # 2步≠H=3
        rep2 = report(bad_end, 'x')
        self.assertEqual(rep2['cohort']['defective'][0]['reason'], 'horizon必须恰好H步')

    def test_support_violation_checks_every_legal_action(self):
        rep = _rep('data/unsupported-history.json')
        self.assertEqual(rep['status'], 'blocked')
        self.assertIn('support_violation', [b['reason'] for b in rep['blockers']])
        v = rep['support']['violations']
        self.assertEqual(len(v), 1)
        self.assertEqual((v[0]['episode_id'], v[0]['t'], v[0]['action']), ('S1-left-left', 0, 'right'))
        self.assertAlmostEqual(v[0]['target_probability'], 0.25, places=15)
        # 所选动作left的行为概率为1>0，违背只在未选动作上：只查所选动作会漏报
        self.assertIn('未到达历史', rep['scope_note'])

    def test_invalid_probability_raises(self):
        value, digest = load('data/invalid-probability.json')
        with self.assertRaises(ValueError):
            report(value, digest)

    def test_degenerate_weights_not_estimable(self):
        ep = {'episode_id': 'e1', 'site': 'S1', 'end': 'terminal', 'steps': [
            {'t': 0, 'action': 'a', 'behavior': {'a': 0.5, 'b': 0.5},
             'target': {'a': 0.9, 'b': 0.1}, 'reward': 1}]}
        value = {'H': 2, 'gamma': 0.9, 'episodes': [ep], 'ess_min': 2, 'min_sites': 1,
                 'bootstrap': {'replicates': 10, 'seed': 1, 'quantiles': [0.025, 0.975]}}
        rep = report(value, 'x')
        self.assertEqual(rep['status'], 'blocked')
        self.assertIn('degenerate_weights', [b['reason'] for b in rep['blockers']])


class LongChainTests(unittest.TestCase):
    """极小概率长链连乘：朴素连乘溢出，对数空间保持精确。"""

    def _chain(self):
        h = 104
        steps = []
        for t in range(h):
            steps.append({'t': t, 'action': 'a',
                          'behavior': {'a': 1e-6, 'b': 1 - 1e-6},
                          'target': {'a': 1.0, 'b': 0.0} if t < 52 else {'a': 1e-12, 'b': 1 - 1e-12},
                          'reward': 1 if t == h - 1 else 0})
        return {'H': h, 'gamma': 1, 'ess_min': 1, 'min_sites': 1,
                'episodes': [{'episode_id': 'e1', 'site': 'S1', 'end': 'horizon', 'steps': steps}],
                'bootstrap': {'replicates': 10, 'seed': 1, 'quantiles': [0.025, 0.975]}}

    def test_log_space_product(self):
        value = self._chain()
        self.assertEqual(math.prod([1e6] * 52), math.inf)  # 朴素连乘在半途已溢出
        complete, _, _ = _cohort(value, 104)
        row = _weights(complete, 104)[0]
        self.assertAlmostEqual(row['log_rho'][51], 52 * math.log(1e6), places=9)
        self.assertAlmostEqual(row['log_rho'][103], 0.0, places=9)  # 先升后降精确回到1
        rep = report(value, 'x')
        self.assertEqual(rep['status'], 'ok')
        self.assertAlmostEqual(rep['estimates']['pdis']['value'], 1.0, places=9)
        self.assertAlmostEqual(rep['estimates']['wpdis']['value'], 1.0, places=9)
        steps = {s['t']: s for s in rep['estimates']['per_step']}
        self.assertIsNone(steps[51]['weight_sum'])  # 1e312不可表示，置空并诊断
        self.assertTrue(any('weight_sum' in d for d in rep['diagnostics']))
        self.assertAlmostEqual(steps[103]['weight_sum'], 1.0, places=9)


class BootstrapTests(unittest.TestCase):
    """site整群bootstrap：固定种子可复现，重复site整组复制且每副本重算分母。"""

    def test_reproducible_with_fixed_seed(self):
        value, digest = load('data/early-terminal.json')
        self.assertEqual(report(value, digest), report(value, digest))

    def test_identical_sites_degenerate_interval(self):
        rep = _rep('data/normal.json')  # 三个site组成完全相同，副本估计恒为1.2
        iv = rep['intervals']
        self.assertTrue(iv['estimable'])
        raw = iv['variants'][0]
        self.assertIsNone(raw['clip'])
        for q in ('0.025', '0.975'):
            self.assertAlmostEqual(raw['pdis'][q], 1.2, places=12)
            self.assertAlmostEqual(raw['wpdis'][q], 1.2, places=12)
        self.assertEqual(raw['failed_replicates'], 0)

    def test_duplicate_site_recounts_group_and_denominator(self):
        value, _ = load('data/early-terminal.json')
        complete, _, _ = _cohort(value, 3)
        rows = _weights(complete, 3)
        idx = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7]  # 抽中S1两次、S2一次：S1整组回合重复
        est = _estimate(rows, idx, 3, 0.9)
        step1 = est['per_step'][1]
        self.assertEqual(step1['n'], 12)
        self.assertAlmostEqual(step1['weight_sum'], 2 * 3.25 + 4.0, places=12)
        self.assertAlmostEqual(step1['wpdis'], 6 / 10.5, places=12)  # 分母按副本重算
        self.assertAlmostEqual(step1['pdis'], 0.5, places=12)

    def test_draws_deterministic(self):
        value, _ = load('data/early-terminal.json')
        complete, _, _ = _cohort(value, 3)
        rows = _weights(complete, 3)
        cfg = _config(value)
        first = _bootstrap(rows, 3, 0.9, cfg)
        second = _bootstrap(rows, 3, 0.9, cfg)
        self.assertEqual(first, second)

    def test_too_few_sites_interval_not_estimable(self):
        value, _ = load('data/normal.json')
        small = copy.deepcopy(value)
        small['episodes'] = [ep for ep in small['episodes'] if ep['site'] != 'S3']
        rep = report(small, 'x')
        self.assertEqual(rep['status'], 'ok')  # 点估计仍可估
        self.assertFalse(rep['intervals']['estimable'])
        self.assertTrue(any('区间不可估' in d for d in rep['diagnostics']))


class ClippingTests(unittest.TestCase):
    """截断权重敏感性：与原始结果分开，ESS按截断后权重重算。"""

    def test_sensitivity_separate_and_ess_recomputed(self):
        rep = _rep('data/normal.json')
        sens = {s['threshold']: s for s in rep['clipping_sensitivity']}
        self.assertEqual(set(sens), {2, 5, 20})
        clip2 = sens[2]
        self.assertTrue(clip2['biased'])
        steps = {s['t']: s for s in clip2['per_step']}
        self.assertAlmostEqual(steps[1]['weight_sum'], 11.25, places=12)  # 2.25被截为2
        self.assertAlmostEqual(steps[1]['ess'], 126.5625 / 15.5625, places=12)  # 截断后权重ESS
        self.assertNotAlmostEqual(steps[1]['ess'], 7.68, places=6)  # 不套用截断前ESS
        self.assertAlmostEqual(clip2['wpdis']['value'], 1.23, places=12)  # 有偏
        self.assertAlmostEqual(rep['estimates']['wpdis']['value'], 1.2, places=12)  # 原始不受影响
        for c in (5, 20):  # 阈值高于最大权重时与原始一致
            self.assertAlmostEqual(sens[c]['pdis']['value'], 1.2, places=12)
            self.assertAlmostEqual(sens[c]['wpdis']['value'], 1.2, places=12)
        clips = {v['clip'] for v in rep['intervals']['variants']}
        self.assertEqual(clips, {None, 2, 5, 20})  # 区间同样按变体分开


class BindingTests(unittest.TestCase):
    """结果绑定概率表与输入摘要。"""

    def test_source_and_canonical_fingerprint(self):
        value, digest = load('data/normal.json')
        rep = report(value, digest, source='data/normal.json')
        self.assertEqual(rep['source']['sha256'], digest)
        self.assertEqual(rep['input']['canonical_sha256'], rep['input']['canonical_sha256'])
        changed = copy.deepcopy(value)
        changed['episodes'][0]['steps'][0]['target'] = {'left': 0.7, 'right': 0.3}
        rep2 = report(changed, digest)
        self.assertNotEqual(rep['input']['canonical_sha256'], rep2['input']['canonical_sha256'])
        self.assertEqual(rep['input']['H'], 3)
        self.assertEqual(rep['input']['complete'], 12)
        self.assertEqual(rep['input']['sites'], ['S1', 'S2', 'S3'])


if __name__ == '__main__':
    unittest.main()
