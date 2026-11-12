# 历史动作日志的序列策略价值与支持域诊断

这是现场机器人应用分析的合成资料项目。`data`保存业务记录，`analysis_data.io`负责严格JSON读取和原始字节指纹，`analysis_data.domain`提供已有的基础汇总。字段与数学口径见[资料约定](docs/data-contract.md)。所有厂区、设备、收益与结果均为合成，不代表任何实际项目的效果。

Python 3.11及以上版本可直接运行，当前无第三方Python依赖。项目根目录中的常用命令：

```sh
python -m unittest discover -s tests -v
python -m compileall -q analysis_data tests
python -m analysis_data.cli inspect data/normal.json
python -m analysis_data.cli ope data/normal.json
```

`inspect`展示原始资料摘要与基础统计。`ope`是离线策略价值分析器：只评估给定概率表，不训练策略也不向设备发动作。它输出逐决策重要性采样PDIS与按每时点归一的WPDIS、每时点权重和/最大权重占比/ESS、截断权重的有偏敏感性（ESS按截断后权重重算），以及site整群bootstrap百分位区间（固定种子，重复site整组复制且每副本重算分母）。提前结束的完整回合按吸收态补齐到H（奖励零、比值1）并留在各时点分母中；截断或缺步回合单列并阻止正式估计；任一已观察历史出现目标正概率而行为零概率的动作即报告支持域违背并停止正式估计，且通过检查不代表未到达历史的全局可识别性。`analysis_data.enumeration`用有限树完整路径枚举给出精确期望，核对无截断估计。所有结果绑定原始文件指纹与解析后输入的规范摘要指纹。

结果写入其他目录时应保留输入来源指纹，原始资料不覆盖。更多边界资料与normal采用同样的JSON结构，invalid前缀文件保存不合法输入。

容器可执行 `docker build -t field-analysis .` 后用 `docker run --rm field-analysis` 查看资料，或 `docker run --rm field-analysis python -m unittest discover -s tests -v` 运行现有检查。运行期间无外网或账号要求。
