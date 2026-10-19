# 严格 AVG 与端到端 AVG

2026-10-19 增加报告版本 `strict-plus-e2e-v1`。actor、judge、逐题判分和历史严格 `AVG` 不变，新增指标用于对包含未作答题的完整运行进行补充比较。

对题型 k，`E2E_Accuracy(k) = 已评分正确题数 / 该类全部题数`。只允许将状态为 `actor_failed` 且分数为 null 的未作答题计为 0；正常答错仍按原来的 0 分处理。六类都存在且其余题正常评分时，`E2E_AVG = sum(E2E_Accuracy(k)) / 6`。这是宏平均，不将所有题混在一起算微平均，也不删除异常题后重新计算分母。

`abnormal_rate = 原异常题数 / 原全部题数`，包含未作答和评分异常。`counts.unanswered_as_zero` 标明多少题在 E2E 中补为 0；`counts.e2e_unscorable` 标明多少题仍无法判分。`e2e_type_accuracy` 为补充的分题型结果，严格 `type_accuracy` 保持原规则。

judge 失败、非法或缺失分数、未知题型和缺失题型不能变成答错题；有这些问题的 E2E 仍为 null。连接断开等基础设施异常继续沿用原流程停止 replay；历史补算要求 `progress.status == completed` 且题数与已完成数、行数一致，不能把尚未执行的题或中断的部分结果包装成完整成绩。

未来运行的 `report.json`、`summary.json` 的比较字段、`REPORT.md` 及轨迹汇总同时携带严格 AVG、E2E AVG 和异常率。节点有效性、UCT 回传及 Pareto 继续使用原严格口径；本次是追加报告指标，没有切换选择目标。

在超算项目根目录补算已有运行：

```sh
.venv/bin/python scripts/report_e2e.py \
  --run runs/full-v03-local-147547/result \
  --output runs/e2e-147547-20261019
```

只读 train/validation 的已有逐题记录，不读取测试集，不调用模型。输出目录须全新并位于原归档之外。新的 `summary.json` 保存数值报告、实现 commit、源报告与逐题文件的 SHA256；`REPORT.md` 给出比较表。源题目与回答不会复制到补充报告，原报告和归档校验不覆盖。
