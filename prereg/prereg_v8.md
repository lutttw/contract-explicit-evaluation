# prereg v8 — judge 生态确认扩展 + unanimity gate 组件化(骨架,2026-11-17)

**来源**:APIN 面板评审(5/5 Major Revision)路线图 E 组与 R1-F8/R4/R5 的判决条件;
**冻结要求**:本文档在任何 v8 数据计算前签署;判据改动走 amendment。

## 0. 口径继承
v7 全部 conventions(E1)生效;judge 生态对象=battle(49,938),合同轴=judge identity
(17 judges);α_port=0.20,SPLIT_CRIT=0.60,δ=0.10 不变。

## 1. 确认性分析(C 系列)

- **C1 两层证书判据化**(来源 R1-F8/R4-C3):margin 分层(cut=0.30,预声明;敏感性
  0.2/0.4 进补充)的 certified 层 share 与 CP 界按 v7 census 流程重算;
  判据:certified 层 CP ≤ 0.10 且 block-DEFF ≤ 1.2;
- **C2 validity 轴**(来源 R4-F1/R5-A3):contested 分解
  judge-error(人类标签清晰而 judge 分裂)/human-ambiguous(gt=tie);
  判据:judge-error 占比报告 + certified 层的 human-clear 内争议率 ≤ 0.05;
- **C3 unanimity gate 组件化**(来源 R5 必要性真空/E1 解剖、R3-F15):
  κ=1 门作为正式提交规则与 learned+gate 同台;判据:J5'/J6'/J7' 同表双侧报告,
  简化性发现(v6 解剖:20/23 conf=1.000)作为先验声明入读出表;
- **C4 逐 judge accuracy 表**(来源 R4-F1):每 judge 对 human-clear 的 accuracy +
  Kα(3 类 0.456/方向 0.504 基线,v7 已测)重测;
- **C5 回溯探针**(来源 R3-F5):frozen v1 registry 的 10 个 threshold 轴翻转
  按 decode-calibration probe 回溯;判据:仅 dual-decode-stable 翻转入 v8 registry;
  **执行状态(2026-11-17)=EXPLORATORY 已执行**(e3_final.py,v7_e3_probe.json):
  atlop 系 9 翻转中 7 个 resolve(prior-matched 与某一注册解码同侧=解码选择效应)、
  2 个 PERSIST(atlop|dreeam 对,第三解码真实不稳定)、dreeam|kddocre 点符号无翻转
  (bootstrap 分布型);v8 动作=将 dual-decode-stable 子集升级为确认判据;
  列映射定谳=重训 npy col c ↔ stored c−1(e3_mapping_test.py:native 0.1748
  /sig05 0.0216 双双精确命中登记 per-run 值)
  **执行诊断(2026-11-17)**:全部 44 run 的原始 logits 可得(RV5 flat npy 97 列 +
  RECOVERED logits_evaluation 目录 96 列三源公共空间);doc 级对齐已打通
  (e3_selftest.py keys→canonical 全命中);**未决=关系/实体列空间**
  (直连/−1 偏移/DocRED-official perm 三种均得 gap≈0,vs 登记值 0.1748)——
  首任务=溯源 RV5 flat npy 的导出器列约定;KD 导出器(evallever kd_infer_score.py)
  的 perm 模板可直接套用;
- **C6 block/judge-panel bootstrap 预声明**(来源 R2-F2):census 报 i.i.d. CP +
  block q90 + panel q90 三数(0.124/0.124/0.143 为 v7 基线)。
- **C7 家族块 bootstrap**(来源 R3-F14):v2/v3 Fisher 对比(43/43 vs 8/24 等)
  按生成器家族(14 折)块重抽样,报告 nominal + cluster-robust p 双列;
  数据需求=逐单元 correctness 旗标(p1_calc 重放管线已具),v8 执行。

## 2. 前瞻变体
- split-sample 仅当 judge 子集满足独立性预筛(per-judge accuracy 相关系数矩阵
  谱隙 ≥ 2)时运行;否则登记为 census-only。

## 3. 失败点(预声明)
- F1:C1 层 CP > 0.10 ⇒ 两层证书降级回探索层;
- F2:C2 judge-error 占比 > 0.70 ⇒ concordance 主张撤回,重定位为 routing 设备;
- F3:C5 双解码下 <50% 翻转存活 ⇒ v1 registry 判据集重审(影响 Part 1-2 表)。

## 4. 产物
v8_verdict.json;per-stratum/per-judge 逐实例 JSON;amendment 续档。

- **E7 rank-stability =EXPLORATORY 已执行**(2026-11-17,v7_rank_stability.json,
  hf-mirror 元数据通道):BT certified 层(34,269)vs 全量(49,938)
  Kendall ρ=0.9977;top-10 保序 10/10;top-20 内最大变动 2 位;≥5 位 0/64;
  认证层头名 gpt-4-1106-preview(常识一致)。leaderboard 级结论:
  剔除 hedge 层不改变任何实质排名——两层证书在排序层零成本。
  v8 动作=rank-stability 与 validity 分解同列确认判据。
