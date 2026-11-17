# prereg v5 — 协议修正与精确界周期(约定清单 + exact weighted-distribution bound)

**状态:FROZEN(2026-11-15,作者批准;会话指令"启动V5")**
冻结发生于查看任何 v5 结果之前;设计输入=prereg_v5_candidates.md
(DRAFT 候选清单,2026-11-15)+ prereg v4 全部产物(pilot disclosure)。
v4 周期(Phase 0-3+判读+修正案 1-6)全部保留;v5 不推翻 v4 判决。

---

## 0. 范围(冻结)

本周期只含两个必选候选 + 一个生长条款;候选 4(第二测试床)与
候选 5(judge 轴)**不在本周期**(单独立项评估)。

- **E1(候选 3,协议修正)**:Declare 步"约定清单"义务化。
- **E2(候选 2,理论+实现)**:exact weighted-distribution 界,
  附预声明判据。
- **G1(候选 1,生长条款,条件启动)**:x2 内 +20 run 至 n=44,
  **仅当 E2 通过判据**时启动(理由:E2 失败则 44 模型的加权证书
  无意义,拆样本 census 证书单独不足以justify GPU);GPU 窗口
  排在 P2 主表收口之后,健康门全套沿 v4。

## 1. E1:约定清单(Declare 步协议修正)

读数表 schema 增设 `conventions` 块,发布义务:

```json
"conventions": {
  "cnt_layout": "p3io.CNT",            # 计数键→槽位映射
  "file_generation": "per_doc|flat",   # 导出代际
  "gold_assignment": "own|unified",    # score/thr 维金标指派
  "variant_keys": ["fixed_0.5", "cal_sigmoid_multilabel"],
  "reading_col_map": {"score_A": 3, "score_R": 1, "gap": 2}
}
```

执行:现有读数表(x2_stratum_24.json)以修正案补发 conventions 块;
此后一切读数表无 conventions 视为不合规(管线 assert)。
**判据(定性)**:补发后 v4 Phase 2/3 数字逐位不变(约定未漂移,
仅显式化);若任何数字变动,按 v4 修正案 6 同款流程登记并重判。

## 2. E2:exact weighted-distribution 界

### 2.1 语义设定(冻结)

单元=registry 检查项,折=cal/test 各半。权重:cell 级 transductive
密度比 w_c = (test 频率+λ)/(cal 频率+λ),λ=0.5,cap 5(v4 C5w 同款)。
风险量:加权错误率 R_w = Σ w_c y_c / Σ w_c(选中集)。

### 2.2 精确界(冻结的实现)

**关键观察**:选中集的单元来自少数 distinct cells(m ≤ 57);
同 cell 的单元共享同一权重 w_c。因此加权错误计数 W = Σ_c w_c·b_c,
其中 b_c = cell c 内选中错误数。对固定权重向量,精确分布由
**卷积/DP 枚举**:P(W ≤ w) = Σ Π Binomial(b_c; n_c, p_c) 在
p_c 的联合后验下离散化(保守:每个 cell 用 Clopper-Pearson 上界
p̄_c = Beta(1−δ/m; b_c+1, n_c−b_c),m=cells,Bonferroni)。
**证书条件**:存在阈值 t 使
  Σ Π Binomial PMF 在 R_w ≤ α 下累计 ≥ 1−δ。
实现:src/v5_exact_bound.py(DP 卷积,m ≤ 57 可枚举);
MC 验证(v4 gate2b 同款测试床:B=2000,真风险口径)。

### 2.3 预声明判据(冻结)

**PASS**:硬池(7 冻结族)gen-block MC 下,加权精确界证书
  exceed(true)≤ 0.10 **且** mean commits ≥ 9.0(MP 的 95%);
**FAIL 分支 a(界仍松)**:exceed 过但 commits < 9.0 → 界改进
  不足以换覆盖,报告精确界 vs Hoeffding 的 gap 量化,候选判死;
**FAIL 分支 b(有效性破)**:exceed > 0.10 → 加权精确界无效,
  理论诊断入稿,候选判死。
**预声明败点**:与 v4 F1-F5 同款适用;另加 F6:DP 卷积在 m=57
  数值溢出/下溢 → 对数域实现失败 → 如实报告计算障碍。

### 2.4 G1 条件启动

E2 PASS → +20 run(种子 20261202-20261211,2 臂 2 比各 5 种子)
→ x2 n=44 → 拆样本 census(T1 前瞻:n=22 校准折零坏)。
E2 FAIL → G1 不启动,GPU 不动。

## 3. 判读表(冻结)

| 分支 | 条件 | 结果形态 |
|---|---|---|
| V5-1 | E1✓ + E2 PASS + G1 完成且 T1 前瞻证书成 | 合并稿 Part 3 升格:前瞻证书+加权界入方法 |
| V5-2 | E1✓ + E2 PASS + G1 T3/T4 | 升格但证书降级交付(与 v4 C2 同款条款) |
| V5-3 | E1✓ + E2 FAIL(a 或 b) | 约定清单入协议;精确界阴性入失败谱系;稿不动 |
| V5-4 | E1 数字变动 | 修正案重判 v4 受影响判据(未预期分支,最高优先排查) |

## 4. 止损与纪律

- 止损门:沿 v4(flips≥35 / census_n≥22)+ E1 合规(conventions 在);
- 修正案通道:冻结后一切偏离先登记;
- 代理分数禁用;预写段落等数字落地后重写(v4 教训);
- E2 判据先行:实现前本文件已含判据,禁止看结果改判据。

## 5. 产物

- out/x2_v4/x2_stratum_24.json(conventions 补发版)
- src/v5_exact_bound.py + out/v5_exact_bound.json(E2)
- G1 启动时的 out/x2_v5/(+20 run)
- prereg_amendments.md 逐条登记
