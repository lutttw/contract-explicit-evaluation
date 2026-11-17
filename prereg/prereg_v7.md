# prereg v7 — judge 生态实例化(五步协议第二生态;17 judge × 49,938 battles)

**状态:FROZEN(2026-11-15 深夜,作者批准;会话指令"继续")**
冻结发生于查看任何 v7 结果之前;设计输入=v5_candidates 候选 5+v6 数据核查
(judge_data_check.json:17 judge × 49,938 battles 全覆盖交集,概率边际
prob_A/B/C 可用)+ prereg v4-v6 全部产物与判读(pilot disclosure)。
v4/v5/v6 判决全部保留;v7 不推翻任何已交付证书,只做**第二生态实例化**。

---

## 0. 对象映射(冻结;与 v4 五步同构)

| DocRE 生态 | judge 生态 |
|---|---|
| 实例 = 训练模型(24/44) | 实例 = **candidate battle**(49,938 个模型-vs-模型对决) |
| 契约轴 = 金标×指标×阈值 | 契约轴 = **judge 身份**(17 个 LLM judge;prompt/模板固定为数据集自带) |
| 读数 r_i(c) = 模型 i 的契约 gap | 读数 = judge c 对 battle i 的**偏好方向**(winner_pred ∈ {A,B,C};prob 边际可用) |
| flip_ij(d) = (i,j) 排序在契约 d 两端反转 | flip_ij(j_1,j_2) = judge j_1 与 j_2 对 battle i 的偏好不一致 |
| 分层 = 架构族 | 分层 = judge 家族(70B/open-7B/mixtral 等,探索性) |
| 定律 π=(1−τ)/2 | **同一恒等式,逐 judge 对成立**(τ=两 judge 偏好序的 Kendall τ) |

## 1. 五步在 judge 生态的实例化(冻结)

1. **Declare**:契约轴=judge 身份(17);battle 集与 ground truth 冻结
   (数据集自带;winner_gt 为平台标注,非本协议产出)。
2. **Measure**:逐 judge 对 (j_1,j_2):翻转密度 π(j_1,j_2)=P(两 judge 对
   同一 battle 偏好不一致),及 Kendall τ_j12(两 judge 偏好序的 τ);
   定律核对:π=(1−τ)/2(恒等式,internal-consistency 口径)。
3. **Scope**:两层——
   (a) **judge 对层**:τ_j12 高(一致)的 judge 对 = 可互认证;
   (b) **battle 层 census**:battle i 被判"争议"当且仅当 17 judge 中
   偏好多数率 ≤ 0.5+α_port/2(即接近半数分裂);bad battle 份额
   b/n 的 CP 上界 cp(b,n,δ=0.1)。
4. **Certify**:两层交付——
   (a) **judge 层 census**:随机抽 judge 对(bootstrap 1000)的
   一致率上界;**拆 judge 前瞻**:17 judge 排序后取偶位=校准 8/
   奇位=测试 9(预声明确定性切分);校准折定义"认证一致率"集,
   测试折 realize(与 v4/v6 拆样本同构,样本=judge);
   (b) **battle 层 census**:全 49,938 battle 的争议份额 CP 上界。
5. **Hedge**:争议 battle(多数率 ≤0.6)清单化,标注"judge 依赖
   排序,不可作为单一裁判依据"。

## 2. 判据(冻结)

- **J1 定律核对(内部一致性)**:136 个 judge 对的 (π, (1−τ)/2)
  偏差绝对值 ≤ 1e-9(恒等式,允许浮点容差);若违=实现错误旗标,
  全链停。非发现性,仅一致性门。
- **J2 层级复制(实质判据)**:judge 家族分层后,翻转密度序与
  探索性预期一致——**预声明方向**:70B 级 judge 两两一致率 ≥ 7B
  级(能力越近越一致);PASS = 方向全对(或 Spearman ≥0.8)。
  **注意**:此为方向性预言,预声明于数据查看前(v7 冻结时未跑任何
  分层统计;方向来源=能力-一致性普遍规律,见引言预告)。
- **J3 census 宽度(实质判据)**:全 battle 争议份额(bad/n)与
  CP 上界——**PASS = 上界 ≤ 0.40**;若 >0.40:judge 排行榜大面积
  争议,如实报(front-page 阴性候选,预声明)。
- **J4 拆 judge 前瞻**:偶位 8 校准/奇位 9 测试(样本=judge);
  校准折零"超 α_port 争议 battle 份额超标"judge ⇒ 测试折同类
  超标 judge 数 ≤ 1(小样本容忍,CP(0,9,0.1)=0.206 档);
  **PASS = 测试折超标 judge ≤1 且其 CP 界可报**。
- **J5 前瞻对照(加分项)**:认证 judge 集 vs 全体的 battle 争议率差
  ≥5pt = 认证层有效;否则如实报"认证层不改善"。

## 3. 判读表(冻结)

| 分支 | 条件 | 结果 |
|---|---|---|
| V7-1 | J1✓+J2✓+J3 上界≤0.40+J4✓(+J5✓) | 第二生态完整入稿:合并稿 Part 3 加 judge 节+摘要句 |
| V7-2 | J1✓+J2✓+J3 边缘(0.40-0.50) | 如实报+hedging 强化;形态同 V7-1 但主张收窄 |
| V7-3 | J3 >0.50 | **judge 排行榜大面积争议**=front-page 阴性(方法正确判定的结果);第二生态节如实报 |
| V7-4 | J1 违(实现错)或 J2 反 | 停链排查;排除实现后如实报(恒等式若不成立=重大异常) |

## 4. 预声明败点

- **F1**:J2 方向预言错误(70B 级 judge 一致率低于 7B)——能力-
  一致性普遍规律在 judge 生态被否,如实报;
- **F2**:争议 battle 集中于特定类别(chat vs reasoning)——层级
  归因改变(非均匀争议),如实报;
- **F3**:拆 judge 前瞻在校准折即超 α_port——前瞻分支判死,如实报;
- **F4**:prob 边际与 winner_pred 不一致(裁判内部矛盾)——数据
  审计发现,如实报。

## 5. 止损与纪律

- 止损门:数据就绪(judge_data_check.json:17×49,938 全覆盖)✓;
  约定清单(v5 E1)义务生效——judge 生态的约定块
  (bias 轴/概率语义/battle 命名)随结果发布;
- 修正案通道;代理分数禁用(本周期全真实判决,无代理);
- 概率边际(prob_A/B/C)只用于次要分析(稳健性),主判据用
  winner_pred 方向(与数据集主口径一致);
- 多重比较:136 judge 对的定律核对为恒等式门(非检验);J2 方向
  判据=逐方向全对或 Spearman≥0.8,不做 Bonferroni(预声明)。

## 6. 产物

- out/x2_v6/judge_pair_law.json(136 对 τ/π 核对表)
- out/x2_v6/judge_census.json(battle 层 census+分层)
- out/x2_v6/judge_prospective.json(拆 judge 前瞻)
- out/x2_v6/v7_verdict.json(判读表执行)
- 合并稿 Part 3 加 judge 节(判读后)
