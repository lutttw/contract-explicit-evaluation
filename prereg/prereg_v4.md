# prereg v4 — Contract-Explicit Evaluation with Scoped Certificates(五步协议确认周期)

**状态:FROZEN(2026-11-14,作者批准;会话指令"prereg v4启动")**
冻结发生于查看任何 v4 确认性结果之前;全部探索性数字(2026-11-13/14 战役,
src/gate1_icc.py 等 13 脚本 → out/*.json)属 pilot disclosure,已向作者展示
并构成设计动机的一部分,特此声明。OSF 外部时间戳 = Phase 0(见 §8),
上传前本文件哈希已入 MANIFEST。
冻结须发生于查看任何 v4 确认性结果之前。本文件起草所依据的全部数字来自
2026-11-13/14 探索性战役(src/gate1_icc.py 等 10 脚本 → out/*.json),属
pilot disclosure,已向作者展示并构成设计动机的一部分,特此声明。

---

## 0. 动机与谱系(协议第四次自我应用;与 Paper 1 理论族的关系)

- **v1**:立纪律(J1 FAIL → 缩减主张)。**v2**:试强保证(注册负结果)。
  **v3**:命题 1 感知门(分支 1,全绿)。**v4(本预注册)**:把 v3 的
  单点证书升级为**可范围裁定的评测方法**——五步协议
  (Declare→Measure→Scope→Certify→Hedge),并首次引入
  **逐维 census 证书**(阳性对照已验证可行)。
- **程序内先例(必引)**:Paper 1 E-A14(退化工件重排近并列,matched
  stand-ins 保序)+ E-A16(代数翻转传播,identity 5/5)——本预注册的
  反相关定律(命题 2,见 §3)为其契约侧孪生:同一比值结构
  (扰动量/实例间距),两个扰动族。novelty 主张 = 契约侧扩展 +
  认证范围裁定协议 + K 地板算术,不主张新保证引擎。
- **v2/v3 负结果全文保留**;v4 不推翻任何 Part 1–Part 2 主张。

## 1. 五步协议(方法本体;冻结)

1. **Declare**:枚举契约轴(金标/指标/阈值/解码),逐轴冻结读数。
2. **Measure**:逐(分层×契约维)测 flip 密度 = 两契约读数秩相关之补
   (命题 2 的可操作化)。
3. **Scope**:可认证性地图——K 地板算术(census 零坏 δ=0.10 需 n≥22,
   bound=1−0.1^(1/n))+ 坏密度(率>α_port 的实例占比)→ 格级分级
   T1(≤.10)/T2(≤.15)/T3(≤.20)/T4(≤.30)/不可认证。
4. **Certify**:过格签证书——
   (a) 逐维 census:cp_upper(坏数, n, δ),FS 跨维组合(维序=§6 预声明);
   (b) 跨模型单元级:固定集 Bauer 序列(FS 语义,**禁用 MP 覆写**——
   Gate 2 实测 MP 在 α≤0.10 为选择效应),CP-RCPS 校准。
5. **Hedge**:未过格 tie 声明 + registry 披露。

**适用性准则(沿用 Part 2 的 2×2 表)**:可交换性 × 翻转代价;
三测试床(DocRE 部分可签/TACRED 交换性 no/WMT parity)为方法边界实例。

## 2. Phase 设计

- **Phase 0(OSF)**:[2026-11-14 修正案 4 撤销——作者决策自证立场;
  zip 已构建留档不上传;确认前置=FROZEN+止损门两条件] ~~本文件 +
  十份探索 JSON + 现版地图(pc2_certmap.json)打包上传 OSF,取外部
  时间戳;先于 Phase 1 任何训练~~。
- **Phase 1(x2 增殖,预注册生长项)**:[2026-11-14 修正案 5 算术更正:
  r025/r100=训练侧负采样比(per_doc 哈希跨 regime 不同已验证),现存模型
  =20 非 10;**+4 run-dir(1 新种子 20261201 × 2 臂 × 2 比)→ n=24≥22**,
  替代原 +12 口径] ~~+12 新模型 = 6 新种子 × 2 臂~~;种子 20261201
  (与探索集不相交);训练管线 = e50 前例 P3 镜像(src/x2_v4/,写路径
  重定向);**前置任务 = 498 池规则考古(R2-M1)✓ 已完成**
  (docs/pool_provenance_r2m1.md)——新模型评测覆盖同池。
  健康门:预测覆盖 ≥95% docs、F1 非退化(≥ 架构中位 −5pt)、预测数
  ≥ 金标数 0.5×;未过者剔除并披露;**若剔除后模型级 n<22,T1 不可达
  如实降级,禁止补跑追门**。
- **Phase 2(registry 刷新 + learned conf)**:14 族 registry 重建
  (x2 22 模型入池);读者 conf 用部署管线真实值(禁 stab 代理——
  Gate 2 边界:全部分析从代理换实测,这是本周期与探索战的本质差异)。
- **Phase 3(确认运行)**:C1–C4 + C5 消融,一次性全链
  (confirmatory_run.py 同款纪律:非 FROZEN 拒跑)。

## 3. 主判据(冻结)

- **C1 跨模型单元级证书(主)**:FS 语义,α\*=0.20;
  PASS = feasible 且 realized 错误率 ≤ α\* 且 commits ≥ 11(零错地板);
  **C1′(副)**:α\*=0.10,PASS = feasible 且 commits ≥ 22。
- **C2 within-x2 逐维 census 证书**:scoring 维,模型级 n=22;
  PASS = 0 坏(T1,bound ≤0.0999 顶格);1+ 坏 = 如实降级报告
  (T3@1 坏,bound≈0.27)——降级本身预注册为合格结果,非失败。
- **C3 地图复制**:维度层级方向(scoring ≪ gold < threshold 的
  within-架构翻转密度序)在 learned conf + 新池上复制;
  PASS = 序不变(Spearman 层级秩相关 ≥0.8,或逐维方向全一致)。
- **C4 定律验证**:格级 flip 密度与契约对秩相关呈单调补关系
  (全 24+ 格,|ρ|≥0.9);PASS = 单调成立。
- **C5 消融(registered secondary)**:weighted conformal
  (tibshirani2019weights,权重=cal/test 折 cell 构成密度比)
  能否扩大可认证格数;主判据=新增认证格 ≥1 且格级 coverage 不降。
  C5 结果**不进方法主主张**,作 §ablation。

## 4. 判读表

| 分支 | 条件 | 主张形态 |
|---|---|---|
| 1 | C1+C1′+C2(T1)+C3+C4 全过 | 方法论文全文:协议+定律+地图+双证书 |
| 2 | C1 过+C2 降级(T3/T4)+C3+C4 过 | 方法论文,证书主张收窄至 T3/T4 |
| 3 | C1 FAIL(α\*=0.20 不可行/超支) | 定律+地图+边界论文(证书主张撤回,positive control 保留为 feasibility 证明) |
| 4 | C3 或 C4 FAIL(定律不复制) | **阴性结果上头版**——定律在 learned conf 上被否证,如实报告;方法主张撤回 |

**止损门(Phase 2 末,运行 Phase 3 前)**:registry 总 flips ≥ 35
(沿 v2 止损门推导口径);x2 模型级 n ≥ 22;两门任一 FAIL → 停,
按对应分支降级处理,禁止中途换池/加族。

## 5. 预声明败点(任一出现,上头版报告)

- **F1**:12 新种子的 scoring 读数出现符号翻转(坏实例 >0)→
  C2 降级,且"scoring 维结构性稳定"的机制主张被部分否证;
- **F2**:learned conf 地图与 oracle stab 地图的认证格集合
  Jaccard <0.5 → 方法的可部署性主张失败(F2 出现时 F2 本身
  上头版,方法退化为"需 oracle 分数的可行性研究");
- **F3**:FS 在真实 registry 的 α\*=0.20 无可行阈值(v2 branch-3
  复现)→ C1 FAIL → 分支 3;
- **F4**:维度层级反转(gold 成为最稳定维)→ 定律否证 → 分支 4;
- **F5**:weighted conformal 扩格失败且格级 coverage 下降 →
  C5 报告为负,不隐藏。

## 6. 预声明参数(冻结)

- δ=0.1(conventional 90% risk-control confidence,沿 v3);
- α_port=0.20 主 / 0.30 敏感性;
- FS 维序 = (scoring, threshold, gold) ——**由探索地图导出,
  属 adaptive ordering,本条即为披露**;备选 Bonferroni(δ/3 逐维)
  并列报告;
- bootstrap B=2000,seed 20261211–15(确认种子,与探索不相交);
- 敏感性臂:**去 FAMS**(7 单种子族)全链重算——单种子族零方差
  单元稀释风险画像的教训(c5 发现)固化为本周期常设臂;
- 报告口径:一切 "certify" 措辞须带 CP 上界数字;realized 率
  与 bound 并列;**禁止** "guarantee" 无限定使用。

## 7. 偏差与修正案纪律

沿用 prereg_amendments.md 账本:冻结后任何偏离只走修正案通道,
先登记后执行;错账披露优先(v3 校准器 bug 先例:
"implementation correction disclosed first")。确认运行前
wrong-counter 单元测试夹具(conf/commit 逐位断言)必须 PASS
——v2 校准器 bug 的永久防 recurrence 条款。

## 8. OSF 清单(Phase 0)

prereg_v4.md(本文件)/ prereg_v1–v3 / amendments 账本 /
探索期 10 JSON + 10 脚本 / pc2 地图 / manifest 哈希表 /
R2-M1 池规则考古件(含 doc IDs)。时间戳取得后回填本文件头部。

---

*起草:2026-11-14。待作者审定后 FROZEN 并盖 OSF 时间戳;确认运行
由 confirmatory_run.py 同款守卫执行(prereg 非 FROZEN 自动拒跑)。*
