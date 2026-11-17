# Paper 3 预注册 v1(FROZEN)

状态:**FROZEN**(冻结时间 2026-11-06 22:5x,经作者会话指令"继续"确认——判据 J1–J4 已于当日会话中完整展示;此后任何偏离记入 prereg_amendments.md,不改冻结文本)
项目:Stability-Aware Contract-Marginalized Aggregation for DocRE(Neurocomputing 投稿)
先例:探索性阶段全部结果(2026-11-06,notes/2026-11-06_v0_first_run.md 及其追记 1–8)已如实登记;
本文件冻结确认性阶段的判据与协议。探索性数字不得作为确认性主张的证据来源。

---

## 0. 红线(与实验室纪律一致)

04_文档级IE 与 paper2 目录只读;冻结 run 永不重训;一切学习在读数层;GPU 零使用;
偏差必须记录(日期 + 理由 + 知悉时点),记入 prereg_amendments.md,不删改已冻结文本。

## 1. 冻结数据与资产

- DocRE:44 run(7 生成器)逐文档计数;双金标 gD/gR;候选 189,556 对;
  规范 498 文档序(X2 ml_D_r025_s20260917 sorted)
- 翻转真值:r1_rank_flip.json(57 检查/13 翻转)+ r64_ec1 五读数器覆盖(冻结件)
- 装载门(已通过,2026-11-06):自有 loader 重建 registry 与 r60 数字逐位一致
  (out/repro_gate_docre.json、out/repro_gate_wmt.json)
- WMT19:de-en/zh-en,16/15 系统,段级三指标读数(缓存 wmt_seg_scores_*.json),
  锚 = wmt-raw 官方人类均值;WMT 翻转:r60 flip_detail(11+3)
- TACRED:45 系统,r59_n1 语料级读数;registry = r64_ea1(88/990,已验证精确对齐)

## 2. 冻结协议声明

- **run 协议(主)**:每生成器取首个 released run(排序规则:目录名单字典序第一个),
  全文一致;**敏感性**:跨 run 平均(并列报告,不进主判据)——run-mean 退化现象
  (探索性:覆盖 2/13)已登记,两口径不可互换
- **split**:文档级 350/148,seed 见 §5;段级(WMT)不设 split(全段训练+全段评估,
  迁移性质);TACRED 无文档级(语料级读数,登记局限)
- **bootstrap**:B=2000,percentile,tie = CI 含 0(touching counts as tied,P1 冻结规则)
- **特征**:7 类 18 列(out/features.npz);剔除 gR 泄漏列 [5,6,8,10];契约内全局 z 标准化
  (WMT 侧;DocRE 侧文档内 z 已冻结于实现)
- **目标函数(冻结)**:契约共识(median 符号 + 强度缩放边际 MARGIN_Z=0.25)+
  TALR 式软 tie(核宽 1.0→0.3 线性退火,60% 进度)+ LOO 翻转惩罚(λ=2, κ=0.5);
  权重 softmax 归一(正性+归一化约束内置)
- **确认性种子**:DocRE/WMT/TACRED learned = {20261201, 20261202, 20261203, 20261204, 20261205}
  (与探索性种子集不相交;探索性种子 20261106–110 仅作实现验证,不入文)

## 3. 主判据(全部运行前冻结;DocRE 主战场)

记 learned 为全特征共识目标聚合器(5 确认性种子 mean);基线 = gD 原生 / AER-corpus
(训练契约等权 z 边际,P1 口径)/ uniform(读数空间)/ consensus-median / corpus-learned /
trivial-features。

- **J1 硬排保真率(主)**:learned 硬排翻转检查的共识一致率 > gD 原生的对应率,
  Fisher 精确检验单侧 p < 0.05;且 learned 自身 ≥ 6/8(即不因种子漂移跌回基线水平)
- **J2 false-tie(主)**:learned false-tie 数 ≤ min(gD 原生, AER-corpus)的对应数
- **J3 覆盖(非劣)**:learned coverage ≥ AER-corpus coverage − 1(允许 1 对之差)
- **J4 质量(双锚)**:
  - DocRE 锚(Pareto):τ_vs_gR ≥ gD 原生 τ − 0.05,且 τ_vs_gD ≥ uniform − 0.05
  - WMT 锚(硬):|τ_learned − τ_uniform(AER)| ≤ 0.02(de-en 与 zh-en 各自成立)

**判读表(运行前冻结)**:
- J1 ∧ J2 ∧ J3 ∧ J4 全过 → 完整主张:"学习聚合移动 coverage–precision 前沿并保持
  硬排保真,跨契约族稳健"
- J1 过、J2 或 J3 部分过 → 缩减主张:以硬排保真为主结果,前沿位置如实报告
- J1 未过(与 gD 无显著差)→ 如实报负结果/边界论文:三测试床适用性准则 + 退化角点
  分析仍成立,降级 short
- 任何情况下:不一致的种子(5 种子中 ≥2 个方向相反)必须在正文披露,不以 mean 掩盖

## 4. 副判(迁移与鲁棒)

- TACRED(边界复制):全读数器 micro 保真率 < 0.5,且 learned ≥ uniform(方向一致,
  不设显著性门槛——n=88 的 Wilson 区间重叠已登记)
- WMT(parity):|τ_learned − τ_uniform| ≤ 0.02(de-en、zh-en)
- 敏感性(并列报告,不进主判据):run-mean、候选量加权(qn vs 均匀)、CI 水平扫描
  50–99%、trivial/corpus 消融
- 归因:硬排翻转对的逐文档贡献分解(探索性方法,描述性报告)

## 5. 确认性运行清单(冻结后按此执行,一次性完成)

1. DocRE:learned-full × 5 确认性种子 → registry 覆盖/false-tie/硬排保真率/τ 双锚 +
   六基线同口径 bootstrap(B=2000)
2. WMT:learned × 3 确认性种子 → τ vs anchor + registry 覆盖/硬排(B=2000)
3. TACRED:learned(global/cond)× 5 确认性种子 → 翻转保真率(Wilson 95%)
4. 汇总:out/confirmatory/(每运行一个 json)+ 回填论文 [CONFIRM] 表格
5. 运行脚本:src/agg_v03/v05/v06/v08/v09 的参数冻结版(入口统一 src/confirmatory_run.py,
   如实现差异与冻结协议冲突,以本文件为准并记偏差)

## 6. 与探索性阶段的差异登记

- 目标函数:探索期曾用 gD 符号目标(v0–v0.1,已证上限=复制 gD)→ 冻结版为共识目标
- 标准化:探索期 WMT 曾误用逐系统列标准化(τ 崩坏)→ 冻结版为契约内全局标准化
- 上述探索性教训本身作为方法设计依据写入论文,但探索性数字不进入确认性主张

## 7. 偏差与修订

冻结后任何偏离(参数、种子、协议)记入 prereg_amendments.md(日期/内容/理由/知悉时点),
与论文 Limitations 互引。审稿人可核对的运行哈希在确认性汇总中落盘。
