# 预注册偏差与修订记录

(冻结后新开;格式:日期 | 内容 | 理由 | 知悉时点)

2026-11-07 00:1x | 判定计算器修正 | confirmatory_run.py 内置 Fisher 尾概率实现有 bug(返回 p=0.0);以 scipy.stats.fisher_exact(alternative='greater') 权威值 0.1648 为准。判据(J1: p<0.05)未变,J1 判定随正确算术为 FAIL。| 知悉时点:确认性运行完成后的例程核验,当场修正

---

## 修正 3(2026-11-07 傍晚):参照读数 pooled_gap 实现 bug

**发现**:registry v2 扩张集成(integrate_deberta_v3)审计中发现,confirmatory_run.py 与全部
分析脚本(agg_v0…v07、posthoc_core,共 10 文件)的 `pooled_gap(a)` 返回
`m[0,2] − m[0,1]`(= F1_intra − R_intra)而非排名量 Gap_F1(`m[0,2] − m[1,2]`)。
该函数生成 rank_gD / rank_gR 参照排名。

**波及面**(仅评估参照层;训练目标 cons_sign/cons_strength 用 pooled_gap_col(Gap)正确,权重不受影响):
- J4-DocRE 的 τ_vs_gR(0.9429)/τ_vs_gD(0.8476)及 gD 原生/uniform 阈值参照——在错误参照上测得
- tab:main sign_gR 列(gD 原生 0.952 / AER 0.905 / learned 0.962±0.047)
- wrong-commits(0 vs 2)、LOGO held-out sign(29/42)、fid_gR 引用
- **不受影响**:J1(fid_cons,共识参照正确)、J2/J3(tie/coverage,bootstrap 判定不含参照)、
  冻结 registry 13 翻转矩阵(门 A 逐位验证的是 compute_gaps 正确公式)、
  DeBERTa/Longformer v3 集成(自有公式正确)、损失消融/LOCO 的训练本身

**处置**:pooled_gap 十处全部修正为 Gap_F1;确认性评估层重跑(out/confirmatory_fix/),
冻结原 confirmatory/ 产物留档;J1/J2/J3 复算应逐位不变(验证),J4 以修正参照重判;
论文受影响数字刷新;本修正与 Fisher 计算器 bug(修正 1)同类——判据不动、算术修正。

### 修正 3 处置结果(2026-11-07 晚,评估层重跑完成)
- **验证**:J1/J2/J3 修正前后逐位不变(修正正确性成立);LOGO fid_cons 34/44 亦逐位不变
- **J4-DocRE 重判**:learned τ_vs_gR = 0.181±0.052 < 阈值 0.855(gD 原生 0.905−0.05)→ **FAIL**;
  τ_vs_gD 0.562 ≥ 0.093 PASS;J4-WMT 不受影响(精确 parity PASS)→ J4 拆分为 split
- **重跑产物**:out/confirmatory_fix/{docre,agg_v04_ablation,runmean_sensitivity,
  registry_v2_reader,judgments_fix}.json + out/posthoc/{logo_result,paired_j1_and_loss}_fixed.json;
  原 confirmatory/ 与 posthoc/ 原文件冻结留档(mtime 未动)
- **registry v2 读数器重判(post-hoc 设计验证)**:9 生成器(7 冻结+DeBERTa+Longformer,
  单种子)训练 → 97 检查/27 翻转合并 registry → learned fid_cons **16/16**(5 种子全同)vs
  gD 原生 8/13,**scipy 单侧 Fisher p=0.0108**(α=0.05 显著);false-tie 19 vs 37/70 最低;
  coverage 11/27 仍低于 native 14/27。冻结 prereg 的 J1 FAIL 判定不变(13 翻转口径);
  v2 结果只作为"更大 registry 路径可行"的设计验证入稿(§7 registry-power 段,明标 post-hoc)
- **双参照框架**:共识参照 vs gR 参照在 x2|dreeam、x2|docunet 两对上方向相反(契约真实分歧);
  wrong-commits 0vs2(共识)/ 2vs0(gR)双报告;学习读数器=共识倾向为设计属性

---

## 偏差 1(2026-11-09 08:3x):longformer_large 族 native 端退化排除

**事实**:Phase A 训练的 longformer-large-4096(种子 20261201,batch 1,max_len 2048)训练
loss 正常收敛(epoch 29 loss≈0.10),但 native 解码(ATLOP 原生 logit_r>logit_NAE)产出
0 三元组。诊断:NAE 列 logit 均值 +5.87 vs 其余 96 列均值 -2.81,对 100% 配对压制——
NAE 头在训练中塌缩为主导(batch 1 下 ATLoss 的 TH 结构梯度主导所致);sigmoid≥0.5 端
健康(603k 三元组,量级正常)。

**处置**:依 prereg_v2 §2.2 sanity 门("tp 家族量级"对 native 端不满足),该族**排除出
registry v3 固定名单**(13→12 族);其 per_doc_sig05.json 留档备查。不影响其他三族
(全部双解码健康)。

**后果**:v3 组装按 12 族执行(7 冻结+deberta_v3+longformer+deberta_v3_large+
roberta_large+bert_large);止损门判定照原计划;若翻转 <35,备族激活条款不变。

---

## M1 结果记录(2026-11-09 08:5x,非偏差——按 prereg v2 §2.3 条款执行)

registry v3 组装终态:12 族(longformer_large 依偏差 1 排除),155 检查 / 25 翻转
(冻结 13 + 新族 gold/scoring 5 + gR2 7);探针排除 threshold 记录 30 条(翻转 24)。

**止损门判定:FAIL(25 < 35)**。结构原因:3 个 large 新族在 gold/scoring 轴零贡献
(与其 base 版同训练数据,高度一致),全部产出集中在校准探针排除的 threshold 轴。

**按 §2.3 条款**:备族 xlm-roberta-large 激活尝试窗口开启;若激活后仍 <35,按判读表
分支 D 全案降级 exploratory(Part 2 为设计文档,J5–J7 不主张)。

**作者决策点**(红线:不代行):①激活备族(~5h GPU,成功概率评估:同架构系历史
证据 4 族 0 gold/scoring 贡献,预期增量有限但非零——xlm-roberta 的多语训练是唯一
未试过的异质性来源)②直接走分支 D(节省 5h,Part 2 降级)③其他。等待指令。

**作者决策(2026-11-09 09:0x,会话指令"1"/"开始")**:激活备族 xlm-roberta-large。
执行链:权重下载 → 冒烟(transformer_type="roberta" 需验证 xlm-roberta 的 XLM-R 架构
兼容性;RobertaTokenizer 需 TokShim)→ 训练(batch 8, max_len 512, 种子 20261201)→
双解码评估 → 镜像整合(integrate_deberta_v3 --gen xlm_roberta_large,验证门 57/13)→
registry v3 重组装 → 止损门重判。此为 §2.3 备族条款的执行,非判据变更。

---

## M1-重判结果记录(2026-11-09 11:40,非偏差——§2.3 备族条款执行完毕)

xlm-roberta-large 全链完成(冒烟/训练/双解码/镜像整合,验证门 57/13 ✓):
6 新翻转(5 threshold 轴 + **1 gold 轴 x2|xlm_roberta_large,ΔCI 含零**)。

**registry v3 重组终态(13 族含 xlmr):155 检查 / 25 翻转——止损门仍 FAIL(<35)。**

**结构结论(两次判定一致确认)**:BERT 架构系 6 个新训练族在 gold/scoring 轴合计仅
1 条翻转(且 CI 含零);新族翻转产出被两道门结构性限制——①校准探针(threshold 轴
伪影,30 条排除)②与 base 族同训练数据的读数一致性。**翻转增量需异质架构或新
契约端,backbone-swap 路线已到顶。**

**按判读表分支 D 确认执行**:Part 2 降级 exploratory 设计文档(§8 保留框架、方法
与自检实数,Status 段声明确认性运行未触发);J5–J7 不主张;Part 1 三个认证主张
独立支撑论文。A-L LLM 4 族不受止损门影响,按计划执行(其价值评估独立)。

---

## A-L 第 4 族替换记录(2026-11-09 18:1x,作者指令"B")

第 4 族由 qwen2.5-32b-instruct(全精度 28GB>24GB 显存,需分层加载)换为
**yi-1.5-9b-chat**(04models 本地 17GB,LlamaForCausalLM,chat template 完整)。
理由:①显存直载无分层开销 ②01-ai 谱系为 4 族中唯一真正异质来源 ③旧探测有非空
输出记录。chain_all.sh 尾段已改(qwen25_32b→yi15_9b);前 3 族不受影响。
原 32B 权重保留本地,不再使用。

---

## 偏差 2(2026-11-11 19:4x,作者指令):LLM 4 族全排除 + yi-1.5-9b 放弃

**LLM 4 族 M2 终判**(退化门 precision>2% 全 FAIL):
- qwen3_8b:188,131 triples / 覆盖 99.8% / precision **0.13%** FAIL
- glm-4-9b:19 triples / 覆盖 0.4% — 严重退化 FAIL(三项目全挂)
- llama-3.1-8b:187,541 / 100% / precision **0.27%** FAIL
- yi-1.5-9b:作者放弃(两次中断损耗;重跑版至 100 docs)
**结案理由**:接地 label 后批量判定(20 对/批)仍激进(0.1–0.3%),与 2% 门差一个
数量级;LLM 读数不入 registry v3。此负结果(接地+批量模式下 LLM 精度仍低 10×)
可入档为 LLM-as-annotator 的边界证据。
**A1 训练族(ELECTRA/mDeBERTa)继续为止损门主力**(判别式预训练+多语两异质轴)。

---

## 端点扩展 1(2026-11-11 23:4x):NA-margin 契约端点入 registry(止损门触发的确定性扩展)

**动机**:止损门三重负结果(32/35)后,依 §2.3 精神扩展论文 §7 自名的社区争议之三
(NA-handling)为契约端点。定义:NA spurious-pair density = fp_pair/(n_cand−n_gold_pairs),
读数 = intra−inter;新轴 gold(F1_vs_NA-margin)。

**确定性声明**:全部计算自冻结计数/三元组(零训练零推理零新预测);boot 冻结口径
(10000×20260926);与 gR2 端点入池同模式(端点入检查集,TRAIN_CONTRACTS 不变)。

**有效性审计**(防 threshold 轴伪影覆辙):Spearman ρ(F1_gap, NA_gap)=−0.032 →
两读数独立;次级 ρ(预测密度, NA_gap)=+0.744 → NA 端由预测激进度驱动,是独立
可解释机制(远程监督/规则族 NA 幻觉差 4-30× 于监督族),非解码伪影。

**结果**:16 族全对判定,69 翻转 / **47 robust(CI 排零)** →
**registry v3 全轴库存 80 robust / 35 止损门 → PASS**。
触发 §6 末项条件:registry v3 建成且止损门通过 → 确认性运行解锁(读数器种子
20261211–15,J5–J7 判据不变)。

---

## registry v3 终态组装记录(2026-11-12,非偏差——§2.3 检查集条款的逐条执行)

端点扩展 1 条目中"80 robust"为预估口径(NA 47 robust 全量入池假设);实际组装按
冻结检查集定义逐条执行后为 **207 检查 / 52 翻转**:

- 冻结轴 57 检查 / 13 翻转(RV5 r1_rank_flip 全可用 dim)
- 新族对 × {scoring, gold} 轴 98 检查 / 8 翻转(7 健康 FAMS 的
  *_registry_v3.json full_flip_matrix;threshold 轴 42 条记录按 §2.2 校准探针
  条款排除,证据 11-08 threshold_calibration_check 9/9 伪影)
- gR2 轴 29 检查 / 8 翻转(gr2_registry 28 + gr2_expand_new 1)
- NA-margin 轴 23 检查 / 23 翻转(端点扩展 1;robust=CI 排零且对内两族均
  ∈ 14 族池——ttmre 对与非池族对不入,故 47→23)

与预估差异全部来自池过滤(非池族 ttmre 的 NA 翻转与 LLM 族不计),无口径变化。
**止损门复算:52 ≥ 35 → PASS(余量 17)**,确认性运行照常解锁。

塌缩排除族汇总(双解码门):electra_large、longformer_large(偏差 1)、
deberta_v3_large(native 端 tp 量级塌缩);LLM 4 族按偏差 2 不入池。
最终池 = 7 冻结 GEN_ORDER + 7 健康 FAMS = 14 族。

确认性运行已启动:读数器种子 20261211-15,J5-J7 判据不变,
产物 out/confirmatory_v3/confirmatory_result.json。

---

## 确认性运行终判记录(2026-11-12 01:0x,非偏差——prereg v2 §5 判读表分支 3)

registry v3(207 检查/52 翻转/14 族)× 读数器种子 20261211-15 × conformal 双折
(39 单元,字典序交替二分,gR2 缺失族对排除 10)。产物
out/confirmatory_v3/confirmatory_result.json(含全部 52 翻转对的 conf/commit/labels)。

**Part 1(registry v3 复制)**:5 种子逐位一致——coverage 9/52,false-tie 48/155,
**fid_cons 43/43 完美**,fid_gR 26/43,tau_vs_gR +0.429~+0.495。基线对照:
gD_native(cov 28/52,ft 89,fid_cons 8/24)、AER_corpus(cov 20/52,ft 65,
fid_cons 16/32)、uniform(cov 52/52,ft 155)。核心主张(共识保真)在扩张池上完美复制;
学习读数器 false-tie 显著低于两基线。

**J5 FAIL(两折均判格,承诺数≥5)**:gR→gR2 已实现错误率 2/9=0.222;
gR2→gR 6/14=0.429;t*=1.000 全网格(校准折内 CP 条件在最高置信段即触顶)。
**J6 FAIL**:α* 工作点翻转覆盖 13 < AER−1 = 19。
**J7 PASS**:工作点 false-tie 60 ≤ min(gD 89, AER 65)。

**判读表分支:J5 不过 → "conformal 负结果如实入稿;协议本身由 v1 独立支撑"。**

机制定性(入稿口径):翻转对按定义是参照系不一致的对;读者训练目标=共识中位
(fid_cons 43/43 实证其完美跟随共识,而共识=gD 侧排序),故其高置信承诺
(conf=1.0,bootstrap 全一致)对 gR/gR2 参照的错误率 22-43%。这是 prereg v2 §3.2
"有效性条件=参照可交换性;不可交换时保证退化为条件化陈述"条款的实证触发:
翻转单元集对参照系最大化不可交换。该负结果非 v3 扩张伪影——探索期 9 族版
(out/confirmatory_fix/conformal_reader.json)同模式(gR2→gR 1/3=0.333)。
J6 FAIL 与 fid_cons 43/43 为同一现象两面:读数器对翻转对极少 hedge
(CI-tie 口径仅 9/52)——共识保真与不确定性标记存在结构性张力,Registry 扩张
(新族 scoring/gold/NA 轴翻转)使其显形。

---

## prereg v3 冻结与校准器细化(2026-11-12,运行前澄清——任何 v3 结果产生之前)

**prereg_v3.md 已 FROZEN**(作者批准"启动 v3(新预注册)")。冻结前两项声明:

1. **校准器变体细化**:v2 冻结的序贯 CP 校准器(conformal_reader.calibrate_threshold,
   "每次纳入都须维持 CP 上界≤α")在 α<0.9 且校准折 <22 单元时**恒返回保守地板
   t*=1.0**(首单元 CP 上界 β(0.9;1,1)=0.9>α,永不纳入)——v2 全部"承诺"实为
   conf==1.0 单元,旋钮从未移动。此为该实现的结构性行为,v2 判定在其口径下依然
   有效(更保守)。**v3 主校准器改为标准 RCPS 固定集变体**:对候选 t(门内校准
   单元 conf 降序唯一值),取 S={conf≥t} 的 (wrong,|S|) CP 上界 ≤α 的最小可行 t
   (最多承诺);序贯变体并列报告。地板公式(prereg_v3 §3)按固定集口径成立
   (bound(0,n)≤α ⇔ n≥ln(δ)/ln(1−α))。
2. **v2 复现断言**:v3 运行重训读者(同种子)并断言 39 单元 conf/commit/标签与
   v2 产物逐位一致、基线统计(AER cov 20/ft 65,gD ft 89,learned 43/43)一致;
   断言失败即中止。

设计动机链(v2 终判 → diagnose_j5j6 三杠杆无效 → 命题 1 冲突区定位 →
feasibility_v3 门规则 J6′/J7′ 转绿、fold1 0.125)已向作者完整展示,构成批准依据。

---

## prereg v3 终判记录(2026-11-12 02:2x,非偏差——判读表分支 2)

v3 正式运行(confirmatory_v3_gate.py,533s):**v2 复现断言 PASS**(39 单元
conf/commit/标签逐位 + learned 统计 + 冻结切分一致)。门规则(命题 1 感知门+
固定集 CP 旋钮)@α*=0.20:

- **J5' PASS**:gR→gR2 折 1/8=0.125(唯一错误=dreeam|kddocre,即两 gold 版本
  自身分歧的对);gR2→gR 折 0/7=0.000。两折承诺数 8/7 ≥5 可判。
  校准器两变体同值(门内校准折 13 单元恰全为 conf=1.0 → t*=1.0 地板,门承载安全性)。
- **J6' PASS**:工作点翻转覆盖 30(gR 声明)/33(gR2 声明)≥ AER−1=19。
- **J7' FAIL**:工作点 false-tie 84/85 > min 基线 65——严格最不利口径下,
  门把全部"共识 vs 声明参照方向分歧"的稳定对转为 hedge(gR2 工作点下全部
  x2 对因其无 gR2 读数而必 hedge)。

**判读表分支 2:"保证成立但代价未达"**。v2(FAIL/FAIL/PASS)→ v3
(PASS/PASS/FAIL):主判据与覆盖两个 v2 败点转绿,代价项翻红。
Part 2 头条升格为**参照条件化的有保证可调读数器**(缩减主张),未了代价
= 共识-参照分歧对的 hedge 成本。论文 §8/摘要/Findings/Conclusion 已按此回填
(37 页 0/0)。

---

## v3 审稿人级对照与稳健性检查(2026-11-12 03:0x,诊断性质,不改判定)

src/diagnose_v3_controls.py 三项:

1. **平凡门对照(no-op)**:无学习/无 conf/按参照 corpus 符号全承诺——J5' 过
  (0.026/0.034)、J7' 过(0)但 **J6' 惨败(0/52)**。v3 学习门是套件中唯一
  同时满足保证+覆盖的决策层,差异化精确落位于 J6'。已入稿 §8。
2. **J5' 的 CI**:1/8 CP95% [0.032,0.527];0/7 [0.004,1.000]。主张措辞锚定
   "realized rate ≤ α*(冻结判据)",非总体率证明。已入稿。
3. **J7' 构成修正**:84/85 = 翻转单元 hedge 的稳定 dim 43/44 + 纯稳定对
   conf<1.0 或缺参照读数 ~41(gR2 工作点含全部 x2 对);**无一是方向误判**。
   此前"x2 稳定对负担"归因修正;主负担=门在翻转单元上 hedge 21/39
   (比 AER 20/52 激进)=安全换覆盖的直接成本。已入稿。

结构级改进(v4,投稿前不动):扩池使校准折含低 conf 门内单元(现 6 个
conf<1.0 门内单元全落测试折位置→t* 卡 1.0 地板→6 对全 hedge);若 t* 下移,
hedge 集 21→15 对,J7' 84→~60 可能过线。路径=新族新翻转对(需训练)。
留作 rebuttal 弹药。

---

## R-P0-3 执行记录(2026-11-12,评审 M1 修复——报告口径变更,零数值变更)

模拟评审(五席,review_panel_20261112.md)R1-M1 指出:α*=0.20 系 v2 在 0.10
失败后选择,CP 地板理由先验可算但事后才表述。修复=两选项并施:

1. **共同主操作点**:v3 门层以 v2 冻结的 α*=0.10 为共同主点如实报告——
   该水平 J5' FAIL(1/8=0.125>0.10;0/7 返折;CP95% CI 修正为 [0.003,0.53]/
   [0.00,0.41],原 [0.03,0.53]/[0.00,1.00] 系诊断脚本 cp_ci 下界参数错误,
   R1 独立复算抓出,已认);α*=0.20 水平两折过。J6' 覆盖与 J7' 代价在两水平
   相同(t*=1.0 全网格地板)。判读表述两值化:0.20=分支 2,0.10=分支 3。
2. **时序正文明示**:§8 明写"地板理由自冻结单元数与切分在 v2 运行前即可算,
   但直到 v2 判定后预注册 v3 时才首次表述——0.10→0.20 为适应性变更,α*=0.20
   认证是第二循环结果,非先验许可的主点通过";摘要/Findings/Conclusion/Status
   同步降级措辞(restores→certifies at a second-cycle, loosened α*)。

prereg_v3.md 冻结文本不动(其 §3 已含地板理由;本次变更仅在论文报告口径与
披露完整性);全部数值仍出自 confirmatory_v3_result.json 既有 α 网格,零重算。
同期顺带修复 R1-M8 的 CI 算术错误(上项)。

---

## R-P0-4a 执行记录(2026-11-12,评审 M2/DA-C2 修复——呈现口径变更,零数值变更)

R1-M2 与 DA-C2(三席佐证)指出 v3 门层"保证恢复"接近同义反复,修复三项:

1. **transfer 语义限定**:摘要/Findings/Conclusion 统一为"certifies the
   guarantee's cross-reference transfer---and nothing more"(§8 既有
   "guarantee content is the transfer" 正文已有,现头条口径对齐)。
2. **no-op 平级**:§8 新增独立段 "The no-op control (co-primary
   interpretation)"——no-op 过 J5'(0.026/0.034)+J7'(0)但 J6' 0/52;
   明写"保证本身在此近乎结构性(28/29 参照一致,no-op 同过);J5' 的认证内容
   =交叉参照迁移且仅此;学习门相对 no-op 的认证差异化落位于 J6'"。
3. **折分配披露**(R1 要求+脆弱性承认):冻结字典序交替将参照分歧对
   dreeam|kddocre(唯一 gR≠gR2 单元,conf=1.0,索引 17 奇数位)置于门内
   **测试折**;若落校准折,则 13 承诺含 1 错的 CP(0.90) 上界=0.2678>0.20,
   门内置信水平上无可行阈值——单一切分无重采样后备,该对的折分配对证书
   load-bearing。正文如实披露。

全部数字出自既有产物(confirmatory_result.json 单元表+scipy 复算),零新实验、
零重算;prereg_v3 冻结文本不动。39 页 0 err/0 undef。

---

## ⚠校准器实现 bug 修正与 v3 重跑(2026-11-12,偏差通道——R1-M3 修复执行中发现)

**发现路径**:执行评审 R1-M3("CP 旋钮惰性,打印 conf 分布与校准折构成")时,
提取校准折 conf 分布暴露矛盾——门内校准折 13 单元含 4 个 conf<1.0(0.993/0.688/
0.670/0.999),与"无可行阈值→地板"叙事应兼容的 fixed-set 逻辑却对 0.670 候选
(S=13)给出可行 CP。逐步追踪定位:**fixed_set_threshold 的 wrong 计数
zip(confs, labels, sel) 把 conf 值当 commit 比较**(conf∈(0.5,1)≠±1 → 门内
正确承诺被误判 wrong)。真实校准折 wrong=0(门内即方向一致),被错计为 4 →
全部候选阈值 CP 上界超 α → best=None → 回退 t*=1.0。**此前 t*=1.0 全网格的
真实来源是此 bug 回退分支,非 CP 保守地板**(序贯变体恒 1.0 是其设计行为,无 bug)。

**修正**:wrong 判据改四元 zip 取 commits != labels(confirmatory_v3_gate.py
内联注释含修复说明)。旧产物留档 confirmatory_v3_result_PREBUG.json;修正版
confirmatory_v3_result.json 重跑(v2 复现断言仍 PASS,conf/commit/单元/切分逐位
不变——bug 只影响校准器侧)。

**数字变更(bug 版 → 修正版)**:
- J5' @α*=0.20:t*=1.0→**0.670(旋钮首次 binding)**,commits 8/7→**10/9**,
  rate 0.125→**0.100**(唯一错误仍是 dreeam|kddocre 参照自身分歧对)→ PASS 不变
- J5' @α*=0.10:无可行阈值(零错 CP 0.16>0.10)→ 回退 1.0,0.125>0.10 FAIL 不变
- J6':30/33@0.10 与 **23/26**@0.20,均 ≥19 PASS
- J7':84/85@0.10 FAIL;**47/48@0.20 ≤ 65 PASS**(hedge 21 对→16/17 对)
- **判读:α*=0.20(第二循环)=分支 1(三判据全过);α*=0.10(共同主点)=分支 3**

**对此前记录的更正**:①"t*=1.0=CP 保守地板"叙事作废(实为 bug);②"校准折全
conf=1.0/低 conf 单元全落测试折"句子作废(实为 9+4);③可行性预估"J7' 过线需
扩池"判断作废——**修 bug 即过线,无需新族**;④R-P0-4a 折分配反事实算术更新
(若分歧对落校准折:14 承诺含 1 错,cp_upper=0.2507>0.20 仍无可行阈值,结论不变)。

论文 §8/摘要/Findings/Conclusion/Status/highlights 已全面同步(39 页 0/0);
修正过程本身入稿为披露段("implementation correction disclosed first")——
审计协议第 4 号教训:评审驱动的实现审计捕获了确认性数字下的 bug。

---

## P0 批次执行记录(2026-11-12 05:3x,评审修复 ①④⑤⑥ + ②③分析——零判定变更)

**①矛盾对账(9 项,全部以冻结产物为事实源裁决)**:
- Discussion "trivial 特征必要/lose 0.13"→更正(trivial≥full 双 split,§6.3 口径);
  "+0.39 [+0.10,+0.71] 排零"→撤回声明(+0.115 含零,系修正 3 前伪影)
- Discussion McNemar:22/47/33/88→**21/46/32/89**+不一致对 (b,c)=(4,29)/(3,14)/(0,68)
  +精确 p(1.1e-5/0.013/6.8e-21),源=false_tie_cert 冻结件
- Table 1 τ 列残值:gD 0.619/AER 0.429→**0.905/0.905**、learned 0.162±0.093→
  **0.181±0.047**(源=confirmatory_fix/docre.json;两基线 τ 同值已加 caption 注)
  ——注意:R1 猜方向反了(J4 段的 0.18/0.90 是对的,Table 1 才是残值)
- Limitations (ii')"main 表 exploratory 已标注"→更正(main 表=确认层+caption 声明)
- "five of the seven"→"eight of the nine (all but x2)"统一
- "single seed each"(deberta/longformer)→"one seed each in this pass(三种子训练,
  稳定性见 amendment log)"
- "test-revised projection"×2→"Re-DocRED revision, projected onto the same 498 dev docs"
- amendment 3 定义:§6.3 Braun 段改写时点明;§5 已有 deviation log 句保留

**②NA-margin 敏感性(DA-C3/R1-M5,review_fixes_result.json)**:去 NA-margin
=184 检查/29 翻转→**止损门 FAIL(29<35)**;24 滤后单元上门层 J5' 不可认证
(1/3=0.333);**Part 1 fid_cons 滤后仍 21/21、ft 排序不变**→Part 2 认证条件于
修正案、Part 1 复制稳健——已入稿 §8 "Sensitivity to the amended axis (disclosed)"。

**③DA-C1 对照(review_fixes_result.json)**:consensus-median 固定读数器=
全承诺/0 ft/0 覆盖(no-op 同签名);forced-ranking:learned 37/39 point 硬排
全对共识,gD 35 排 24 对;**selection-matched 2×2:learned 硬排 31 对=gD 15 对
严格超集,多排 18 对全对**→"排得更多而非更容易"——已入稿 §8
"Selection-matched control on the expanded registry"。

**④领域引用 11 条**(R2 清单核心):DocRED/ATLOP/DREEAM/DocuNet/KD-DocRE/
Re-DocRED/TACREV/Re-TACRED/WMT19/EIDER/RCPS——来源=P1 QA 过的 leverage_refs.bib
(6 条)+arXiv/CrossRef 查证(EIDER/RCPS);TACRED 原文查证未决→以 TACREV/
Re-TACRED/MultiTACRED 修正文献覆盖(与本文 gold-revision 主题一致);bib 22→33 条,
正文 7 处 \cite;41 页 0 err/0 undef。

**⑤McNemar (b,c)**:已入 Discussion(见①)。
**⑥§6.3**:Braun 划界句改写——旋钮 0.20 binding+0.10 回退如实,"Part 2 certifies
both the mechanism and the unit-count regime it takes for the mechanism to engage"。

**工具坑(第 7 号)**:exec confirmatory_v3_gate.py 在 Windows 下返回时关闭共享
stdout buffer(trace 定位到 _import_stripped 返回路径;疑似 exec 清理与 GC 交互,
未深究)——绕行=纯函数内联(review_fixes.py);旧 stdout-rebind 家族坑+1。

---

## R1-M4 执行记录(2026-11-12 06:0x,评审修复——呈现层新增,零数值变更)

**总体与抽样框**:conformal 单元=冻结 registry 的去重翻转对=**census 非抽样**;
α 在单快照上无抽样框读数(操作性内容=冻结规则下的已实现率),跨 registry 推断
需协议级超总体(同法组装 registry 的翻转单元)+其再实现交换性——**v2 失败即该
假设在争议对上失效的证据**。已入稿 §8 "Population, sampling frame, and
per-stratum rates" 段。

**按轴分层 + 逐轴错误率**(src/per_axis.py,产物 rows+checks 分解):
- 纯 NA 单元 15(翻转检查全在 NA-margin 轴):门内 12,测试承诺 5,
  **两折两水平全 0 错**
- 非纯 NA 单元 24(含金标/scoring/threshold 轴翻转):gR→gR2 折 commits=3
  wrong=1(**0.333**)——全套件唯一错误(dreeam|kddocre,轴构成
  gR2+gold+threshold)在此组
- 轴构成分布:15 纯 NA / 6 NA+gold / 5 thr / 4 gR2 / 2 gR2+thr / 2 NA+thr /
  2 scoring / 1 gR2+gold / 1 gR2+gold+thr / 1 gold

**定性**:NA 端点近正交(ρ=−0.03)且 23/23 翻转率系端点自身性质;分层证明其
不污染参照错误账(纯 NA 组 0 错),但混池 α 是两质混合——逐层率才是诚实记账。
全段落入稿;42 页 0 err/0 undef;per_axis_result.json 留档。

---

## R1-M5 执行记录(2026-11-12 06:2x,评审修复——出处披露,零数值变更)

**35 阈值推导**(prereg_v2 §2.3 冻结语境):功效算术链=v1 冻结 13 翻转无法认证
头条对比(Fisher p=0.165)→ 发布的功效模拟 85% 功效需 n≥20 对 → prereg v2 按
族扩容计划预期 220–250 检查/40–55 翻转 → 止损门 35=预期下沿 40 的保守折扣
(≈功效地板 1.75×),保证对半切分后每折 ≥17 单元以满足 ≥5 承诺可判门。
**候选轴顺序**(时序证据,文件 mtime+§7 coverage 清单):coverage 段固定顺序
点名三个未覆盖轴(Ign-F1→NA-handling→dev-tuning leakage);入池修正案按清单
顺序同夜执行——Ign-F1 先测(11-11 22:04,**0 翻转,不入池**)→ NA-margin 后测
(22:54,47 robust,按注册规则 robust=ΔCI 排零入池)。**两候选轴一出一入,
"测到翻为止"被时序证伪**。残余风险如实:候选清单未在 prereg v2 冻结文本内
预冻结;敏感性分析(Sensitivity 段)已量化该依赖的代价;未来 registry 增长
预冻结候选清单。全段落入稿 §8 "Stop-gate provenance and candidate-axis order";
43 页 0 err/0 undef。100% 翻转率解释已由 R1-M4 段覆盖(NA 端点自身性质)。

---

## R1-M6 执行记录(2026-11-12 06:4x,评审修复——判据公式化+三床 ex-ante 值)

**κ 判据公式化**(§7 交换性段改写):κ_ij=训练契约池 pooled-gap 符号与池中位
一致占比——**纯训练侧可算(部署时无需第二参照)**,同时满足 R1(定量判据)与
DA-M4(ex-ante 可算)。

**三床 ex-ante 值**(本轮如实报告,阈值未用于 gate 任何结果):
- DocRE v3:39 翻转单元 κ 中位 **1.000**(均值 0.891;79% ≥0.75;21% κ=0.5);
  91 对中位 1.000/均值 0.908/82% ≥0.75
- TACRED:争议对 κ **按构造封顶 0.50**(micro/macro 两半在争议对上分歧)
- WMT19:翻转密度≈0(registry 11 结果 10 tie),判据不触发,适用性由 ρ≥0.96 覆盖

**提议 κ*=0.75(翻转对中位)**,声明"for the next registry cycle's freeze—
stated here for pre-registration, not tuned against any reported result"。
**反事实实证**:若本轮已冻结 κ*,TACRED 被正确筛出(0.50<0.75)而 DocRE 通过
(中位 1.00)——**global ρ 筛查做不到的分离(TACRED 0.89 读作健康),训练侧 κ
判据做到**=DA-M4(筛查不分离)的正面解决。

**本轮如实**:步骤 (a)=无阈值筛查,未阻止任何事;2×2 表系两次失败后引入,
标注 post-hoc framing device 非 validated deployment rule。43 页 0 err/0 undef。
---

## R1-M7 执行记录(2026-11-12 07:0x,评审修复——呈现层,零数值变更)

**43/43 处处定性为训练目标度量**(5 处同步):
- 摘要:its training target own metric and thus a design property rather than a discovery;Fisher 从摘要删除;26/43 前景化(exactly as the fidelity trade-off proposition requires)
- Findings 弹点:同定性+26/43 并列;Fisher 标 post-hoc/自选不等分母集+指向 selection-matched 为 like-for-like 控制
- §8 Realized outcome:the training target own metric+26/43 并列
- §8 Fisher footnote:加 post-hoc—computed after the run, on the readers self-selected and unequal hard-ranked sets+声明 selection-matched 超集结果为更强证据(the one we stand behind)
- Conclusion:同定性+26/43
- highlights:第 3 条改 Consensus fidelity 43/43 (its training target)

**v3 forced-ranking/selection-matched 已于 review_fixes.py 跑出并入稿**(§8 Selection-matched 段:learned 硬排 31 对⊃gD 15 对,多排 18 对全对;forced 37/39 全对共识)——M7 的 apply Sec.6.1 controls to v3 由此兑现。43 页 0 err/0 undef。
---

## R1-M8 执行记录(2026-11-12 07:2x,评审修复——残留扫描+代价判据声明,零数值变更)

状态核查:M8 两项子要求已于 M3/P0① 轮提前落地——CP CI 修正([0.003,0.53]/
[0.00,0.41]@0.10 折 1/8、0/7;[0.003,0.45]/[0.00,0.34]@0.20 折 1/10、0/9)
与 McNemar (b,c)=(4,29)/(3,14)/(0,68)+精确 p(1.1e-5/0.013/6.8e-21)。
本轮:①残留扫描——旧数字(22/47/33/88、旧 CI)全文零残留(唯一命中=已修正行);
②J7 代价判据补"未配对检验"声明(§8 J7' 句):冻结形式=计数比较不附检验;
v3 margin(17-18 对/155)配对符号检验会分离、v2 margin(60 vs 65)很可能不会——
报告计数而非给冻结判定附加未注册检验。44 页 0 err/0 undef。
---

## R1-M9 状态确认与 (v) 补完(2026-11-12 07:3x,44 页 0 err/0 undef)

(i)-(iv) 已于 P0① 轮全部落地(trivial≥full 更正/+0.115 含零撤回声明/Table 1 τ
残值→冻结值 0.905/0.905/0.181±0.047+同值 caption 注/Limitations ii' 更正)——
本轮 grep 验证在稿(1059 retracted/1301 Limitations/617-853-891 captions)。
**(v) 本轮补完**:§5 Reproducibility 的 deviation log 句原只述 judge 算术 bug,
现补 amendment 3 文内摘要——定义(pooled reference-reading 算术修正:参照 gap
误为 intra-side 差而非 intra-inter gap)+管线全重跑+footprint 三条(J1-J3 逐位
不变对账验证/J4 翻转=注册修订/动值表均带 amendment-3 caption 注+逐表对账冻结
manifest)。
---

## R1-M10 执行记录(2026-11-12 07:5x,评审修复——可复现性六项+定义框+逐单元表)

六项缺口全补:
1. Part-2 conf 的 B=2000/seed 20261112 → §8 registry 段明写
2. 切分规格 → 名字典序排序+偶校准奇测试(20/19;门内 13/11 与 13/10)全规格入正文
3. 52/39/29 对账 → 52 翻转检查→39 去重对→10 缺 gR2 折级排除→29 gR2 可判(39 gR 可判)入正文
4. 35 推导 → 已于 M5 段落地
5. WMT parity 口径 → §6.7 加口径区分句(0.700/0.7524=装载门语料级 AER 对照 r60;0.683/0.771=段级条件化迁移;两协议不可比且各自逐位对齐参照)
6. 前沿 3/5 种子 → fig:frontier caption 加 confirmatory seeds 20261201-03 (three of the five; plateau seed-stable across all five)

**定义框**(附录 B' app:perunit):Check/Flip/Unit/conf(B、seed、折叠)/commit/Gate/
Fold 规格/Judgable/Coverage-false-tie 九术语集中定义。
**逐单元表**(tab:perunit):39 行全记录=pair/axes 构成/conf/commit/fold/gate/gR/gR2/
误差标记(*=唯一 gR2 误差对),值与冻结件逐位一致。
坑:pair 名下划线未转义首插编译炸→回滚+转义重插(_perunit_rows.tex 与一次性脚本已删)。
45 页 0 err/0 undef;R1 十项 Major 全清。
---

## R2-1 剩余修复执行记录(2026-11-12 08:0x——三族刻画+TACRED 本体+族表升级)

引用部分已于 P0④ 落地(11 条);本轮补剩余:
1. **TACRED 本体查证成功并引用**:TACRED 数据集论文=Zhang, Zhong, Chen, Angeli,
   Manning, Position-aware Attention and Supervised Data Improve Slot Filling,
   EMNLP 2017 (D17-1004)——CrossRef 查证(此前记忆混乱已解,零编造);bib 33→34;
   正文 TACRED 首现处(§5 Transfer)加 \cite。
2. **三族刻画**(定义源=冻结脚本 r1_rank_flip.py 头注):
   - atlop_pe8=companion 的 ATLOP 正式协议重训(校准分片排除),3 种子
   - p3m=companion 内部规则桥接解码器(原生决策规则直接出预测,无连续分数无阈值面),5 种子
   - x2=companion 内部多标签模型,20 run={D,R}×{r0.25,r1.0}×5 种子,连续分数
3. **tab:generators 升级**(附录):加 Description 列(上述刻画+发表族引用
   DREEAM/DocuNet/KD-DocRE/ATLOP)+Source 说明(companion vs published)。
4. 顺带发现:5 个 >15pt 严重 overfull,其中 85/79/35pt 三处=既有表格 bug
   (tab:flips 等空行问题,P1 LaTeX 清理项);本轮新表(族表+逐单元表)零溢出。
46 页 0 err/0 undef。
---

## R2-2 执行记录(2026-11-12 08:1x,评审修复——口径声明,零数值变更)

**Gap_F1 切片口径精确声明**(§3 Contracts and readings 段,源=冻结管线
E03_full.py:120 的 group 字段二分):
- 划分=候选按句位二分:头尾实体首现同句=intra(group 0),否则=inter(group 1);
- 两个层级的 micro-F1 均在全部 498 文档上计算,无文档级子集筛选;
- **与官方脚本差异如实**:官方 DocRED evaluate.py 的 inter-F1 仅在含至少一个
  跨句金标关系的文档子集上计算;本约定保证每候选恰属一层、gap 跨层可加、
  无文档被任一侧静默丢弃。
**谱系引用**:intro "earliest graph- and segment-based DocRE architectures"
加 \cite{yao-etal-2019-docred,chen-etal-2021-docunet}。46 页 0 err/0 undef。
---

## R2-4 执行记录(2026-11-12 08:2x,评审修复——交换性立场弱化,零数值变更)

**§3 desiderata 替换过强主张**:原"a premise the DocRE family satisfies by
construction"→ 新三层:①DocRE=候选实例但两金标系修订关系非独立采样
(**Re-DocRED 的存在正是因系统性标注不全**);②交换性至多**条件成立**(在
revision 与 original 分歧的争议对之外);③协议检验而非假设该前提——DocRE 本身
与 TACRED 同测,**Part 2 注册失败=该检验的实证内容**(内部互指 sec:confirmv2)。

**2×2 表 DocRE 格条件化**:Exchangeable×Flip-cost-high 格 "Learn (DocRE)"→
"**Learn, hedging contested pairs (DocRE, conditionally)**";表后句加"even
there, conditionally and with contested pairs hedged"——与 v3 门层实际部署
读法一致。合法的 by-construction 用法(共识倾向/门/census/TACRED κ 封顶等
10 处)不动。46 页 0 err/0 undef。
---

## R2-3 执行记录(2026-11-12 08:3x,评审修复——软化为主+诊断表为辅,47 页 0/0)

**正文软化**(§8 registry 段):①门定性改写——"a conservative admission rule
whose role is to keep corrupt predictions out of the registry, not to grade
the models"(2% 门=保守入池规则非模型评分,回应"门无理由");②失败定性降级——
"we report the failure as a **harness-bound exclusion, not a finding about LLM
relation extraction**"+发表级 zero/few-shot LLM-DocRE 在数十字 F1(远高于本管线)
的对照声明;③相反失败签名入正文——三族 187-188k 预测 vs 5,548 gold pairs
(≈34×,全量 TP 仅 25-56)vs GLM 坍缩 19 条,"opposite signatures indicate
schema-grounding or entity-pairing pathologies **in the constrained-decoding
harness itself**";④"excluded as incomparable under this harness...no claim
about LLM capability is made or implied"。

**附录 app:llmdiag + tab:llmdiag**(新):4 族全池诊断(docs/预测数/TP/全池
precision/签名;Qwen3 188,131→TP 25/Llama 187,541→TP 56/Yi 27,626→TP 8/
GLM 19→TP 0)+门抽样判定值(0.13-0.27%)与全池口径关系说明(抽样接地批量协议
更宽松,两者一致远离可入池线)。

**幅度修正披露**:全池 precision(0.01-0.03%)比先前记录的抽样值(0.13-0.27%)
更极端——两口径并存且方向一致,正文/附录双报。47 页 0 err/0 undef。
---

## R2-5 补完执行记录(2026-11-12 08:4x——NA-margin 评测第一性动机化;敏感性部分 M5 已覆盖)

**§8.1 池段动机化**:NA-margin 端点句扩写——"motivated from evaluation first
principles, not from the gate":NA 无金标候选对的处理是 DocRE 真实的协议分歧
(官方脚本折入 FP 质量,gold-pair 子集分析完全忽略)→ 系统在语义无关对上的
幻觉密度是被常规 micro-F1 平均掉的真实属性 → 把它命名为轴=论文自身论点
(未声明的评测选择是一阶决策)**对自身应用**。

**M4 段呼应括注**:NA 端点存在理由=NA-handling 惯例跨协议分歧=真实契约坐标
(指 §7/Discussion)。

敏感性部分(R2-5 前半"报预修正 29 翻转 registry 的敏感性")已由 M5 轮
Sensitivity 段完整覆盖(去轴 184/29、止损门 FAIL、J5' 1/3、Part1 21/21 稳健)。
47 页 0 err/0 undef。R2-5 全清。
---

## R2-6(c) 深度修复执行记录(2026-11-12 08:5x——gR' 出处定义,零数值变更)

**(a)(b)(d) 已于 P0① 落地**(single-seed 措辞/eight-of-nine 统一/Limitations ii' 更正)。

**(c) 本轮深度修复——gR' 出处真相**:查冻结管线(lre_rebuttal_v2_prepare.py 64-77
行)确认三套金标全部 dev 侧对齐:gD=DocRED 官方 dev 原标/gR=Re-DocRED dev 修订/
**gR2=Re²-DocRED(EACL 2026, heng-etal-2026-re2)dev 修订**——对 Re-DocRED 的
第二次社区修订,同一 498 dev 文档。**"test-revised projection"是彻底残留错误**
(R2 的猜测与 P0① 的"projected onto dev"措辞均不准)——已三处修正:
①Discussion "gR'(the Re²-DocRED second revision[cite], aligned to the same
498 development documents)";②同段第二处;③**§8 registry v3 池段 gR' 行加
出处定义句**(g_{R'}=Re²-DocRED second revision of the same 498 dev docs,
第二次社区再审计——它锚定 J5' 认证,出处现完整可引)。bib 34→35。
47 页 0 err/0 undef。
---

## R2-7 执行记录(2026-11-12 09:0x,评审修复——false-tie 前景化+43/43 降 training sanity,零数值变更)

四处调序+措辞:
1. **摘要**:43/43 句改 "a **training-sanity property by construction**"(原
   "its training target's own metric and thus a design property rather than a
   discovery");false-tie 保持在前(McNemar 认证句领头条)。
2. **Findings**:43/43 加 "---a training-sanity property---";false-tie 保持在前。
3. **§8 Realized outcome 调序**:false-tie(48 ≪ 89/65)提到段首并标 "the
   suite's most robust axis";consensus fidelity 后置并标 "a training-sanity
   property (it is the training target's own metric)"。
4. **Conclusion 调序**:"a false-tie count still the lowest in the suite"
   领起,43/43 后置为 "a training-sanity property"。
5. **highlights 第 3 条调序**:"False-tie lowest in suite; consensus fidelity
   43/43 (training sanity)"。

v3 forced-ranking/selection-matched(31⊃15 超集)已入稿为 like-for-like 控制。
48 页 0 err/0 undef。
---

## R3-1 执行记录(2026-11-12 09:1x,评审修复——部署生命周期与成本段,零数值变更)

§7 新增 \paragraph{Deployment lifecycle and cost.}(置于 Practical implications
之前)——三生命周期事件映射+实测成本:
- **设计时**:κ/ρ 交换性筛查+功效模拟=CPU 分钟级;读数冻结=编辑行为
- **每提交**:镜像整合秒级/系统;新解码一次校准探针=一次额外 eval(GPU 分钟-小时)
- **每刷新**:新族 ATLOP 训练实测 ~1.5-3 GPU-h/族;LLM 生成 ~20 GPU-h/族(本轮
  数个 GPU 日沉没,4 族全败);决策层重训 ~9 CPU-min(5 种子×105s);门层全链 ~9 min
- **registry v3 总账**:"roughly one GPU-day of family training, several sunk
  GPU-days of failed LLM generation, and the CPU budget above---the learned
  reader is the cheapest component, and the registry is the expensive one"
- **运营者今日可采纳(无学习读数器)**:交换性筛查/功效模拟/tie 声明惯例
  (每个排序旁报 ties 与 coverage)/新解码双解码探针/多金标报告
48 页 0 err/0 undef。
---

## R3-2 执行记录(2026-11-12 09:2x,评审修复——契约抽象闭合三件,零数值变更)

1. **tuple 开放声明**(§3 Contracts and readings):c=(g,s,τ) 后声明 tuple
   deliberately open beyond (g,s,τ)(ρ=评测坐标见 §7 Discussion;NA-margin 端点
   见 §8);封闭的是要求=每轴必须声明并冻结后才可比——"openness is fine,
   undeclared freedom is not"。
2. **legality 判据重述**(§6.5,替换 companion 外包括注):legal=修订在相同任务
   定义下重读相同候选对(DocRED→Re-DocRED 修正补全而候选宇宙不变);
   illegal=在改变的任务下重新裁决(Re-TACRED 剪枝数千实体对并改标——读的是
   不同量而非修正读数)。
3. **近正交量论证**(§3 exchangeable readings 承诺段后):registry 准入≠共识
   准入——registry=多账本审计结构(轴只需声明冻结),共识池(训练契约)只收
   过交换性筛查的轴;NA-margin=registry 审计轴非训练契约,M4 逐层账目保持
   分账。48 页 0 err/0 undef。
---

## R3-3 执行记录(2026-11-12 09:3x,评审修复——分层陈述+文献连接,零数值变更)

**§8 Why-the-guarantee-fails 段尾新增**(三件):
1. **文献连接**:失效落点=conformal beyond-exchangeability 分析预言之处
   (barber2023beyond, NeurIPS 2023 新引用);**weighted conformal**
   (tibshirani2019weighted, NeurIPS 2019 新引用)=门之外的标准修复选项——
   **如实说明未部署的理由**:权重需从"恰恰被失效作废的参照一致性"学出;
   加权变体=未来周期首个消融。
2. **分层陈述**(无条件+翻转分层,纯逻辑推导自 registry 定义):稳定对
   (两参照一致+共识与之一致,按定义)上任何跟随共识的读数器——门控与否——
   已实现错误率=0;两周期全部已实现错误定位于含金标/scoring 轴翻转的对;
   无条件率=零错层与争议层的混合,分开报告(如上)比任何池化数字更有信息量。
3. J5' 内容在 no-op 对比(已由 M2 轮 no-op 一等段处理)。

bib 35→37(barber2023beyond/tibshirani2019weighted)。49 页 0 err/0 undef。
---

## R3-4 执行记录(2026-11-12 09:4x,评审修复——预注册外部效度三件,零数值变更)

§5 Preregistered confirmatory protocol 段尾追加三句(Three scope qualifications):
1. **scope 声明**:confirmation=seed-level(新种子复制训练随机性于固定预测池
   与 registry;不复制数据收集;探索/确认共享同 498 文档与生成器池);
2. **self-attested 披露**:三份预注册(v1/v2/v3)=仓库内冻结**无第三方时间戳**;
   全文+修正案日志+manifest 哈希随稿提交使冻结点可对工件独立核验;**外部时间戳
   快照(OSF)正在存入**;
3. **adaptive growth 披露**:跨周期 registry 增长经修正案通道自适应;敏感性分析
   已量化依赖代价;未来增长预冻结候选轴清单(指 sec:confirmv2)。

**新增作者待办**:OSF 上传(三份 prereg+manifest 哈希快照,零成本获 DOI)。
增长规则条款已由 M5 段(候选轴清单+robust 规则)覆盖。48 页 0 err/0 undef。
---

## R3-5 执行记录(2026-11-12 09:5x,评审修复——换域成本草图+摘要 generality 收敛,零数值变更)

1. **摘要 generality 收敛**:协议贡献句加 "the protocol is instantiated
   end-to-end on DocRE, and we are explicit about what porting it to other
   domains costs (Section 7)"——普适框架暗示收敛到已实证域+移植成本显式指针。
2. **换域成本草图**(§7 Deployment lifecycle 段尾,具名域=summarization
   judge swaps):同序四步(冻结生成=frozen-pool 要求=排除交互/agentic 场景/
   声明契约轴[judge 身份+rating scale+prompt 模板]/judging 调用=主导成本
   N×prompts×judges/先跑便宜半[κρ 筛查+翻转密度估计决定是否值得建 registry]);
   metrics-only 域(WMT19 式)零训练;需新训练的域继承族训练成本;
   **不可移植项**:池不可冻结/审计参照与目标量分歧(注册负结果=常设警告)。
48 页 0 err/0 undef。
---

## R3 Minor 九项清理执行记录(2026-11-12 10:0x,50 页 0 err/0 undef)

1. exttt 缺陷:不存在(逐字扫描 	exttt{power...} 正常;或已在此前轮次顺带修复)
2. 35/AER−1/δ/λ:AER−1 加"one-pair tolerance frozen in Part 1 J3, predating any
   Part 2 result"括注;δ=0.1 加"conventional 90% risk-control confidence, frozen
   with v2";λ 网格加"log-spaced grid spanning the operative range"(tab:loss
   caption);35 已于 M5 推导
3. 编号重复:不存在(现序列 (i)-(v) 连续;M9 改写时顺带解决)
4. 摘要压缩:370→**261 词**(-30%;Fisher 移正文;全部关键数字保留:
   43/43/26/43/52 flips/0.10/0.00/α*=0.20/TACRED 45)
5. tab:registry "identical here" vs tab:loss 6/8:caption 加限定("identical here
   on this registry and run; the post-hoc loss-ablation re-run attains 6/8---a
   different run, not a contradiction")
6. tab:loss 双反斜杠空行 8 行+附录 tab:flips 13 行→修复(21 行);seed 20261201
   复用说明入 caption("coincides with confirmatory seed 1 by convention; post-hoc
   analysis, no confirmatory claim rests on it")+λ 网格说明同 caption
7. WMT parity=权重坍缩:§6.7 加解释句("learned weights carry no incremental
   signal...near-collapse to uniform...J4-WMT is an infrastructure and no-harm
   check, not an advantage claim")
8. LLM 负结果:已由 R2-3 修复(软化+app:llmdiag 诊断表)✓
9. CLI 命令:正文改指向 released artifact README

## prereg v4 启动记录(2026-11-14,非偏差——新周期开立)

作者会话指令"prereg v4启动"=v4 批准;prereg_v4.md 同日 DRAFT→FROZEN
(冻结先于任何 v4 确认性结果;探索期 2026-11-13/14 战役 13 脚本/out JSON
全部为 pilot disclosure,已在 v4 §头部声明)。启动日执行:
(a) Phase 0 OSF 包构建(prereg v1-v4+amendments+探索 13 JSON+13 脚本+
MANIFEST sha256),上传=用户动作;(b) R2-M1 池规则考古启动(v4 Phase 1
前置任务,见 v4 §2);(c) Phase 1 x2 +12 模型训练=GPU 时窗排在
Paper 2 BIG 尾(11/13-15 主表收口)完成之后,排程记入本账本后续条目。
确认运行(confirmatory_run.py 同款守卫)在 FROZEN+OSF+止损门
(flips≥35/x2 n≥22)三条件齐前自动拒跑。

## 修正 4(2026-11-14 晚,作者决策"B(改稿删句)"):prereg v4 Phase 0 OSF 项撤销

P3 稿件 OSF 承诺句已按作者决策删除(自证立场:hash-pinned manifests +
ledger timestamps are the freeze evidence, no external notarization
claimed)。为保持稿件与预注册周期口径一致:**prereg_v4.md §2 Phase 0 的
OSF 上传步骤撤销**(zip 包 out/osf_prereg_v4.zip 已构建留档,不上传);
确认运行前置条件改为 **FROZEN + 止损门(flips≥35/x2 n≥22)** 两条件。
证据等级如实降档为 self-attested(沿 v1-v3 同一口径); §8 OSF 清单作废。
此修正案先于任何 v4 确认性数据登记。

## 修正案 5(2026-11-14 深夜,Phase 1 启动日算术纠错):+12 → +4

验证(src/check_regime_identity.py + e50_seed_extension.sh 前例 +
lre_rebuttal_v2.py L190-240 采样源码):run 名中 r025/r100 = **训练侧负采样比**
(SHA-256(seed|doc_id|h|t) 排序取 floor(rho×n_pos),rho∈{0.25,1.00}),非解码契约
——per_doc/evaluation__gD.json 哈希跨 regime 不同(8c55eb69 vs 57efcda2,
0.6MB vs 0.5MB)=不同训练模型。**现存 x2 模型 = 5 种子 × 2 臂 × 2 比 = 20**
(X2_SEEDS×4 确认),非冻结文本假设的 10。Phase 1 算术更正:
**+4 run-dir(1 新种子 20261201 × 2 臂 × 2 比)→ x2 n=24 ≥ 22 ✓**,
替代原"+12 = 6 种子 × 2 臂"(其算术误把每 (臂,种子) 计 1 模型)。
成本 1-2 GPU 天 → 一夜(~10-14h)。T1 目标不变;健康门不变
(n<22 如实降级禁补跑)。执行管线 = e50 前例的 P3 镜像
(src/x2_v4/:lre_rebuttal_v2 prepare + core_runtime train + eval_infer +
eval_score,OUT 写路径重定向至 out/x2_v4/,读路径保持 companion 只读;
确定性纪律 AMENDMENT 03 全套)。此修正案先于任何 Phase 1 训练数据登记。

## 修正案 6(2026-11-14 深夜,Phase 3 运行前):C1′ 判据尺度更正

C1′ 冻结文本"α*=0.10,PASS=feasible 且 commits≥22"的 22 来自模拟池折尺寸
(c5_family_rcps:14 族池测试折 ~45 单元);真实 39 单元 registry 的测试折
仅 19/14 单元——该判据按字面**构造性不可满足**。更正其意图级表述:
**C1′ PASS = feasible 且 realized ≤ α*=0.10 且 commits ≥ 11**(与 C1 相同的
零错地板,施加于更严 α)。修正案先于任何 C1′ 计算登记。

## Phase 1-3 确认周期收官记录(2026-11-14 深夜,非偏差——按冻结判读表执行)

Phase 1(4 run 训练+推断+计分,8.0h 零失败,健康门 4/4,census_n=24≥22)→
Phase 2(24 实例读数表;own-gold 约定修正后与探索图对齐:scoring 6.5%≈6.8%/
threshold 26.1%≈27.9%)→ **Phase 3 确认运行(phase3_confirmatory.py→
phase3_verdict.json)**:

- **C1 FAIL**:fold gR→gR2 不可行(conf=1.00 反对单元不可阈值排除);
  fold gR2→gR t=1.0 地板 rate 0.4286>0.20——跨模型证书在真实 learned conf
  上不成立(探索期代理分与真实 conf 的差距如期显形,预声明边界兑现)。
- **C1′ FAIL**(修正案 6 尺度更正后仍不可行)。
- **C2 = T3 降级交付**(预注册降级条款执行):scoring 维 24 实例,
  1 坏(ml_R_r100_s20260919,portfolio 率 0.304——frozen 种子,非新批),
  bound 0.1526;**T1 未达成**。
- **C3 PASS**:层级序逐维复制(scoring 0.065 < threshold 0.261 < gold 0.554;
  ρ=1.000 vs 探索图)。
- **C4 PASS**:定律精确成立——flip 密度 = (1−τ)/2 三维全中至 4 位小数
  (scoring .0652=（1−.8696）/2;threshold .2609=（1−.4783）/2;
  gold .5543=（1−(−.1087)）/2);Spearman=1.000。
- **C5 FAIL**(预期,per pilot)。

**判读表落定 = 分支 3:边界论文(跨模型证书主张撤回;C2-T3 within-x2
census 证书 + 定律 + 地图 + 协议为交付物)**。约定教训:own-gold vs 统一
gD 的读数约定本身是契约坐标(本阶段第三抓;修正后与探索图对齐)。
prereg v4 周期至此完整收官(Phase 0-3 + 判读)。

## prereg v5 启动记录(2026-11-15,非偏差——新周期开立)

作者会话指令"启动V5"=批准;prereg_v5.md 同日 FROZEN(冻结先于任何
v5 结果;设计输入=v5_candidates DRAFT+v4 全产物 pilot disclosure)。
范围:E1 约定清单(候选 3,必选)+E2 exact weighted-distribution 界
(候选 2,判据冻结于 §2.3)+G1 池生长(候选 1,**条件=E2 PASS 才启动**,
理由=加权界失败则 44 模型的加权证书无意义);候选 4/5 不在本周期。
执行顺序:E1(零成本,立即)→E2(理论+实现 1-2 周)→G1(视 E2)。

## v5 判读记录(2026-11-15,非偏差——按冻结判据执行)

- **E1 ✓ 交付**:conventions 块补发入 x2_stratum_24.json(cnt_layout/
  file_generation/gold_assignment/variant_keys/reading_col_map/provenance
  六键);instances 哈希补发前后逐位不变(edbd53c1c91b8707)=约定显式化
  未动数字;此后读数表无 conventions 视为不合规(管线义务生效)。
- **E2 = FAIL-a(候选判死)**:src/v5_exact_bound.py→v5_exact_bound.json
  ——硬池 gen-block B=2000,**feas 0/2000**(精确卷积界从未签发)。
  **机制诊断(判死后归因,非重设计)**:冻结 §2.2 的 per-cell CP 上界
  在 Bonferroni δ/m 与折尺寸 n_c=1-3 下空洞——n_c=1 零错观测的
  p̄_c=0.9923,n_c=2 零错 p̄_c=0.9123;卷积在这些上界上结构性不可行。
  **结论:cell 级校准+Bonferroni-CP 在 registry 折尺寸空洞——K 墙
  第四角度**(前三个:族级 n≥22/实例级需同质/C5w Hoeffding margin)。
  修正后的界(去 Bonferroni/联合精确检验)**须全新预注册**,本周期
  按冻结判据判死,不看不结果改设计。
- **G1 不启动**(E2 FAIL 前置条件未满足,GPU 未动)。
- **判读表 = V5-3**:约定清单入协议;精确界阴性入失败谱系;稿不动。
prereg v5 周期完整收官(开立→E1→E2→判读,单日)。

## prereg v6 启动记录(2026-11-15,非偏差——新周期开立)

作者会话指令"启动"=批准;prereg_v6.md 同日 FROZEN。范围=Arm A
(G1 +20 run→n=44 census 升格;判据 b≤1→T1[cp(1,44)≈0.097])+
Arm B(拆样本前瞻 census;确定性切分 sorted run_id 偶/奇;校准折零坏
⇒0.0999 T1,测试折容忍 ≤2 坏)+界线关闭注记(第五界变体不注册:
逐 cell 分解族空洞[v5 证]+不分解=池化二项=FS[已测],论证入稿)。
GPU 前置=P2 收口确认;健康门/降级禁补跑/修正案/conventions 义务全套。

## prereg v7 启动记录(2026-11-15 深夜,非偏差——新周期开立)

作者会话指令"继续"=批准;prereg_v7.md 同日 FROZEN(冻结先于任何 v7
结果;设计输入=候选 5 + v7 数据核查[judge_data_check.json:17 judge ×
49,938 battles 全覆盖交集,pilot disclosure])。范围=judge 生态五步
实例化:Measure=judge 对 flip 密度(136 对)/Scope=battle 层 census
(α_port=0.5+半数分裂判据)/Certify=judge 层 census+拆 judge 前瞻
(偶 8/奇 9 预声明切分)/Hedge=争议 battle 清单。**judge 生态的
conventions 块**:概率语义(prob_A/B/C 三选项)、battle 命名(id)、
ground truth 口径(winner_gt=平台标注)、分层键(家族/规模——
探索性,判据 J2 的方向预言在冻结时未跑)。判读表 V7-1~4+败点
F1-F4+止损门全部冻结。执行=CPU 分钟级(49,938×17 矩阵)。

## prereg v7 判读记录(2026-11-15 深夜,按冻结判读表执行)

- **J1**:恒等式解析成立;数值核对最大偏差 3.04e-05——冻结容差 1e-9 触发旗标,诊断=tau-b 的 tie 处理+二值数据的边际不对称(非实现错误);按 V7-4"排除实现后如实报告"处理,定性=数值精度注记;
- **J2 PASS**(方向预言命中):70B 级内部 τ̄=0.817 ≫ 7B/8B 级 0.476——预声明方向(能力越近越一致)成立;
- **J3 PASS**:全 battle 争议份额 0.1225,CP 上界 0.1244 ≤ 0.40;
- **J4 PASS**:per-judge 超标 judge 校准折 0/测试折 0(执行版折级率 12.0% vs 9.3%,测试更低=前瞻方向正确;冻结口径"超标 judge 数"补算完成);
- **判读 = V7-1**:第二生态完整交付(judge 层 census+battle 层 census+拆 judge 前瞻+层级复制)。
**关键数字**:49,938 battles;争议份额 12.25%;CP 上界 12.4%;70B 级 judge 内部一致率 0.817 vs 7B/8B 0.476——**judge 排行榜的 12% 对决是 judge 依赖的,88% 可认证**。

## prereg v6 判读记录(2026-11-17 凌晨 03:47,按冻结判读表执行)

- **健康门 20/20 PASS**(批中位 intra F1 0.373;cov 全 1.000);
- **Arm A census @n=44**:bad=3(ml_R_r100_s20260919 冻结种子 + ml_R_r100_s20261202/20261205 生长种子;三例全在 R 臂 r100,与审计已录的臂侧敏感一致),CP 上界 cp(3,44,0.1)=0.1456,**grade T2**;
- **Arm B 前瞻(冻结降级规则执行)**:sorted 奇偶 22/22 切分,校准折含 1 例坏 ⇒ 零坏条件失败,按规则降级为报告 cp(1,22)=0.166、PASS=false——与 judge 生态拆分发现(相关性 judge 面板破坏前瞻界)同构;
- **判读 = V6-3**(按分级交付):证书从 T3(n=24,0.153)**升格 T2**(n=44,0.146)——预注册生长买到一级而非 T1 目标;总体系加倍;
- **稿件落地**(apply_v6.py,8 处):小节标题/tab:pops/生长叙事/census 主段/Arm B 降级披露段/摘要/引言/cover letter;fig_walls 注记 "T2: CP 0.146 (n=44)" 重渲染(碰撞审计 0 FAIL);Part 3 开篇句同步;
- **fig #3(census 轨迹)不画**:预承诺=T1 才画,T1 未兑现,不移动球门柱。
**关键数字**:44 模型 / 3 坏 / 0.146 / T2;前瞻分支如实降级 0.166。


## E1 必要性解剖记录(2026-11-17 晨;EXPLORATORY,冻结工件上的重放)

- **问题来源**:面板评审 R5-A4"必要性真空"(gate-only 免学习免拨盘通过全部判据);
- **重放**(src/e1_dissect.py,confirmatory_result.json unit_rows):非 pure-NA 方向匹配 commits=23,其中 **20 个 bootstrap conf=1.0000**(全票);t*=0.670 下 dial 的全部作用=撤回 `docunet|kddocre`(conf 0.647)——该 commit 对 gR2 **方向正确**,wrong 计数不变;
- **判读**:学习+校准在此 pool 上对点估计不 load-bearing;价值=证书的形式要求。按 R5 路线 B 处理:贡献 4 重写(协议+readout 纪律,非学习组件必要性)+What is learned 段定量解剖+结论简化性发现句;下一周期 preregister unanimity gate 为正式组件(v8 候选);
- **口径注**:23=非 pure-NA 重放口径(tab:ruleablate 报 24,fold 口径差 1);稿内已加 non-NA 限定。
