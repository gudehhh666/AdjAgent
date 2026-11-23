# 领域依据与项目映射

检索与官方页面核对：2026-09-06。对象：AdjAgent 的佐剂概念设计报告。没有在本次检索中找到统一的官方“佐剂设计报告模板”；下述组合是项目设计选择，不能表述成 WHO/EMA 规定的八章模板。

| 来源 | 版本与查阅范围 | 采用的原则 | 项目内转化 |
|---|---|---|---|
| [WHO TRS 987 Annex 2](https://cdn.who.int/media/docs/default-source/biologicals/vaccine-standardization/trs_987_annex2.pdf) | 2013 年通过，TRS 2014；读到全文，重点目录、A、B、C、D；PDF 42 页 | 佐剂依据、组合制剂、免疫和非临床安全、模型适用性需结合具体产品评价 | 配方依据、免疫证据、验证与外推边界四类核查 |
| [WHO 产品目标框架](https://www.who.int/our-work/science-division/research-for-health/optimizing-research-processes/product-profiles) | 官方现行网页，未以抓取时间冒充发布日期 | TPP 可区分最低与理想属性；PPC 是较早期的战略性偏好 | 用目标矩阵标记题设硬约束、偏好、缺失阈值 |
| [EMA 佐剂指南状态页](https://www.ema.europa.eu/en/adjuvants-vaccines-human-use-scientific-guideline)及[指南](https://www.ema.europa.eu/en/documents/scientific-guideline/guideline-adjuvants-vaccines-human-use-see-also-explanatory-note_en.pdf) | EMEA/CHMP/VEG/134716/2004，2005 生效；状态页 2023 更新。状态页全文可读；PDF 打开失败，搜索索引可见第 3 节目录和质量摘录 | 官方明确第 4/5 节已被替代，第 3 节质量部分仍有效 | 只据第 3 节结构补足单独佐剂、抗原组合和终产品质量；不使用旧非临床/临床章节作现行要求 |
| [NIAID Generic Vaccine PDP](https://www.niaid.nih.gov/sites/default/files/pdp-generic-vaccine-2023.pdf) | 2023 文件名；检索索引读到第 25 页 §3.2.3.1，PDF 直接打开失败 | 佐剂的增量价值和制剂属性均影响研发选择 | 记录 added benefit 与制剂关键属性，而非仅写通路名称 |
| [ARRIVE 2.0](https://arriveguidelines.org/arrive-guidelines)及[作者解释文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7360025/) | 2020；官方网页、Essential 10 和相关条目 | 实验组/单位、样本量依据、纳排、随机化、盲法、终点、统计及结果透明报告 | 用于验证计划的设计检查；不把计划写成实验已经实施 |
| [WHO 保护相关指标说明](https://www.who.int/publications/i/item/WHO-IVB-13.01) | 2013，官方出版说明页 | 免疫指标与保护结局之间需要可辩护的联系 | 抗体滴度、功能与保护分别登记，不静默互换 |

## 几个明确的领域取舍

1. WHO Annex 2 Part B 要求佐剂选择结合希望获得的免疫反应，并指出对一个抗原的效果通常不能直接外推到另一个抗原。我们据此增加“研究条件一致性”和“迁移缺口”字段。
2. EMA 的质量维度与 NIAID 对 formulation 的强调，使“组分名称 + 机制”不足以作为完整设计。我们增加制剂身份、兼容性与质量属性条目；没有数据时登记待验证，不自动推定合格。
3. ARRIVE 原本是动物研究报告指南。这里只将其用于提前检查验证设计，不能宣称纯文献设计已符合完整 ARRIVE，更不能据此填造结果。
4. 八章标题、R/D/V 标签、检索停止条件、300–500 字摘要、最多两轮自查和 Host 渲染方式，全部是本项目的工程约定。它们不是监管阈值，需通过 AdjAgent 实际运行和人工评审继续调整。

## 检索范围与停止原因

检索覆盖 WHO 佐剂非临床评价、TPP/PPC、EMA adjuvants guideline 状态与质量章节、ARRIVE 2.0 及 NIAID vaccine PDP。先检索再打开官方来源，遇到 PDF 打开失败后使用原站搜索索引并保留覆盖限制。没有采用过期草案替代正式文本，也没有使用二手媒体作为规范依据。

这些材料足以形成概念设计输出维度；未检索特定国家的完整申报要求，也未验证任何具体候选配方的科学优劣。后者由执行任务的 AdjAgent 针对题目继续检索。
