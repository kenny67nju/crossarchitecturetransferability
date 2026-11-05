# 参考文献修正与中英文 PDF 最终验收

## 结论

本轮已完成审查清单中的参考文献修正，更新 36 个已引用条目，并重新编译中英文完整 camera-ready PDF。论文正文和实验数据未改动；新旧 PDF 的英文正文 10 页、中文正文 9 页逐页提取文本完全一致。审查范围为中英文实际引用并集 61 条；未引用的历史 BibTeX 条目保持不动。

## 已完成的修正

- 实质错误：LogME 第二作者改为 Yong Liu；PACTran 的 LNCS 卷号改为 13694；Aircraft 作者顺序改为 Maji、Rahtu、Kannala、Blaschko、Vedaldi。
- 保护 RankMe 大小写；补全 Léon Bottou 等已核实的姓名写法；STL-10 作者顺序保留正式 PDF 首页的 Coates、Lee、Ng。
- EuroSAT 改为 IEEE JSTARS 12(7):2217–2226, 2019；LiTS 改为 Medical Image Analysis 84:102680, 2023，并加入正式 DOI。
- Task2Vec、DINO、Ranking Neural Checkpoints、ConvNeXt、GBC、OTCE、NCTI、MoCo 共八篇统一为 IEEE 正式出版元数据页码，并加入 DOI。CVF 原页码有来源，本轮属于版本统一，并非把这些条目一概判为抄写错误。
- MoCo 新增核查依据：IEEE 向 Crossref 登记的 DOI 10.1109/CVPR42600.2020.00975，页码 9726–9735；原始返回记录保存在本轮结果目录。
- 补全 Xu 五位作者、DeepDRiD 期号、iBean 2020 年、Food-101 / DTD / STL-10 / Flowers / Pets / VNE 页码和适用的 PMLR 卷号。
- DiSCO 使用正式 Proceedings of the AAAI Conference on Artificial Intelligence 39(21):22560–22568, 2025。
- SVHN 使用 2011 年历史名称 NIPS Workshop；VTAB 改为预印本 article 类型，不再出现“In arXiv preprint”。没有确认正式发表替代版本的预印本保留原始文献信息。
- SVCCA 按官方 BibTeX 保留 NIPS 2017 卷 30，删去未取得一手目录佐证的页码。
- timm 用兼容当前 BibTeX 样式的 misc 类型并注明 Software，消除不支持 software 类型的警告。

完整逐字段新旧对照见 `results/reference_corrections_20261104/reference_correction_acceptance.json` 与 `references.patch`。

## 旧版保护

备份：`archive/camera_ready_before_reference_corrections_20261104.tar.gz`。

包含修正前的 references.bib、两种语言的论文源文件、checklist、六个 PDF、构建/验收脚本、原验收 JSON 以及 .gitignore。已逐文件验证归档内容 SHA256 与备份前一致。没有覆盖已有历史归档。

## 编译与页数验收

| 检查项 | 英文 | 中文 |
|---|---:|---:|
| 正文结束页 | 10 | 9 |
| 致谢及利益冲突披露开始页 | 11 | 10 |
| 参考文献数量 | 60 | 45 |
| 参考文献页范围 | 11–14 | 10–12 |
| 附录开始页 | 15 | 13 |
| Checklist 开始页 | 20 | 17 |
| 完整 PDF 页数 | 26 | 19 |

- `Paper06_camera_ready_en.pdf` / `Paper06_camera_ready_zh.pdf` 均已包含正文、披露、参考文献、附录、checklist。
- `combined_en` / `combined_zh` 与对应完整 PDF 字节一致，是兼容文件名，不需要再手工合并。
- `Paper06_supplement_en.pdf` / `Paper06_supplement_zh.pdf` 只是同步更新的附录提取版，不包含下载的外部文献。
- 中英文引用集合在本轮前就不相同；本轮修正共享书目，不擅自增删正文引用。

## 自动核查

1. 新构建 BibTeX 日志无错误、无警告。
2. 无未定义引用、重复标签、缺字符或 Overfull 溢出。
3. 全部实际引用键均对应生成的 BibTeX 条目；未悄悄增删条目或修改未引用历史条目。
4. 三处关键书目修正及正式版本信息已核到新生成的 .bbl 和实际 PDF，不只检查 .bib。
5. 新旧正文逐页文本一致；论文源文件未改动。
6. 原有 `code/audit_camera_ready_pdfs.py` 通过：已审定主表数字、每池任务数、页数限制、披露、附录/checklist 顺序、字体嵌入和 PDF 链接检查通过。
7. 此次未重新运行实验；上述数据检查是对现有审定结果与论文的回归验收，不应解释为重新验证全部历史实验。

仍有非阻断性 LaTeX 提示：XeLaTeX 忽略 inputenc、浮动位置从 h 调整为 ht、部分 Underfull，以及中文版 Arial Unicode 无斜体形状而使用默认字形。没有因此缺字；这些提示在结果 JSON 中原样记录，不宣称构建“零警告”。

## 视觉核查

已渲染并查看英文第 10–15 页、中文第 9–13 页，覆盖全部参考文献页、正文结束、披露以及附录开始。作者顺序、卷号、RankMe 大小写、页码、DOI、中文重音姓名显示正常，未见文字遮挡、裁切或页面越界。总页数和章节分界未因书目修正改变。

## 下载材料隔离与证据边界

- 根 .gitignore 已加入 `/paper/reference_audit_*/`；已验证目录中没有 Git 已跟踪文件，下载 PDF 和审查工作文件均被忽略。本轮没有执行 commit 或 push。
- 新 PDF 没有 EmbeddedFiles，构建程序仅编译论文并提取附录，不会扫描或打包下载目录。备份也使用明确文件清单，不包含第三方论文。
- 仍是 55/58 篇论文已取得完整 PDF；USPS、器官定位、BUSI 的 PDF 未取得，其中 BUSI 已取得全文 XML，其余使用出版记录核查。此次修正完成不代表这三份 PDF 已下载，亦不宣称每一条都做过原 PDF 全文核验。

## 最终文件

- 英文：`paper/Paper06_camera_ready_en.pdf`
- 中文：`paper/Paper06_camera_ready_zh.pdf`
- 本轮验收：`results/reference_corrections_20261104/reference_correction_acceptance.json`
- 主验收回归：`results/camera_ready_p0_20261104/pdf_acceptance.json`

结论：本轮审查中已确认的参考文献问题已落实修正；新编译 PDF 通过书目、页数、链接和正文不变性验收。
