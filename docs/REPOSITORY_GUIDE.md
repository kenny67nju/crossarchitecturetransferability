# 最终公开版本导航

## 论文

- [英文完整 camera-ready PDF](../paper/Paper06_camera_ready_en.pdf)：正文 10 页，总计 26 页。
- [中文完整译稿](../paper/Paper06_camera_ready_zh.pdf)：正文 9 页，总计 19 页。

两份都包含披露、参考文献、附录和 checklist。当前源文件仍名为 main_v18.tex / main_v18_zh.tex，不代表旧投稿稿件。combined 和 supplement 中英文四份 PDF 仅本地保留，构建时可重新生成。

## 公开范围

| 路径 | 用途 |
|---|---|
| paper/ | 最终中英文 PDF、LaTeX、参考文献、矢量图和表格 |
| code/metrics/ | RankMe、SA、SSR‡、官方 DiSCO 评分核 |
| code/final_selection.py | 最终固定支持集上的选择与统计辅助函数 |
| code/build_p0_camera_assets.py | 最终结果到论文表图的入口 |
| results/p0_vgg_pair_20261104/ | 最终分数、共同支持、对照及 9 文件来源哈希清单 |
| results/camera_ready/ | 最终 VTAB/EX 输入和参考消融 |
| results/camera_ready_p0_20261104/ | 表图对应数据与 paper_values.json |
| results/camera_ready_review_20261104/ | 论文仍使用的阶段、相关性和预处理诊断 |
| results/disco/ | 官方 DiSCO 的 EX 分数和分池汇总 |
| docs/DATASET_SOURCES.md | 原始数据集文献和任务别名 |
| docs/RELEASE_VERIFICATION.md | 最终版本复核步骤及边界 |
| RELEASE_FILES.txt | 允许公开的文件清单 |

日期型目录名为保持输入哈希和脚本兼容而保留，并不表示其中都是中间结果。
两份 historical_*.py 是用户确认保留的提取来源证据；不作为当前提取入口，也不执行。
历史修复脚本所需的最终纯计算函数已提取到 code/final_selection.py。

## 仅保留在本地

archive/ 整个目录、旧 rebuttal、过程性审计报告、旧评分表、一次性补跑/迁移脚本、特征缓存、日志、下载的第三方 PDF 和备份均不属于当前公开版本。
通过取消 Git 跟踪保留原路径文件，不删除本地原始资料。
.gitignore 采用公开文件白名单，新增发布文件需同时更新 RELEASE_FILES.txt 和忽略规则。

这次清理只改变分支最新快照，不改写 Git 历史；旧提交中的文件仍可被访问。
