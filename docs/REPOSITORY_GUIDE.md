# 文件导航与整理说明（2026-11-05）

## 最新论文

- 英文完整提交版：[`paper/Paper06_camera_ready_en.pdf`](../paper/Paper06_camera_ready_en.pdf)。
- 中文完整译稿：[`paper/Paper06_camera_ready_zh.pdf`](../paper/Paper06_camera_ready_zh.pdf)。
- 两份均含正文、披露、参考文献、附录及 checklist。
- `combined` 是兼容副本，与对应完整版本逐字节相同；`supplement` 只是可选附录摘录。
- `paper/main_v18.tex`、`paper/main_v18_zh.tex` 是当前源文件；文件名保留 v18 不代表旧投稿内容。

## 目录职责

| 目录/文件 | 用途 |
|---|---|
| `paper/` | 当前论文、源文件、图、表、摘要与修改记录 |
| `code/` | 评分、分析、构建和验收脚本；当前入口以根 README 为准 |
| `results/p0_vgg_pair_20261104/PUBLIC_RELEASE_MANIFEST.json` | 公开的 11 个最终输入/脚本快照及哈希，不是完整实验包 |
| `results/camera_ready_p0_20261104/` | 最终表图数据、输入哈希和验收结果 |
| `results/camera_ready_review_20261104/` | 正文使用的阶段诊断与相关性结果 |
| `results/camera_ready/` | 保留的 VTAB/EX 输入和历史审计，按当前输入哈希选择文件 |
| `docs/` | 数据集来源、验收报告、后续研究及本说明 |
| `archive/paper/` | 历史稿件与 rebuttal，仅供追溯 |
| `archive/local/` | 本地下载论文、备份与构建残留；Git 忽略，不推送 |
| `tmp/` | 本地临时渲染与文本；Git 忽略，不推送 |

`results/` 中既有实验路径不整体移动，避免破坏分析脚本和历史来源记录。当前发布入口与最终支持集以根 README 为准，不直接运行历史 `reproduce_all.py` 来复现 camera-ready。

## 本轮移动（内容原样保留）

| 原位置 | 新位置 |
|---|---|
| `paper/REBUTTAL_*`、`paper/FINAL_COMMENT_TO_AC_*` | `archive/paper/rebuttals/` |
| `paper/main_v18.pdf`、`paper/fig_pertask_rho_24m.pdf` | `archive/paper/history/` |
| `paper/main_v18.aux/.bbl/.blg/.fdb_latexmk/.fls/.log/.out`、`paper/missfont.log` | `archive/local/latex_build/` |
| `paper/reference_audit_20261104/` | `archive/local/reference_audits/reference_audit_20261104/` |
| `archive/camera_ready_before_*.tar.gz`、`archive/references_before_history_completion_*.bib` | `archive/local/backups/` |

历史验收文档保留当时记录的路径，不批量重写其历史描述。主论文 PDF、核心源码及最终结果未因整理而改动。实验中间数据仍保存在本地原路径，并由 `.gitignore` 排除；不删除原始数据。
