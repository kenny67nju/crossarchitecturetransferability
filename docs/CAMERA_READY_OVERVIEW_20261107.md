# 方法总览图与正文解释（2026-11-07）

## 修改范围

- 在中英文引言加入 `paper/fig_method_overview.png`，作为 Figure 1 / 图 1；两版共用英文图面，图注及正文解释分别使用英文、中文。
- 原宽度散点图、几何诊断图和两阶段图依次顺延为图 2、3、4；原数据和图文件未变。
- 引言说明应用输入：无标签目标数据、固定且与目标分离的参考数据、异构候选编码器。每个候选分别编码两组数据；只在同一编码器内部计算对齐度，跨编码器比较标量分数。
- 理论解释联系图 1(a)：秩一目标保持奇异值不变而改变相对参考方向；示例取 k=1，不更改实际实验 k=20。这个例子说明额外几何信息，不推断准确率排序。
- 保持 SCR 分数族与 SSR‡ 默认实例的区别。图内公式为原有定义，柱形图只重述已有 22 单元的 RankMe 0.133、SSR‡ 0.069。
- checklist 的 AI 辅助使用说明补充概念图制作；方法与评估并不使用生成模型打分。
- 不改变实验、表格、参考文献、结论或关键限制，不修改模板、字体大小或页边距。

## 图像来源与复用

使用内置 image_gen 编辑作者已确认的概念预览，保存为上述项目内 PNG。不是矢量图，也不是新的实验输出。最终编辑要求修正数据箭头及目标标签措辞，保留公式和数值。图中单个编码器框代表对两组数据分别执行同一编码器，而非拼接数据。

最终编辑提示词：

> Precisely edit the supplied scientific diagram for use in a paper. Preserve all three panels, positions, all typography and mathematical formulas, all numeric values (0.133, 0.069, 22), all colors and all lower arrows. Change ONLY the connection between the two dataset cylinders and the shared encoder rectangle in the top of panel (b): REMOVE the horizontal black line that merges the datasets and REMOVE its central downward arrow. Instead draw TWO independent vertical downward arrows: one directly from the blue target cylinder into the upper-left area of the SAME 'Candidate encoder m' rectangle, and one directly from the orange reference cylinder into the upper-right area of that same rectangle. They must NOT join outside the encoder. Preserve the single wide shared encoder rectangle. This represents separately applying the same encoder to two datasets, NOT concatenating them. Also change the final italic sentence of panel (c) from 'Labels used only for downstream evaluation.' to 'Target labels used only for evaluation.' to clarify label-free selection does not preclude supervised pretraining. Everything else unchanged. White background, crisp black text, print-quality resolution, no additions.

## 旧版本

本轮修改前的论文和验收文件已保存至本地忽略目录：

`archive/local/backups/method_overview_20261107/before_overview.tar.gz`

本次未提交或推送。

## 验收

- 数值检查：240 项通过；22 单元的平均值、配对差和聚类统计保持不变。
- 数据集/checklist 检查：29 个任务、30 个必需来源引用、16 个中英文一致的回答通过。
- PDF 编译：英文正文 10 页、完整 PDF 26 页；中文正文 9 页、完整 PDF 19 页。包含致谢与披露、参考文献、附录和末尾 checklist。
- 总览图在两版第 2 页；原图全部保留并顺延。图注通过交叉引用连接现有 Table 9 / 表 9。
- PDF 结构检查通过，记录在 `results/camera_ready_p0_20261104/pdf_acceptance.json`。全篇渲染总览、中英文图 1 所在页及结论页已检查，未观察到裁切、重叠或缺字；编译无 overfull、缺字或未定义引用。存在常见的 underfull 和浮动位置提示，不影响上述验收。
- PNG 内的 SSR‡ 公式、减号、分母、下标、双输入箭头和两项均值已视觉核验；这部分不由 PDF 文本提取校验覆盖。
