def write_markdown_report(path, source_note, feature_cols, summary_rows):
    lines = [
        "# 家庭电力消耗多变量时间序列预测实验报告",
        "",
        "## 1. 问题介绍",
        "",
        "本项目使用过去 90 天多变量日级用电序列，分别预测未来 90 天与 365 天的 global_active_power。",
        "",
        f"数据来源：{source_note}。使用特征：" + "、".join(feature_cols) + "。",
        "",
        "## 2. 模型",
        "",
        "模型包括 LSTM、Transformer 以及 CNN-Transformer 改进模型。",
        "",
        "## 3. 结果与分析",
        "",
        "| 预测长度 | 模型 | MSE均值 | MSE std | MAE均值 | MAE std |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['horizon']} | {row['model']} | {row['mse_mean']:.4f} | {row['mse_std']:.4f} | {row['mae_mean']:.4f} | {row['mae_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 4. 讨论",
            "",
            "详细 LaTeX 报告请见 report.tex。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

