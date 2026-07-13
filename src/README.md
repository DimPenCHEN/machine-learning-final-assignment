# src 组件说明

本目录将实验代码按工作模块拆分，便于复现实验和撰写报告。

| 文件 | 作用 |
| --- | --- |
| `config.py` | 统一管理路径、目标变量、输入长度、预测长度、题目指定聚合字段。 |
| `data_processing.py` | 数据读取、UCI 下载、缺失值识别、日级聚合、课程 `train.csv/test.csv/tes.csv` 优先读取。 |
| `features.py` | 特征标准化与滑动窗口样本构造，将时间序列转换为监督学习样本。 |
| `models.py` | 三种模型结构：LSTM、Transformer、CNN-Transformer 改进模型。 |
| `training.py` | 单轮训练、损失函数、梯度裁剪、测试集 MSE/MAE 计算。 |
| `visualization.py` | 不依赖 `matplotlib`，直接输出 Ground Truth 与预测曲线 SVG。 |
| `reporting.py` | 根据实验汇总表生成简要 Markdown 报告。 |

主入口仍为上级目录的 `run_experiments.py`。运行方式：

```powershell
conda activate jsyyx
python run_experiments.py --epochs 8
```

若当前目录存在 `train.csv` 与 `test.csv`/`tes.csv`，脚本优先使用课程给定划分；否则自动使用 UCI 公开数据并按时间顺序划分。

