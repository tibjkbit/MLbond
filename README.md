# 中国债券市场收益预测项目

本项目旨在通过机器学习方法预测中国债券市场的收益。包括数据预处理、模型构建、结果评估和可视化分析。

## 文件结构

- `all.ipynb`: 主文档文件，包含项目的所有代码和分析。
- `requirements.txt`: 列出了运行项目所需的Python库。
- `all.md`: 项目的Markdown版本。
- `all.pdf`: 项目的PDF版本。
- `all_files/`: 用于`all.md`文件中图片的显示。


## 最近更新

- 日期：2024年3月21日
- 维护人员：Liu Jingkang & Yu Shukun

## 安装指南

要安装所需的依赖，请确保您已安装Python，并运行以下命令：

```bash
pip install -r requirements.txt
```

## 使用说明

要运行项目，请打开 `all.ipynb` 文件并按顺序执行单元格。您可以在Jupyter Notebook或JupyterLab环境中运行它。

## 神经网络

神经网络算法的相关代码在nn.py文件中，通过调整全局变量和超参数以实现不同的网络架构。

### 函数说明

### `process_data(filename, start_date=None, end_date=None)`

读取并预处理给定的CSV数据文件。可以选择性地根据起止日期过滤数据。返回处理后的Y、X、F、M矩阵和时间索引。

### `R2OOS(y_true, y_forecast)`

计算并返回样本外R2分数，评估预测性能。

### `Model_one` 和 `Model_two` 类

`Model_one` 和 `Model_two` 是PyTorch神经网络模型，用于处理不同的输入数据。这些模型使用ReLU激活函数、批量归一化和dropout层来防止过拟合。

### `IntegratedModel` 类

`IntegratedModel` 结合了 `Model_one` 和多个 `Model_two` 实例的输出，通过一个最终的线性层产生预测。

### `hyperparameter_search(...)`

执行超参数搜索，找到最佳的dropout率和L2正则化权重。返回最佳参数组合。

### `ensure_dir(directory)`

确保指定的目录存在，如果不存在，则创建它。

### `save_model_config(model, file_path)`

将模型配置保存到文件。

### `main(...)`

主函数，负责整个数据处理、模型训练和评估流程。

## 贡献

若有任何建议或问题，请随时提交issue或pull request。
