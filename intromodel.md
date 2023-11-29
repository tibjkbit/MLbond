## 弹性网

ElasticNet 是一种线性回归方法，它使用 L1 和 L2 正则化作为正则化项。这种方法的目标是最小化以下的目标函数：

$$
\frac{1}{2 \times n_{\text{samples}}} \times \|y - Xw\|_2^2 + \alpha \times l1_{\text{ratio}} \times \|w\|_1 + \frac{1}{2} \times \alpha \times (1 - l1_{\text{ratio}}) \times \|w\|_2^2
$$

其中：

- $w$ 是模型的参数向量。
- $X$ 是输入数据的特征矩阵。
- $y$ 是目标变量的向量。
- $n_{\text{samples}}$ 是样本的数量。
- $\alpha$ 是一个控制正则化强度的常数。
- $l1_{\text{ratio}}$ 是 L1 和 L2 正则化项的混合比例，范围在 0 到 1 之间。

如果你对控制 L1 和 L2 惩罚项感兴趣，需要注意的是，这等价于以下的形式：

$$
a \times \|w\|_1 + \frac{1}{2} \times b \times \|w\|_2^2
$$

其中：

$$
\alpha = a + b
$$
$$
l1_{\text{ratio}} = \frac{a}{a + b}
$$

参数 `l1_ratio` 对应于 R 包 glmnet 中的 alpha 参数，而 `alpha` 对应于 glmnet 中的 lambda 参数。特别地，当 `l1_ratio = 1` 时，它是 Lasso 惩罚。当前，`l1_ratio <= 0.01` 不可靠，除非你提供自己的 alpha 序列。





## PCR

主成分分析（PCA）是一种线性降维技术，通过奇异值分解（SVD）将数据投影到低维空间。输入数据在应用SVD之前会被居中，但不会被缩放。

在PCA中，我们首先计算数据的协方差矩阵：

$$
\mathbf{C} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}
$$

其中，$\mathbf{X}$ 是一个维度为 $n \times p$ 的数据矩阵，每行是一个样本，每列是一个特征。

然后，我们对协方差矩阵进行奇异值分解（SVD）：

$$
\mathbf{C} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

其中，$\mathbf{U}$ 和 $\mathbf{V}$ 是正交矩阵，$\mathbf{\Sigma}$ 是一个对角矩阵，其对角线上的元素是奇异值。

主成分是协方差矩阵的特征向量，可以从 $\mathbf{V}$ 中获得，它们代表数据中方差最大的方向。每个主成分都与一个奇异值相关联，这个奇异值的平方表示该主成分解释的方差。

最后，我们可以通过选择前 $k$ 个主成分来降低数据的维度，其中 $k$ 是一个小于特征数 $p$ 的整数。这样，我们可以将数据从 $n \times p$ 的矩阵降维到 $n \times k$ 的矩阵。这个降维的过程可以通过下面的公式表示：

$$
\mathbf{X}_{\text{new}} = \mathbf{X} \mathbf{V}_{k}
$$

其中，$\mathbf{V}_{k}$ 是前 $k$ 个主成分构成的矩阵。

在实际应用中，我们通常会选择保留足够多的主成分，以便捕获数据中的大部分方差。这个选择可以通过设定参数 `n_components` 来实现，它可以是一个整数，表示保留的主成分个数；也可以是一个浮点数，表示保留的方差比例；还可以是字符串 'mle'，这时会使用 Minka 的 MLE 方法来估计最佳的主成分个数。





## Lasso

Lasso回归是一种线性模型，使用L1正则化作为惩罚项。其优化目标函数为：

$$
\frac{1}{2n_{\text{samples}}} \left\| y - Xw \right\|_2^2 + \alpha \left\| w \right\|_1
$$

其中：

- $ n_{\text{samples}} $ 是样本数。
- $ y $ 是目标值。
- $ X $ 是特征矩阵。
- $ w $ 是权重向量。
- $ \alpha $ 是控制正则化强度的常数，必须是非负浮点数。
- $ \left\| \cdot \right\|_2 $ 是L2范数。
- $ \left\| \cdot \right\|_1 $ 是L1范数。

当 $ \alpha = 0 $ 时，目标函数等价于普通最小二乘法。

参数说明：

- `alpha`：正则化强度的常数，控制L1项的权重。必须是非负浮点数。
- `fit_intercept`：是否计算截距。如果设置为False，计算时不使用截距（即数据应该是中心化的）。
- `precompute`：是否使用预计算的Gram矩阵来加速计算。
- `copy_X`：如果为True，则复制X；否则，X可能会被覆盖。
- `max_iter`：最大迭代次数。
- `tol`：优化的容忍度。如果更新小于tol，优化代码会检查对偶间隙的最优性，然后继续，直到对偶间隙小于tol。
- `warm_start`：如果设置为True，使用上一次调用fit的解作为初始化；否则，就清除上一次的解。
- `positive`：如果设置为True，强制系数为正。
- `random_state`：随机数生成器的种子，用于当`selection='random'`时选择一个随机特征进行更新。
- `selection`：如果设置为'random'，每次迭代会更新一个随机系数，而不是按顺序循环遍历特征。这通常会导致更快的收敛，特别是当tol高于1e-4时。

属性说明：

- `coef_`：权重向量。
- `dual_gap_`：给定参数alpha的优化结束时的对偶间隙。
- `sparse_coef_`：权重向量的稀疏表示。
- `intercept_`：截距。
- `n_iter_`：达到指定容忍度所需的迭代次数。
- `n_features_in_`：fit时看到的特征数。
- `feature_names_in_`：fit时看到的特征名。



## OLS

线性回归（Linear Regression）旨在通过拟合线性模型来最小化观测目标和线性近似目标之间的残差平方和。目标函数为：

$$
\min_{w} \sum_{i=1}^{n} (y_i - \sum_{j=1}^{p} w_j x_{ij})^2
$$

其中，$n$ 是样本数量，$p$ 是特征数量，$y_i$ 是第 $i$ 个样本的目标值，$w_j$ 是第 $j$ 个特征的权重，$x_{ij}$ 是第 $i$ 个样本的第 $j$ 个特征值。

线性回归的参数：
- `fit_intercept`：是否计算截距。如果设置为 `False`，则不使用截距进行计算（即数据被认为是已经中心化的）。
- `copy_X`：如果设置为 `True`，则复制输入数据；否则，输入数据可能被覆盖。
- `n_jobs`：用于计算的作业数。只有在 `n_targets > 1` 且 `X` 是稀疏的或 `positive` 设置为 `True` 时，这个参数才会提供速度提升。`None` 表示 1，除非在 `joblib.parallel_backend` 上下文中。`-1` 表示使用所有处理器。
- `positive`：如果设置为 `True`，则强制系数为正。

线性回归的属性：
- `coef_`：线性回归问题的估计系数。如果在拟合时传递了多个目标（y 是 2D 的），这是一个形状为 (n_targets, n_features) 的 2D 数组；如果只传递了一个目标，这是一个长度为 n_features 的 1D 数组。
- `intercept_`：线性模型中的独立项。如果 `fit_intercept = False`，则设置为 0.0。
- `n_features_in_`：拟合时看到的特征数。
- `feature_names_in_`：拟合时看到的特征的名称。仅当 `X` 的特征名称都是字符串时定义。

线性回归的方法：
- `fit`：拟合线性模型。
- `predict`：使用线性模型进行预测。
- `score`：返回预测的决定系数 $R^2$。

决定系数 $R^2$ 定义为：

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

其中，$\hat{y}_i$ 是第 $i$ 个样本的预测值，$\bar{y}$ 是目标值的平均值。$R^2$ 的最佳可能得分是 1.0，它也可以是负的（因为模型可以是任意差的）。一个始终预测 `y` 的期望值、不考虑输入特征的常数模型会得到一个 $R^2$ 分数为 0.0。



## 提升回归树

梯度提升回归（Gradient Boosting for regression）是一种增量模型，通过逐步构建一个加法模型来优化任意可微的损失函数。在每个阶段，都会拟合一个回归树，用于逼近给定损失函数的负梯度。

以下是梯度提升回归中涉及到的公式以及超参数的中文解释：

### 公式

1. 加权不纯度减少方程：

$$
\frac{N_t}{N} \left(\text{impurity} - \frac{N_{t_R}}{N_t} \text{right impurity} - \frac{N_{t_L}}{N_t} \text{left impurity}\right)
$$

其中：
- $N$ 是总样本数。
- $N_t$ 是当前节点的样本数。
- $N_{t_L}$ 是左子节点的样本数。
- $N_{t_R}$ 是右子节点的样本数。

2. 确定系数 $R^2$：

$$
1 - \frac{u}{v}
$$

其中：
- $u$ 是残差平方和 $(y_{\text{true}} - y_{\text{pred}})^2$.
- $v$ 是总平方和 $(y_{\text{true}} - y_{\text{true}}.\text{mean}())^2$.

### 超参数

- `loss`: 损失函数，可选的值包括 {'squared_error', 'absolute_error', 'huber', 'quantile'}，默认为 'squared_error'。
- `learning_rate`: 学习率，范围为 [0.0, inf)，默认为 0.1。
- `n_estimators`: 提升阶段的数量，范围为 [1, inf)，默认为 100。
- `subsample`: 用于拟合单个基础学习器的样本的分数，范围为 (0.0, 1.0]，默认为 1.0。
- `criterion`: 分裂质量的函数，可选的值包括 {'friedman_mse', 'squared_error'}，默认为 'friedman_mse'。
- `min_samples_split`: 分裂内部节点所需的最小样本数，范围为 [2, inf) 或 (0.0, 1.0]，默认为 2。
- `min_samples_leaf`: 叶节点所需的最小样本数，范围为 [1, inf) 或 (0.0, 1.0]，默认为 1。
- `min_weight_fraction_leaf`: 叶节点所需的最小样本权重的分数，范围为 [0.0, 0.5]，默认为 0.0。
- `max_depth`: 单个回归估计器的最大深度，范围为 [1, inf) 或 None，默认为 3。
- `min_impurity_decrease`: 分裂节点所需的最小不纯度减少量，范围为 [0.0, inf)，默认为 0.0。
- `init`: 用于计算初始预测的估计器，或为 'zero'，默认为 None。
- `random_state`: 控制随机种子，可以是整数、RandomState 实例或 None，默认为 None。
- `max_features`: 搜索最佳分裂时考虑的特征数量，可以是 {'auto', 'sqrt', 'log2'}、整数或浮点数，默认为 None。
- `alpha`: huber 损失函数和分位数损失函数的 alpha-分位数，范围为 (0.0, 1.0)，默认为 0.9。
- `verbose`: 是否启用详细输出，范围为 [0, inf)，默认为 0。
- `max_leaf_nodes`: 最大叶节点数，范围为 [2, inf) 或 None，默认为 None。
- `warm_start`: 是否重用前一次调用的解决方案，增加更多的估计器，默认为 False。
- `validation_fraction`: 作为验证集的训练数据的比例，范围为 (0.0, 1.0)，默认为 0.1。
- `n_iter_no_change`: 用于决定是否使用提前停止来终止训练，默认为 None。
- `tol`: 提前停止的容忍度，范围为 [0.0, inf)，默认为 0.0001。
- `ccp_alpha`: 用于最小成本复杂性剪枝的复杂性参数，默认为 0.0。



## 随机森林

随机森林回归（Random Forest Regressor）是一种元估计器（meta estimator），它会在数据集的不同子样本上拟合多个决策树分类器，并使用平均值来提高预测的准确性并控制过拟合。如果 `bootstrap=True`（默认），则会控制子样本的大小，否则将使用整个数据集来构建每棵树。

随机森林回归器的主要参数和公式如下：

1. 超参数（Hyperparameters）:
   - `n_estimators`: 森林中树的数量，默认为100。
   - `criterion`: 分割的质量的测量函数。支持的标准有：
     - "squared_error"：均方误差，等于方差减少作为特征选择标准，并通过使用每个终端节点的均值来最小化L2损失。
     - "absolute_error"：平均绝对误差，使用每个终端节点的中位数来最小化L1损失。
     - "friedman_mse"：使用Friedman的改进分数进行均方误差的计算。
     - "poisson"：使用泊松偏差的减少来查找分割。
   - `max_depth`: 树的最大深度。如果为None，则节点将展开，直到所有叶子都是纯净的，或者所有叶子都包含少于`min_samples_split`个样本。
   - `min_samples_split`: 拆分内部节点所需的最小样本数。
   - `min_samples_leaf`: 一个叶节点所需的最小样本数。
   - `min_weight_fraction_leaf`: 叶节点所需的最小加权分数。
   - `max_features`: 寻找最佳分割时要考虑的特征数。
   - `max_leaf_nodes`: 以最佳优先方式生长的最大叶节点数。
   - `min_impurity_decrease`: 如果拆分导致不纯度的减少大于或等于这个值，则会进行拆分。
   - `bootstrap`: 是否使用自助样本来构建树。
   - `oob_score`: 是否使用袋外样本来估计泛化分数。
   - `n_jobs`: 并行运行的作业数。
   - `random_state`: 控制用于构建树的样本的随机性。
   - `verbose`: 控制拟合和预测的冗长程度。
   - `warm_start`: 是否重用前一次调用fit的解决方案，并添加更多的估计器到集合中。
   - `ccp_alpha`: 用于最小成本复杂性剪枝的复杂性参数。
   - `max_samples`: 如果bootstrap为True，从X中抽取的样本数。

2. 公式:
   - 加权不纯度减少方程为：
     $$
     \frac{N_t}{N} \left(\text{impurity} - \frac{N_{t_R}}{N_t} \times \text{right impurity} - \frac{N_{t_L}}{N_t} \times \text{left impurity}\right)
     $$
     其中，$N$ 是样本的总数，$N_t$ 是当前节点的样本数，$N_{t_L}$ 是左子节点的样本数，$N_{t_R}$ 是右子节点的样本数。如果传递了`sample_weight`，则所有的$N$，$N_t$，$N_{t_R}$和$N_{t_L}$都是加权和。





## huber

Huber损失函数在HuberRegressor中的定义是：

$$
L(y, Xw + c) = 
\begin{cases} 
\frac{1}{2}(y - Xw - c)^2 & \text{当 } \left| \frac{y - Xw - c}{\sigma} \right| \leq \epsilon \text{ 时} \\
\epsilon \left| \frac{y - Xw - c}{\sigma} \right| - \frac{1}{2}\epsilon^2 & \text{其他情况}
\end{cases}
$$

其中:
- $y$ 是目标值。
- $X$ 是特征矩阵。
- $w$ 是系数向量。
- $c$ 是截距。
- $\sigma$ 是尺度参数。
- $\epsilon$ 是切换平方损失和线性损失的阈值。

Huber Regressor优化这个带有L2正则化的损失函数，也就是说，最终最小化的损失函数是：

$$
L(y, Xw + c) + \alpha \|w\|^2
$$

其中 $\alpha$ 是L2正则化的强度。