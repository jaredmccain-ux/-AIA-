# TDMNN（DEAP）二分类 / 三分类复现实验记录

本文档记录 `interactive/TDMNN/` 目录下用于 **DEAP 情绪识别** 的二分类与三分类任务，重点说明：

- 二分类 / 三分类具体怎么做
- 标签怎么打
- 数据集怎么划分（train/test split）
- 数据具体怎么切成样本（window → sequence）
- 怎么验证、指标怎么统计与保存

## 目录与脚本职责

- **二分类训练/评估（Valence 或 Arousal）**：`run_deap_original_style.py`
- **三分类训练/评估（Valence 或 Arousal）**：`run_deap_threeclass.py`
- **数据准备（从 `.dat` 生成 `.mat` 特征 + 二分类标签）**：`prepare_deap_data.py`
- **指标汇总（三分类额外 macro P/R/F1、混淆矩阵）**：`tabulate_metrics.py`

---

## 数据来源与文件格式约定

### 1) 原始（预处理后）DEAP `.dat`

默认路径：`/root/inpainting-work/interactive/new/data_preprocessed_custom/sXX.dat`

该文件是 pickle（`pickle.load(..., encoding="latin1")`），至少包含：

- `obj["data"]`：EEG 数据（按 trial/通道/时间组织）
- `obj["labels"]`：每个 trial 的主观评分（形状 `(40,4)`）
  - `labels[:,0]`：valence
  - `labels[:,1]`：arousal

### 2) 训练用 `.mat`

输出路径：`interactive/TDMNN/deap_mat/DE_sXX.mat`

由 `prepare_deap_data.py` 生成，包含：

- `data`：DE 特征 topomap（形状 **`(4800, 4, 8, 9)`**，window 级）
- `valence_labels`：二分类 valence 标签（形状 **`(4800,)`**，window 级）
- `arousal_labels`：二分类 arousal 标签（形状 **`(4800,)`**，window 级）

---

## 样本（x）如何构造：window → sequence

### window 级别：每个 subject 4800 个 window

`prepare_deap_data.py` 按复刻逻辑对每个 subject：

- 共有 **40 trials**
- 每个 trial 切成 **120 个 window**
- 因此每个 subject 总 window 数为 \(40 \times 120 = 4800\)

`.mat` 里的 `data` 即是 window 级别的特征（4800 个）。

### sequence 级别：每 6 个 window 组成 1 个训练样本

训练脚本（**二分类与三分类一致**）会把 window 按顺序拼成长度 `t=6` 的序列样本：

- `t = 6`
- 从 `(4800, 4, 8, 9)` 转换成 **`x.shape = (800, 6, 8, 9, 4)`**
- 因为 \(4800 / 6 = 800\)，所以每个 subject 最终得到 **800 个 sequence 样本**

对应实现：

- 二分类：`run_deap_original_style.py` 的 `run_subject()`：
  - `x = data.transpose([0,2,3,1]).reshape((-1, t, 8, 9, 4))`
- 三分类：`run_deap_threeclass.py` 的 `run_subject()`：
  - `x = data.transpose([0,2,3,1]).reshape((-1, t, 8, 9, 4))`

> 模型输入形式：每个 sequence 的 6 帧会被拆成 **6 路输入**（`[x[:,0], ..., x[:,5]]`），每帧经 CNN 提取特征后由 LSTM 聚合输出分类结果。

---

## 标签（y）怎么打

本项目里 **二分类标签** 与 **三分类标签** 的来源不同：

- 二分类标签：在 `prepare_deap_data.py` 中生成并写入 `.mat`
- 三分类标签：在 `run_deap_threeclass.py` 中运行时直接从 `.dat` 读取评分生成

### 二分类（binary）：阈值 5（>5 为高，否则为低）

标签来自 `.dat` 的 trial 评分（40 个）：

- valence：`labels[:,0] > 5`
- arousal：`labels[:,1] > 5`

然后把每个 trial 的标签 **repeat 120 次**，扩展成 window 级 4800 个标签并写入 `.mat`：

- `final_valence = np.repeat(valence_labels, 120)`
- `final_arousal = np.repeat(arousal_labels, 120)`

对应实现：`prepare_deap_data.py` 的 `process_subject()`。

### 三分类（three-class）：阈值 low/high（默认 low=4, high=7）

三分类标签由评分 `v` 映射到 3 类：

- `v < low` → `0`（low）
- `low <= v < high` → `1`（mid）
- `v >= high` → `2`（high）

实现位于 `run_deap_threeclass.py`：

- `rating_to_three_class(v, low=4.0, high=7.0)`

三分类脚本支持 `--target v|a`：

- `target="v"`：读取 `labels[:,0]`（valence）
- `target="a"`：读取 `labels[:,1]`（arousal）

---

## window 标签如何对齐到 sequence 标签（非常关键）

训练使用的是 **sequence 样本**（800 个），因此最终训练/评估标签也是 sequence 级（800 个）。

### 二分类：window→sequence 的标签对齐策略

`.mat` 中二分类标签是 window 级（4800）。二分类训练脚本会：

1. 先将 window 标签 one-hot：`to_categorical(..., 2)` 得到 4800×2
2. 再把标签变成 sequence 级：**每 6 个 window 取第 1 个**作为 sequence 标签来源

即（伪代码）：

- `y_seq[j] = y_window[j * t]`，其中 `t=6`

对应实现：`run_deap_original_style.py` 的 `run_subject()`：

- `y_v2 = np.vstack([y_v[j * t] for j in range(len(y_v) // t)])`
- `y_a2 = np.vstack([y_a[j * t] for j in range(len(y_a) // t)])`

### 三分类：window→sequence 的标签对齐策略

三分类标签直接在 `build_threeclass_labels_from_dat()` 里生成 sequence 级标签：

1. `trial_scores`（40）repeat 为 window 分数（4800）
2. `seq_labels[j] = rating_to_three_class(window_scores[j * t])`
3. `to_categorical(seq_labels, 3)` 得到 800×3

> 总结：二分类与三分类对齐方式一致，都是 **“每 6 个 window 取该 sequence 起始 window 对应的评分/标签”**。

---

## 数据集怎么划分（train/test）、怎么验证

### 划分粒度：sequence 级样本

每个 subject 最终得到：

- `x`：800 个 sequence 样本
- `y`：800 个 sequence 标签

### 划分方式：Subject-dependent 的 5-fold StratifiedKFold

二分类与三分类训练脚本均采用：

- **每个 subject 单独训练与评估**（subject-dependent）
- 在该 subject 的 800 个 sequence 上做 **5 折分层交叉验证**：
  - `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`
  - 分层依据是 `y.argmax(1)`（类别 id），确保每折类别比例尽量一致

对应实现：

- 二分类：`run_deap_original_style.py` → `kfold = StratifiedKFold(...)`，并用 `kfold.split(x, y.argmax(1))`
- 三分类：`run_deap_threeclass.py` → 同样 `kfold.split(x, y.argmax(1))`

### 验证（validation）怎么做：每折 test fold 同时充当验证集与评估集

每一折训练时：

- `train_idx`：训练集
- `test_idx`：作为 `validation_data=(x_test, y_test)` 提供给 Keras
  - 用于 `EarlyStopping(monitor="val_accuracy", ...)`
  - 用于（可选的）`ModelCheckpoint(monitor="val_accuracy", save_best_only=True)`
- 训练完成后，在同一个 `test_idx` 上推理得到 accuracy（投票融合后）

因此这里的“验证”本质上就是 **5-fold 交叉验证**，并没有额外从训练集再切一个独立的 val 子集。

---

## 训练过程（每折内）与模型输出

### 模型输入：6 路输入 + CNN + LSTM + Softmax

模型有 6 个输入张量（对应一个 sequence 的 6 帧），每帧 shape 为 `(8, 9, 4)`，最终输出为：

- 二分类：`num_classes=2`
- 三分类：`num_classes=3`

每个 fold 会同时训练 3 个分支模型（`model1/2/3`），最终通过投票得到最终预测。

### EarlyStopping 与（可选）保存 best checkpoints

二分类与三分类训练脚本均使用：

- `EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True)`
- 可选保存：`--save-models --model-dir ...`
  - 每折保存三份 best 模型：`best_model1.keras / best_model2.keras / best_model3.keras`
  - 路径结构：`{model_dir}/subject_{sid}/fold_{k}/best_model*.keras`

### Label smoothing

训练时对标签做 label smoothing（默认 `factor=0.01`）：

- `smooth_labels(y_train)` 仅用于训练
- 验证（`validation_data`）使用未平滑的 `y_test`

---

## 评估：三分支投票融合 + accuracy

### 投票融合规则（p1/p2/p3 → vote）

每折在 test fold 上：

1. `p1/p2/p3 = argmax(model.predict(...))`
2. 采用一致优先的投票规则：
   - 若 `p1==p2` → 取 `p1`
   - 否则若 `p1==p3` → 取 `p1`
   - 否则若 `p2==p3` → 取 `p2`
   - 否则 → 取 `p1`
3. `acc_fold = mean(vote == true) * 100`

该投票规则在二分类与三分类脚本中一致。

### 输出与汇总（JSON）

每个 subject：

- `fold_scores`：5 折 accuracy（百分比）
- `mean/std`：该 subject 的 5 折均值与标准差

跨 subject 汇总（论文常见报法）：

- `acc_avg`：所有 subject 的 `mean` 再求均值
- `std_subject`：所有 subject 的 `mean` 的标准差（个体差异）
- `std_avg`：所有 subject 的 5-fold `std` 的均值（脚本兼容字段）

---

## 方案A/方案B：留出“真正独立测试集”的评测（推荐）

现有 `run_deap_original_style.py` / `run_deap_threeclass.py` 的 5-fold 写法，会把 fold 的 `test` 同时当作 `validation_data`（早停/选模型）和最终打分集合；如果你希望 **测试集完全不参与 early-stopping / model selection**，请使用统一脚本：

- `interactive/TDMNN/run_deap_proper_eval.py`

它实现两种协议，并且**按 trial 切分**避免泄漏（同一 trial 的相邻 window/sequence 不会被拆到不同集合）：

### 协议 A（同被试 / within-subject）：每个 subject 按 trial 切 train/val/test

- 默认 `--test-size 0.2`：40 个 trials 留 20%（约 8 个）做真正 test trials
- 默认 `--val-size 0.2`：剩余 trials 里再留 20% 做 val trials（仅用于 early stopping）
- 最终只在 test trials 上报告 accuracy

### 协议 B（跨被试 / LOSO）：留一被试做 test，被试级别留 val 被试做早停

- 每次留 1 个 subject 作为 test subject（该 subject 的 800 个 sequence 只用于最终评估）
- 其余 subject 用于训练；其中再随机留出 `--val-subjects` 个 subject 作为 val（仅用于 early stopping）



