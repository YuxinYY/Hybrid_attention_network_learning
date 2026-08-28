# 推荐工作流

## 数据位置

- 原始/过滤后的 ndjson 数据：Google Cloud Storage bucket `han_project_data`
- Snowflake stage：`HAN_DATABASE.PUBLIC.HAN_PROJECT_DATA`
- 本地代码仓：`/Users/yuxin/Documents/Local_InterviewProjects/Hybrid_attention_network_learning`
- 大数据（候选 CSV、股票面板、embedding 输出、缓存）默认在 `/Volumes/T9/.../data_processing`，若未挂载可临时改为 `./data_processing`

## 开发工作流

1. **本地开发** — 在本地 IDE 编辑代码、测试、commit
2. **push 到 GitHub** — `git push origin main`
3. **Snowflake Workspace 拉取最新代码** — 在 Workspace 中 pull/refresh
4. **在 Snowflake 上运行训练** — 训练代码从 stage 读取数据

## 原则

- Git 是唯一权威代码版本，不在 Snowflake Workspace 里直接改代码
- 数据路径通过环境变量/参数传入，不硬编码本地路径
- 大文件（ndjson 数据）只存在于 GCS / Snowflake stage，不提交到 Git

---

## 当前主流程：5 日异质波动率（IVOL）行业预测

这是目前效果最好的任务，测试结果显著高于基线。

### 目标

预测行业（GICS sector）在未来 5 个交易日内的**异质波动率（idiosyncratic volatility, IVOL）**高低：
- 用 CAPM 市场模型残差剥离市场波动；
- 未来 5 日残差的标准差作为 IVOL；
- 按全局中位数二分类：高于中位数 = 高波动（1），否则 = 低波动（0）。

### 代码入口与文件顺序

1. **候选预筛选**（已生成好，通常不重新跑）
   - `data_processing/scripts/filter_candidate_posts.py`
   - 输入：`wallstreetbets_submissions_{2021,2022,2023}.ndjson` + `comments_{2021,2022,2023}.ndjson`
   - 输出：`submissions_2021_2023_candidates.csv`、`comments_2021_2023_candidates.csv`

2. **主 pipeline**（文本匹配 + embedding + 样本构建）
   - `data_processing/scripts/pipeline.py`
   - 输出：`day_dict.pt`、`train_samples.pt`、`val_samples.pt`、`test_samples.pt`、`config.pt`
   - 注意：`pipeline.py` 当前文件被意外覆盖成了 `run.py` 副本，修复前先用 `data_processing/scripts/build_capm_ivol5_samples.py` 作为替代：
     ```bash
     python data_processing/scripts/build_capm_ivol5_samples.py \
       --stocks stocks.csv \
       --sp500 data_processing/sp500.csv \
       --val_start_date 2023-01-01 \
       --test_start_date 2023-07-01
     ```

3. **训练**
   - `run.py`
   - 默认样本目录改到 `./data_processing/samples_capm_ivol5`：
     ```bash
     env HAN_SAMPLE_DIR=./data_processing/samples_capm_ivol5 \
         HAN_DAY_DICT=./data_processing/day_dict.pt \
         HAN_USE_DAY_FEAT=0 \
         ./.venv/bin/python run.py
     ```

### 关键工具函数

- `data_processing/scripts/helpers.py`
  - `compute_idiosyncratic_vol(...)`: CAPM 滚动 beta 残差 + 未来 N 日 std
  - `compute_future_return(...)`: 真正的前向窗口（已修复 shift-rolling 错位 bug）
  - `compute_future_realized_vol(...)`: 未来 RV（同样修复前向窗口）
  - `build_time_series_samples(...)`: 从 `day_dict` 构建 HAN 时间序列样本
  - `temporal_train_val_test_split(...)`: 按日期切分 train/val/test

### 标签定义细节（CAPM IVOL）

1. 计算行业等权日收益 `sector_RET`
2. 用过去 60 个交易日的 trailing 窗口估计 CAPM beta：
   ```
   sector_RET - RF = alpha + beta * (SP500_RET - RF) + resid
   ```
   目前实现使用 CAPM（只用市场因子），未加入 SMB/HML。
3. 用 t 时刻已知的 `(alpha, beta)` 计算当日残差：
   ```
   resid_t = sector_RET_t - (alpha_t + beta_t * SP500_RET_t)
   ```
4. 未来 5 日 IVOL：
   ```
   IVOL_5(t) = std( resid_{t+1 .. t+5} )
   ```
5. 按全局中位数二分类：`ivol_5 > median → 1`

### 已修复的关键 bug

- **前向窗口 bug**: 原 `shift(-1).rolling(window)` 实际覆盖的是 `t-window+2 .. t+1`（大部分是过去），已改为 `shift(-1).rolling(window).sum().shift(-(window-1))`，确保覆盖真正的未来窗口 `t+1 .. t+window`。
- **类别权重 bug**: 原 `CrossEntropyLoss(weight=[1.0, 1.4])` 叠加在 60% 正类训练集上，导致模型塌缩到全猜一类；已改为无权重 + `WeightedRandomSampler` 平衡批次。
- **内存爆炸**: 全量 comments 时若直接 embedding 所有匹配行会爆 16GB 内存；`pipeline.py` 已改为两遍扫描 + 每 `(sector, date)` 限 top 150 帖再 embedding。

---

## 实验结果

### IVOL_5 前向 + comments（时间序列版，全局中位数）

| 集合 | 样本数 | 高波动(1) 占比 |
|------|--------|----------------|
| train | 4,675 | 55.4% |
| val   | 1,364 | 38.0% |
| test  | 1,166 | 40.2% |

**Test 结果**：

```
Average Loss: 0.6668
Overall Accuracy: 67.3242%
HighIVOL Precision: 0.5661
HighIVOL Recall:   0.8038
HighIVOL F1:       0.6643
macro avg F1:      0.6730

混淆矩阵：
         Pred_Low  Pred_High
Low:        408       289
High:        92       377
```

- 基线（全猜低波动）：`1 - 40.2% = 59.8%`
- 模型超过基线 +7.5pp（67.3% vs 59.8%）
- 这是第一次用真正前向标签跑出显著信号。

### IVOL_5 xs_median（截面中位数版 top 50%，当前最佳）

把标签改成“该行业 IVOL 是否高于**同一天**其他行业的中位数”：

```python
xs_median(t, sector) = IVOL_5(t, sector) - median(IVOL_5(t, all_sectors))
```

| 集合 | 样本数 | 高波动(1) 占比 |
|------|--------|----------------|
| train | 4,250 | 50.0% |
| val   | 1,240 | 50.0% |
| test  | 1,060 | 50.0% |

**Test 结果**：

```
Average Loss: 0.6379
Overall Accuracy: 68.3019%
HighIVOL Precision: 0.6502
HighIVOL Recall:   0.7925
HighIVOL F1:       0.7143
macro avg F1:      0.6792

混淆矩阵：
         Pred_Low  Pred_High
Low:        304       226
High:       110       420
```

- 基线（全猜低波动或全猜高波动）：50.0%
- **模型超过基线 +18.3pp（68.3% vs 50.0%）**
- 截面标签每天强制 50/50，剔除“市场整体高/低波动”的 regime 作弊路径；precision 提升到 65.0%，说明文本能区分哪些行业相对更动荡。

### IVOL_5 + day features（帖子量）

在 HAN 中额外输入 `log1p(当日帖子数)`：

```
Overall Accuracy: 61.0635%
HighIVOL Precision: 0.5102
HighIVOL Recall:   0.7974
HighIVOL F1:       0.6223
```

- 基线 59.8%，仅超过 +1.3pp
- **不加 day features 时 67.3%，加之后反而降到 61.1%**
- 原因：当前 day feature 只有帖子量，信息含量低；增加输入维度后样本量（4,675）不足以支撑额外容量，导致过拟合

### 历史实验对比

| 任务 | 标签 | Test acc | 基线 | 相对基线 | 结论 |
|------|------|----------|------|----------|------|
| RV_20 | 未来 20 日 RV | 25.1% | 71.4% | -46.3pp | 标签窗口方向错误 + 类别权重错误，模型塌缩 |
| exret_20 | 未来 20 日超额收益 | 53.4% | 54.2% | -0.8pp | 标签窗口方向错误，模型学的是过去 |
| exret_20（修正） | 真正前向超额收益 + comments | 46.9% | 60.6% | -13.7pp | 真正前向预测，无显著信号 |
| IVOL_5（时间序列） | 真正前向 5 日 IVOL + comments | 67.3% | 59.8% | +7.5pp | 首次出现显著前向信号 |
| IVOL_5 + day features | 同上 + 帖子量 | 61.1% | 59.8% | +1.3pp | day feature 稀释信号 |
| **IVOL_5 xs_median** | **截面中位数 5 日 IVOL + comments** | **68.3%** | **50.0%** | **+18.3pp** | **剔除市场 regime 后仍有稳定信号，当前最佳** |
| IVOL_5 xs_median top 20% | 截面 top 20% 5 日 IVOL + comments | 63.8% | 80.0% | -16.2pp | 稀有正类导致 precision 太低，当前样本量不足以支撑 |

**关键洞察**：
- **5 天比 20 天更可预测**：Reddit 情绪半衰期短，5 日 horizon 信噪比更高。
- **IVOL 比超额收益更可预测**：超额收益受共同因子（FOMC、宏观）主导，Reddit 文本难以预测；IVOL 剥离市场后留下的行业自身波动更容易被社交媒体讨论捕捉。
- **截面中位数比全局中位数更干净**：剔除市场整体波动 regime 后，模型精度反而提升，precision 从 56.6% → 65.0%。
- **top 20% 截面标签反而更难**：虽然与个股研究口径对齐，但正类占比降到 20% 后基线升至 80%，模型 precision 仅 33.8%，说明当前数据/模型难以在高稀有度设定下保持精确；扩大样本量后可再试。
- **简单的 day feature（帖子量）没有帮助**：需要更丰富的日级统计量（情绪、注意力、行业 relative volume）才值得重新尝试。
- **数据质量很重要**：加入 comments 后文本覆盖率从 ~83% 提升到 ~96%，模型才开始稳定学习。

### IVOL_5 xs_median top 20%（截面分位点版）

在 xs_median 的基础上把分位点从 0.5 提高到 0.8，每天只把 IVOL 最高的 **top 20% 行业** 标记为高波动：

```bash
python data_processing/scripts/build_capm_ivol5_samples.py \
  --stocks stocks.csv \
  --sp500 data_processing/sp500.csv \
  --day_dict data_processing/day_dict.pt \
  --src_config data_processing/config.pt \
  --xs_median --xs_quantile 0.8 \
  --val_start_date 2023-01-01 \
  --test_start_date 2023-07-01

env HAN_SAMPLE_DIR=./data_processing/samples_capm_ivol5_top20 \
    HAN_DAY_DICT=./data_processing/day_dict.pt \
    HAN_USE_DAY_FEAT=0 \
    HAN_RUN_TAG=ivol5_top20 \
    python run.py
```

| 集合 | 样本数 | 高波动(1) 占比 |
|------|--------|----------------|
| train | 4,250 | 20.0% |
| val   | 1,240 | 20.0% |
| test  | 1,060 | 20.0% |

**Test 结果**（early stop at epoch 3/20，best val loss）:

```
Average Loss: 0.6357
Overall Accuracy: 63.7736%
HighIVOL Precision: 0.3377
HighIVOL Recall:   0.8443
HighIVOL F1:       0.4825
macro avg F1:      0.6019

混淆矩阵：
            Pred_Low  Pred_High
Low:          497       351
High:          33       179
```

- 基线（全猜低波动）：80.0%
- 模型：63.8%，低于基线 -16.2pp
- 模型在所有正类上 recall 很高（84.4%），但 precision 只有 33.8% —— 说明文本能“预感”高波动行业，但误报太多；top 20% 的稀有正类对当前小样本 HAN 来说太难。

---

## 扩展到 2021–2025 数据后的结果

已把 2024–2025 的 submissions + comments 跑完完整 GLiNER + FinBERT pipeline，并与 2021–2023 的 `day_dict.pt` 合并：

- 合并后 `day_dict.pt`：1.8 GB，**15,206** 个 `(sector, date)` 条目（原 7,886）
- 日期覆盖：2021-01-05 ~ 2025-12-31
- 匹配到的总帖子数：~304 万行
- 已修复 `data_processing/sp500.csv` 覆盖不足问题（换用 `sp500_2020_2026.csv`）

样本按 **2024-01-01 / 2024-07-01** 切分：

| 集合 | 样本数 | 日期范围 | top 50 正例 | top 20 正例 |
|------|--------|----------|-------------|-------------|
| train | 7,843 | 2021-03-02 ~ 2023-12-29 | 45.5% | 18.2% |
| val   | 1,364 | 2024-01-02 ~ 2024-06-28 | 45.5% | 18.2% |
| test  | 4,158 | 2024-07-01 ~ 2025-12-31 | 45.5% | 18.2% |

训练使用 **day features（当日帖子量 log1p）**，结果：

### 2021–2025 xs_median top 50%

```
Average Loss: 0.7643
Overall Accuracy: 49.1101%
HighIVOL Precision: 0.4701
HighIVOL Recall:   0.9407
HighIVOL F1:       0.6269

混淆矩阵：
           Pred_Low  Pred_High
Low:        264       2004
High:       112       1778
```

- 基线（全猜高波动）：45.5%；随机：50%
- 模型准确率 49.1%，**接近随机 / 全猜正例**
- Precision 47.0% ≈ 正例占比 45.5%，说明模型基本没有区分能力

### 2021–2025 xs_median top 20%

```
Average Loss: 0.9393
Overall Accuracy: 31.2650%
HighIVOL Precision: 0.2036
HighIVOL Recall:   0.9550
HighIVOL F1:       0.3357

混淆矩阵：
           Pred_Low  Pred_High
Low:        578       2824
High:        34        722
```

- 基线（全猜高波动）：18.2%；全猜低波动：81.8%
- 模型准确率 31.3%，预测了 85% 的样本为正例，precision 仅 20.4%（≈18.2% baseline）
- 几乎退化为“全猜高波动”

### 关键结论

1. **扩展到 2024–2025 后，模型失效**。之前在 2021–2023 上表现最好的 xs_median（68.3% acc）在 2024–2025 测试期上降到 49.1%，说明：
   - 2021–2023 的结果可能没有真正泛化到未来年份；
   - 或者 Reddit 文本与行业 IVOL 的关系在 2024 年后发生了显著变化（ regime shift / 非平稳性）。
2. **day features（帖子量）没有挽救模型**：两次训练都用了 day feat，但模型仍塌缩到全猜正例。
3. **top 20% 标签更难**：稀有正类 + 跨期泛化，precision 接近随机。

## 历史里程碑（已完成的早期探索）

- **2021–2023 submissions + comments 候选生成**：submissions 75.6 万行，comments 16,027,058 行。
- **Top 300 ticker universe**：覆盖全部 11 个 GICS sector，由历史 cashtag 扫描 + GLiNER 匹配生成。
- **文本匹配**：直接 ticker 命中占绝大多数，GLiNER 只在少量公司名称命中的帖子上运行。
- **Sector 级聚合**：`day_dict` key 从 `(ticker, date)` 改为 `(sector, date)`，每天最多保留 top 50 帖。

---

## 下一步选项

1. ✅ **加入 day features（帖子量）** — 已试，无提升；2024–2025 扩展后仍无帮助。
2. ✅ **截面 IVOL 版本** — 在 2021–2023 上表现最好，但扩展到 2024–2025 后失效。
3. ✅ **扩展到 2024–2025 数据** — 已完成；结果显示模型没有跨期泛化能力。
4. **诊断 2024–2025 失效原因**
   - 检查 2021–2023 vs 2024–2025 的文本分布、标签分布、sector 覆盖是否发生显著变化。
   - 用 2021–2023 训练、2021–2023 内时间切分测试，复现 68.3% 并确认是否只是对特定测试期的过拟合。
5. **尝试不同的预测目标**
   - 既然 IVOL 跨期泛化差，可试 **未来 5 日收益方向**、**已实现波动率（RV）**、**异常交易量**。
   - 也可试 **日内波动 / 跳空** 等更短 horizon。
6. **更丰富的 day features**
   - 如果仍想拯救 IVOL 任务，需要 sentiment、keyword intensity、industry relative attention 等 engineered features，而不是单一帖子量。
7. **Attention 可解释性**
   - 抽取 temporal/news-level attention，看模型是否在学习有意义的帖子/日期，还是只学了噪声。
