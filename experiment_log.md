# Experiment Log: gci-compe-test-opu

- **run tag**: `gci-compe-test-opu`
- **branch**: `exp/gci-compe-test-opu`
- **date**: 2026-04-08
- **best HEAD**: `20c9e77` (val_auc = **0.894016**)
- **total experiments**: 33 (1 crash, 8 keep, 24 discard)

---

## Dataset

- **File**: `data/train.csv`
- **Rows**: 1568, **Columns**: 22 (21 features + `target`)
- **Target**: `target` (binary, ROC-AUC)
- **CV**: StratifiedKFold 10-fold (固定、`prepare.py` 管理)

### Columns

| Column | Type | Notes |
|--------|------|-------|
| customer_id | int | ID列。モデルには使わない |
| birth_year | int | 一部に外れ値あり（1890年代等） |
| education_level | str | Bachelor, Associate, PhD, Master, High School |
| marital_status | str | Married, Partner, Single, Other, Divorced, Widow |
| annual_income | float | NaN少数あり。99th pctile以上に外れ値 |
| num_children | int | |
| num_teenagers | int | |
| registration_date | str | YYYY-MM-DD形式 |
| days_since_last_purchase | int | |
| spend_wines/fruits/meat/fish/sweets/gold | int | 6つの支出カテゴリ |
| deals/web/catalog/store_purchases | int | 4つの購入チャネル |
| monthly_web_visits | int | |
| has_complaint | int | 0/1 |

---

## Progression of Best Scores

| exp# | val_auc | description | key insight |
|------|---------|-------------|-------------|
| 1 | 0.856298 | baseline LightGBM + label encoding | カテゴリカル列が文字列のため初回はcrash。label encoding+customer_id除外で動作 |
| 2 | 0.860619 | 基本特徴量 (age, total_spend, ratios等) | +0.004。基本的な特徴量追加は有効 |
| 4 | 0.868494 | freq encoding, log transforms, missing flag | +0.008。freq encoding・log変換が効いた |
| 6 | 0.886385 | CatBoostに切り替え | **+0.018 最大の改善**。CatBoostのネイティブカテゴリカル処理が非常に効果的 |
| 9 | 0.887337 | birth_year(1940-2005) + income(99th pctile) clip | +0.001。外れ値処理が微改善 |
| 18 | 0.890620 | CatBoost(0.7) + LightGBM(0.3) blend | +0.003。ブレンドは有効 |
| 20 | 0.890809 | blend比率 0.6/0.4 | +0.0002。微調整 |
| 24 | 0.893020 | birth_year clip緩和 (1930-2010) | +0.002。クリッピングを緩めることで情報量増加 |
| **29** | **0.894016** | **birth_year clip完全削除** | **+0.001。外れ値もモデルに任せた方が良い** |

---

## Thinking Process & Key Findings

### 1. このデータセットの特性

- **小データ（1568行）**: 特徴量追加がオーバーフィットに直結する。追加特徴量の実験は**12回中11回が悪化**。
- **CatBoostが圧倒的**: LightGBMベースライン（0.868）→ CatBoost（0.886）で+0.018。このデータのカテゴリカル変数の扱いに強い。
- **ブレンドは有効**: CatBoost単体（0.889）→ CatBoost+LGB blend（0.894）で+0.005。
- **外れ値処理は慎重に**: birth_yearはclipしない方が良い。incomeは99th pctileでclipが最適。

### 2. 試したが悪化したもの（重要：再試行不要）

#### 特徴量追加系（全て悪化）
- **交互作用特徴量** (exp3, 7, 22): income_x_spend, age_x_recency等。全て悪化。小データでは交互作用は木モデルが内部で処理する方が良い。
- **統計量特徴量** (exp5): spend_std, spend_max/min/range。悪化。
- **ビニング** (exp11): age_bin, income_bin, spend_bin。大幅悪化（0.864）。CatBoostにはビニングは不要。
- **ランク特徴量** (exp17): percentile rank。悪化。
- **日付分解** (exp10): reg_year, reg_month, reg_quarter。悪化。days_since_regで十分。
- **組み合わせカテゴリカル** (exp15): education_level x marital_status。悪化。カテゴリ組み合わせが希薄。
- **支出比率特徴量** (exp16): wine_meat_ratio, premium_everyday_ratio。悪化。
- **purchasing_power** (exp22): income_per_member x total_spend。悪化。

#### 特徴量削減系
- **比率特徴量の削除** (exp8): spending/purchase ratio列を削除 → 悪化。これらは有用。
- **freq encoding/log/missing flag削除** (exp14): 悪化。これらも有用。

#### モデル系
- **XGBoost単体** (exp12): 0.875。CatBoostに劣る。
- **3モデルブレンド** (exp19): CatBoost+LGB+XGB → XGBが足を引っ張る。
- **Stacking** (exp23): LogisticRegression meta-learner → 悪化。小データではオーバーフィット。
- **CatBoost Ordered boosting** (exp32): 悪化。

#### 外れ値・欠損系
- **NaN処理をCatBoostに任せる** (exp13): -999 sentinelの方が良い。
- **income clip削除** (exp26, 30): 悪化。incomeのclipは必要。
- **spending列のclip** (exp27): 悪化。

### 3. 現在のベスト構成（exp29: commit 20c9e77）

**特徴量**:
- customer_id 削除
- birth_year → age（クリッピングなし）
- annual_income: 99th pctile clip
- total_spend, total_purchases (合計)
- 各spend/purchaseの比率（ratio系）
- income_per_member, household_size
- spend_per_purchase
- days_since_reg (registration_dateから算出)
- education_level, marital_status: CatBoost category型 + freq encoding
- income_missing flag, log_income, log_total_spend, has_children
- NaN → -999 fill

**モデル**: CatBoost(0.6) + LightGBM(0.4) blend
- CatBoost: iterations=1000, lr=0.05, depth=6, l2_leaf_reg=3, early_stop=50
- LightGBM: n_estimators=1000, lr=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, early_stop=50

---

## Next Steps (未探索のアイデア)

### 有望度: 高
1. **fold-wiseターゲットエンコーディング**: education_level, marital_statusに対して、CV fold内でリーク防止しつつターゲットエンコード値を数値特徴量として追加。training loopの改造が必要だが、CatBoost内部のtarget encodingとは別の角度で効く可能性。
2. **CatBoostのgrow_policy変更**: `Lossguide` や `Depthwise` の切替（モデル構成の本質的変更として許容される可能性）。
3. **特徴量の非線形変換**: sqrt, 二乗変換を限定的に試す（logは既に追加済み）。

### 有望度: 中
4. **KNN-based features**: 類似顧客の平均targetをfold-wiseで計算（リーク防止が複雑）。
5. **PCA/UMAPで次元圧縮した特徴量**: spending 6列やpurchase 4列をPCA 1-2成分に集約。
6. **欠損パターン特徴量**: 複数列のNaNパターンを組み合わせた特徴量。
7. **異なるblend方式**: rank average blending（確率値の代わりにランクで統合）。

### 有望度: 低（既に失敗パターンに類似）
8. 交互作用特徴量の追加（12回中11回悪化の実績）
9. ビニング・カテゴリカル組み合わせ（悪化実績あり）
10. XGBoostベースの構成（LightGBMにも劣る）

### 禁止事項（program.md制約）
- CV数・分割方法の変更
- seed変更・multi-seed化
- ハイパーパラメータ値の手動変更
- Optuna試行
- pyproject.toml外のパッケージ追加

---

## Full Experiment Log

| # | commit | val_auc | elapsed | status | description |
|---|--------|---------|---------|--------|-------------|
| 0 | 8696d65 | 0.000000 | 0.0s | crash | baseline LightGBM (str columns) |
| 1 | e990707 | 0.856298 | 10.4s | keep | baseline LightGBM with label encoding |
| 2 | a840534 | 0.860619 | 11.6s | keep | basic feature engineering |
| 3 | f17dfc4 | 0.859533 | 14.0s | discard | interaction features, log transforms, flags |
| 4 | a674026 | 0.868494 | 12.0s | keep | freq encoding, log transforms, missing flag |
| 5 | e351dcd | 0.866487 | 12.6s | discard | spending stats, recency_x_spend, web_conversion |
| 6 | 99dcef5 | 0.886385 | 12.1s | keep | CatBoost with native categorical |
| 7 | d9005fd | 0.884095 | 12.6s | discard | interaction features on CatBoost |
| 8 | 3363326 | 0.881651 | 10.5s | discard | remove ratio features |
| 9 | bfc22ff | 0.887337 | 12.0s | keep | clip birth_year + income |
| 10 | 012f58d | 0.885190 | 13.3s | discard | reg_year, reg_month, reg_quarter |
| 11 | 3174897 | 0.864118 | 15.5s | discard | age/income/spend bins |
| 12 | 852f2fa | 0.875199 | 5.4s | discard | XGBoost |
| 13 | 0d9a63e | 0.879684 | 12.4s | discard | CatBoost native NaN |
| 14 | 5b77e31 | 0.881436 | 11.4s | discard | remove freq/log/missing features |
| 15 | f85a54a | 0.880770 | 11.9s | discard | edu x marital combo |
| 16 | 6bea3b4 | 0.878735 | 12.9s | discard | wine_meat_ratio, premium_everyday |
| 17 | 9f3500c | 0.881289 | 14.5s | discard | rank-based percentile features |
| 18 | a8de268 | 0.890620 | 26.6s | keep | CatBoost(0.7)+LGB(0.3) blend |
| 19 | e3ef1c3 | 0.889614 | 27.6s | discard | 3-model blend (Cat+LGB+XGB) |
| 20 | 0d27fde | 0.890809 | 23.0s | keep | blend ratio 0.6/0.4 |
| 21 | 409a074 | 0.890614 | 25.9s | discard | blend ratio 0.5/0.5 |
| 22 | 52d5642 | 0.883893 | 27.0s | discard | purchasing_power feature |
| 23 | 59380bd | 0.888231 | 25.9s | discard | stacking with LogReg meta |
| 24 | 5376d2c | 0.893020 | 22.6s | keep | birth_year clip 1930-2010 |
| 25 | c049f48 | 0.889659 | 22.8s | discard | income clip 99.5th pctile |
| 26 | 9b0f9ea | 0.887974 | 23.9s | discard | remove income clip |
| 27 | b3fc2b7 | 0.884361 | 24.2s | discard | clip spending at 99th |
| 28 | 06aab9f | 0.893020 | 20.2s | discard | birth_year clip 1920-2010 |
| 29 | 20c9e77 | **0.894016** | 13.5s | **keep** | **remove birth_year clip entirely** |
| 30 | 31a6904 | 0.888605 | 13.9s | discard | remove all clipping |
| 31 | 8ba1974 | 0.889038 | 7.1s | discard | CatBoost only (no blend) |
| 32 | 667c163 | 0.890755 | 31.4s | discard | CatBoost Ordered boosting |

---

## Resume Instructions

```bash
cd /Users/tmp_seitarou_abe/workspace/cc-loop-test/tabular-comp
uv run python experiment.py status
# If on main: uv run python experiment.py resume --branch exp/gci-compe-test-opu
# Then continue the experiment loop per program.md
```
