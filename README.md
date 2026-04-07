# tabular-comp

テーブルデータの二値分類コンペで、AIエージェントが自律的に実験を回してスコアを上げ続けるフレームワーク。

[autoresearch](https://github.com/karpathy/autoresearch) の設計思想をテーブルデータコンペに適用したもの。エージェントがコードを変更 → 実験実行 → 結果記録 → 改善なら保持/そうでなければ破棄、を永遠にループする。

## 仕組み

リポジトリは意図的に小さく保たれており、重要なファイルは3つだけ:

- **`prepare.py`** — 固定の評価フレームワーク。データ読込、CV分割（StratifiedKFold）、ROC-AUC評価関数。**変更不可**。
- **`train.py`** — エージェントが編集する唯一のファイル。特徴量エンジニアリング、モデル定義、アンサンブル、前処理 — 全てが変更対象。**エージェントが編集**。
- **`program.md`** — エージェントへの指示書。自律実験ループの定義。**人間が編集**。

評価指標は **ROC-AUC**（高いほど良い）。CVはStratifiedKFold 10分割で固定されているため、実験間の比較が公平に行える。

## クイックスタート

### 前提条件

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（パッケージマネージャ）

### セットアップ

```bash
# 1. uv をインストール（まだなら）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 依存パッケージをインストール
cd tabular-comp
uv sync

# 3. データを配置
#    data/train.csv を置く（ターゲットカラム名はデフォルトで "target"）
#    必要なら data/test.csv も置く（提出用予測の生成時に使用）

# 4. 手動でベースラインを実行してみる
uv run train.py
```

### データの準備

`data/train.csv` に学習データを配置する。必要な形式:

| feature_1 | feature_2 | ... | target |
|-----------|-----------|-----|--------|
| 0.5       | "cat_a"   | ... | 1      |
| 1.2       | "cat_b"   | ... | 0      |

- **ターゲットカラム**: デフォルトは `target`。別名の場合は `prepare.py` の `TARGET_COL` を変更する。
- **特徴量カラム**: `target` 以外の全カラムが自動的に特徴量として使われる。
- **データ型**: 数値・カテゴリカル混在OK。前処理は `train.py` 内で行う。

## エージェントによる自律実験

Claude Code（または他のAIエージェント）をこのディレクトリで起動し、以下のように指示する:

```
program.md を読んで実験を開始してください
```

エージェントは `experiment.py` を使って以下のループを自律的に回す:

```
LOOP FOREVER:
  1. uv run python experiment.py status
  2. 実験のアイデアを考える
  3. train.py を編集
  4. git commit
  5. uv run python experiment.py run --description "実験内容"
  6. 改善 → keep（ブランチを進める）
     悪化 → discard（git reset で戻す）
  7. 次の実験へ
```

セッションが途中で切れた場合も、`uv run python experiment.py status` で現在のブランチ、ベストスコア、未記録の `run.log`、次に実行すべき再開アクションを確認できる。

寝ている間に放置すれば、朝起きたときに数十件の実験結果と改善されたモデルが手に入る。

## プロジェクト構成

```
tabular-comp/
├── .gitignore        # data/, results.tsv, run.log を除外
├── prepare.py        # 固定: データ読込、CV分割、ROC-AUC評価（変更不可）
├── train.py          # モデル・特徴量・HP（エージェントが編集）
├── experiment.py     # 実験実行・記録・再開補助CLI
├── program.md        # エージェント指示書（人間が編集）
├── pyproject.toml    # 依存パッケージ定義
├── data/             # コンペデータ置き場（git管理外）
│   ├── train.csv     # 学習データ
│   └── test.csv      # テストデータ（任意）
├── results.tsv       # 実験ログ（git管理外）
└── run.log           # 最新の実行ログ（git管理外）
```

## 利用可能なライブラリ

`pyproject.toml` で定義済み。エージェントはこれらのみ使用可能:

| ライブラリ | 用途 |
|-----------|------|
| pandas | データ操作 |
| numpy | 数値計算 |
| scikit-learn | 前処理、特徴量変換、メトリクス |
| lightgbm | 勾配ブースティング（ベースライン） |
| xgboost | 勾配ブースティング（代替） |
| catboost | 勾配ブースティング（カテゴリカル特化） |
| optuna | 別途行うハイパーパラメータ最適化 |

## エージェントが探索する領域

`program.md` で定義されている探索領域（優先度順）:

1. **特徴量エンジニアリング** — 交互作用、多項式、統計量集約、ビニング、ターゲットエンコーディング
2. **欠損値処理** — imputation戦略、欠損フラグ特徴量
3. **モデル変更** — LightGBM / XGBoost / CatBoost / スタッキング / ブレンディング
4. **カテゴリカル変数** — label / target / frequency encoding
5. **外れ値処理** — クリッピング、除外
6. **アンサンブル** — 加重平均、スタッキング
7. **特徴量選択** — importance-based, null importance, 相関フィルタ
8. **Optuna準備** — 手動の値調整ではなく、必要なら探索空間や目的関数を改善

## 出力フォーマット

`train.py` 実行後、以下の形式で結果が出力される:

```
Fold 0: AUC = 0.876543
Fold 1: AUC = 0.881234
Fold 2: AUC = 0.873456
Fold 3: AUC = 0.879012
Fold 4: AUC = 0.872345
---
val_auc:          0.876518
mean_fold_auc:    0.876518
std_fold_auc:     0.003412
elapsed_seconds:  12.3
n_features:       42
n_samples:        10000
```

## 実験ログ（results.tsv）

各実験の結果は `experiment.py` によってタブ区切りで記録される:

```
commit	val_auc	elapsed_sec	status	description
a1b2c3d	0.876543	12.3	keep	baseline LightGBM
b2c3d4e	0.882100	15.1	keep	add interaction features
c3d4e5f	0.875200	18.7	discard	switch to XGBoost (worse)
d4e5f6g	0.000000	0.0	crash	target encoding bug
```

## 途中再開

実験セッションを再開する時は、まず状態を確認する:

```bash
uv run python experiment.py status
```

既存の実験ブランチへ戻る:

```bash
uv run python experiment.py resume --branch exp/<tag>
```

`<tag>` は `uv run python experiment.py status` の `experiment_branches` と `recommended_action` を見て選ぶ。`status` はローカルの Git ブランチ一覧、`results.tsv` のベスト `keep` 行、現在の `run.log` のスコア、現在の `HEAD` が記録済みかどうかを参照して再開方法を案内する。

`run.log` に完了済みの結果が残っていて、現在の `HEAD` がまだ `results.tsv` に記録されていない場合だけ、再実行せずに記録する:

```bash
uv run python experiment.py record-last --description "completed run description"
```

通常の実験実行は、必ず実験内容をコミットしてから行う:

```bash
git add train.py
git commit -m "expN: description"
uv run python experiment.py run --description "description"
```

`experiment.py` は tracked file に未コミット差分がある場合は実行や再開を拒否する。悪化またはクラッシュした実験は `results.tsv` に記録した後、HEAD が対象コミットのままで tracked tree が clean の場合だけ `git reset --hard HEAD~1` で破棄する。

## カスタマイズ

### ターゲットカラム名の変更

`prepare.py` の `TARGET_COL` を変更:

```python
TARGET_COL = "your_target_column_name"
```

### CV分割数の変更

`prepare.py` の `N_SPLITS` を変更:

```python
N_SPLITS = 10  # 10-fold CV
```

### 評価指標の変更（別のコンペ用）

`prepare.py` の `evaluate()` 関数を変更。例えば回帰コンペなら:

```python
from sklearn.metrics import mean_squared_error

def evaluate(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)  # RMSE
```

この場合、`program.md` と `train.py` の出力フォーマットも合わせて変更する。

## 設計思想

- **単一ファイル編集**: エージェントは `train.py` だけを触る。スコープが明確で、差分がレビューしやすい。
- **固定評価**: `prepare.py` の評価関数は不変。実験間の比較が常に公平。
- **自己完結**: 外部サービスや複雑な設定は不要。1ファイル、1メトリック。
- **Git駆動**: 各実験がコミット単位。改善は保持、失敗は破棄。履歴と `results.tsv` が実験ログになる。
- **再開可能**: `experiment.py status` で中断後の状態を診断し、`resume` / `record-last` / `run` のどれで続けるべきか確認できる。

## ライセンス

MIT
