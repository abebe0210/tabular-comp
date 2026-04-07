# tabular-comp

テーブルデータの二値分類コンペで、自律的に実験を回してスコアを上げ続ける。

## Setup

新しい実験セッションを始めるために、ユーザーと以下を進める:

1. **run tag を決める**: 日付ベースで提案（例: `apr7`）。ブランチ `exp/<tag>` が存在しないことを確認。
2. **ブランチ作成**: `git checkout -b exp/<tag>` を現在の main/master から切る。
3. **ファイルを読む**: リポジトリは小さい。以下を全て読んで文脈を把握:
   - `prepare.py` — 固定の評価関数、データ読込、CV分割。変更不可。
   - `train.py` — あなたが編集するファイル。特徴量、モデル、ハイパーパラメータ。
4. **データ確認**: `data/train.csv` が存在することを確認。なければユーザーに配置を依頼。
5. **results.tsv 初期化**: ヘッダー行だけの `results.tsv` を作成。
6. **確認して開始**: セットアップ完了を確認。

確認が取れたら、実験ループを開始。

## Experimentation

実験は `uv run train.py` で実行する。

**変更できること:**
- `train.py` のみ。特徴量エンジニアリング、モデル選択、ハイパーパラメータ、アンサンブル、前処理、後処理 — 全て自由。

**変更できないこと:**
- `prepare.py` — 読み取り専用。評価関数・CV分割・データ読込が入っている。
- パッケージの追加。`pyproject.toml` にあるものだけ使う。
- 評価方法の変更。`evaluate()` 関数が真のメトリック。

**目標: val_auc を最大化する。** 全てが自由: 特徴量、モデル、ハイパーパラメータ、アンサンブル。制約はコードがクラッシュしないことだけ。

**探索すべきアイデア（優先度順）:**
1. 特徴量エンジニアリング: 交互作用特徴量、多項式特徴量、統計量（mean/std/min/max by group）、ビニング、ターゲットエンコーディング（リーク回避）
2. 欠損値処理: 異なるimputation戦略、欠損フラグ特徴量
3. モデル変更: LightGBM → XGBoost → CatBoost → スタッキング/ブレンディング
4. ハイパーパラメータチューニング: num_leaves, learning_rate, max_depth, reg_alpha/lambda, subsample
5. カテゴリカル変数処理: label encoding, target encoding, frequency encoding
6. 外れ値処理: クリッピング、除外
7. アンサンブル: 複数モデルの加重平均、スタッキング
8. 特徴量選択: importance-based, null importance, 相関フィルタ

## Output format

スクリプト終了時に以下のフォーマットで出力される:

```
---
val_auc:          0.876543
mean_fold_auc:    0.876100
std_fold_auc:     0.003200
elapsed_seconds:  12.3
n_features:       42
n_samples:        10000
```

結果の抽出:

```
grep "^val_auc:" run.log
```

## Logging results

実験完了時に `results.tsv`（タブ区切り）に記録する。

ヘッダーと5列:

```
commit	val_auc	elapsed_sec	status	description
```

1. git commit hash（短縮7文字）
2. val_auc（例: 0.876543）— クラッシュ時は 0.000000
3. 実行時間（秒）— クラッシュ時は 0.0
4. status: `keep`, `discard`, `crash`
5. 実験の短い説明

例:

```
commit	val_auc	elapsed_sec	status	description
a1b2c3d	0.876543	12.3	keep	baseline LightGBM
b2c3d4e	0.882100	15.1	keep	add interaction features
c3d4e5f	0.875200	18.7	discard	switch to XGBoost (worse)
d4e5f6g	0.000000	0.0	crash	target encoding bug
```

## The experiment loop

実験は専用ブランチ（例: `exp/apr7`）で行う。

LOOP FOREVER:

1. 現在の git 状態を確認
2. 実験のアイデアを考え、`train.py` を編集
3. `git commit`
4. 実験実行: `uv run train.py > run.log 2>&1`
5. 結果を確認: `grep "^val_auc:" run.log`
6. grep出力が空ならクラッシュ。`tail -n 50 run.log` でエラーを確認して修正を試みる。数回試してダメなら諦める
7. `results.tsv` に記録（NOTE: results.tsv は git 管理外にする）
8. val_auc が改善（高くなった）→ ブランチを進める（keep）
9. val_auc が同等以下 → `git reset` で戻す（discard）

**タイムアウト**: 各実験は通常数分以内。10分を超えたら kill して failure 扱い。

**クラッシュ**: タイポやインポート忘れなら修正して再実行。アイデア自体が破綻していたらスキップして次へ。

**絶対に止まるな**: 実験ループ開始後、人間に「続けますか？」と聞くな。人間は寝ているかもしれない。アイデアが尽きたら、もっと考えろ — 過去のニアミスの組み合わせ、より大胆なアーキテクチャ変更、別の前処理戦略を試せ。人間が手動で止めるまでループし続けろ。
