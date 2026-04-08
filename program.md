# tabular-comp

テーブルデータの二値分類コンペで、自律的に実験を回してスコアを上げ続ける。

## Setup

新しい実験セッションを始めるために、ユーザーと以下を進める:

1. **run tag を決める**: 日付ベースで提案（例: `apr7`）。ブランチ `exp/<tag>` が存在しないことを確認。
2. **ブランチ作成**: `git checkout -b exp/<tag>` を現在の main/master から切る。
3. **依存パッケージインストール**: `uv sync` を実行して `.venv` と全依存パッケージを準備。
4. **ファイルを読む**: リポジトリは小さい。以下を全て読んで文脈を把握:
   - `prepare.py` — 固定の評価関数、データ読込、CV分割。変更不可。
   - `train.py` — あなたが編集するファイル。特徴量、モデル、アンサンブル、前処理。
5. **データ確認**: `data/train.csv` が存在することを確認。なければユーザーに配置を依頼。
6. **results.tsv 初期化**: ヘッダー行だけの `results.tsv` を作成（`.gitignore` 管理下のため git には入らない）。
7. **再開状態確認**: `uv run python experiment.py status` を実行し、既存の `exp/<tag>` ブランチや未記録の `run.log` がないか確認。
8. **確認して開始**: セットアップ完了を確認。

確認が取れたら、実験ループを開始。

## Experimentation

実験は `experiment.py` 経由で実行する。

```
uv run python experiment.py run --description "short experiment description"
```

補助コマンド:

```
uv run python experiment.py status
uv run python experiment.py resume --branch exp/<tag>
uv run python experiment.py record-last --description "completed run description"
```

`experiment.py` は `run.log` の生成、`val_auc` / `elapsed_seconds` の解析、`results.tsv` への追記、改善/悪化判定、悪化時の安全な `git reset --hard HEAD~1` を担当する。セッションが途中で切れた場合は、まず `status` を実行して推奨アクションに従う。
`exp/<tag>` ブランチで `run` が `keep` になった場合は、そのコミットを自動で remote に push する。push に失敗しても warning を出してループは継続する。

**変更できること:**
- `train.py` のみ。特徴量エンジニアリング、モデル選択、アンサンブル、前処理、後処理 — 全て自由。

**変更できないこと:**
- `prepare.py` — 読み取り専用。評価関数・CV分割・データ読込が入っている。
- パッケージの追加。`pyproject.toml` にあるものだけ使う。
- 評価方法の変更。`evaluate()` 関数が真のメトリック。

**非本質的試行は絶対禁止:**
- CV数・CV分割方法・`prepare.py` の `N_SPLITS` / `RANDOM_STATE` / `get_cv_splits()` は変更しない。
- seed変更は禁止。`train.py` 内の `SEED`, `SEEDS`, `random_state`, `random_seed` の値変更、seed追加、multi-seed化、seed平均・seed bagging は行わない。
- ハイパーパラメータ値の手動変更は禁止。`num_leaves`, `learning_rate`, `max_depth`, `reg_alpha`, `reg_lambda`, `subsample`, `colsample_*`, `iterations`, `depth`, `l2_leaf_reg`, `min_child_*`, `n_estimators` などの値だけを変える実験は行わない。
- Optuna試行は禁止。Optuna study/trial の追加・実行、探索空間や目的関数の追加、Optuna用ログや保存形式の実装もこの自律実験ループでは行わない。

**目標: val_auc を最大化する。** 自由に変更してよいのは特徴量、前処理、後処理、モデル構成、アンサンブルなど、問題構造に関わる本質的な変更のみ。CV・seed・ハイパラ・Optunaに関する試行は禁止。

**探索すべきアイデア（優先度順）:**
1. 特徴量エンジニアリング: 交互作用特徴量、多項式特徴量、統計量（mean/std/min/max by group）、ビニング、ターゲットエンコーディング（リーク回避）
2. 欠損値処理: 異なるimputation戦略、欠損フラグ特徴量
3. モデル変更: LightGBM → XGBoost → CatBoost → スタッキング/ブレンディング
4. カテゴリカル変数処理: label encoding, target encoding, frequency encoding
5. 外れ値処理: クリッピング、除外
6. アンサンブル: 複数モデルの加重平均、スタッキング
7. 特徴量選択: importance-based, null importance, 相関フィルタ

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

実験完了時に `experiment.py` が `results.tsv`（タブ区切り）に記録する。

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

実験は専用ブランチ（`exp/<tag>`）で行う。

LOOP FOREVER:

1. `uv run python experiment.py status` で現在の git 状態、ベストスコア、未記録ログを確認
2. 実験のアイデアを考え、`train.py` を編集
3. `git add train.py && git commit -m "expN: description"`
4. 実験実行と記録: `uv run python experiment.py run --description "description"`
5. `experiment.py` が `run.log` を解析し、`results.tsv` に記録する
6. val_auc が改善（高くなった）→ `keep` としてブランチを進め、自動で remote に push する
7. val_auc が同等以下またはクラッシュ → `discard` / `crash` として記録後、安全条件を満たす場合だけ直前の実験コミットを `git reset --hard HEAD~1` で取り消す
8. 次の実験へ

**タイムアウト**: 各実験は通常数分以内。10分を超えたら kill して failure 扱い。

**クラッシュ**: タイポやインポート忘れなら修正して再実行。アイデア自体が破綻していたらスキップして次へ。

**再開**: セッションが中断したら、同じディレクトリで `uv run python experiment.py status` を実行する。`main` に戻っていて既存の `exp/<tag>` ブランチがある場合は `uv run python experiment.py resume --branch exp/<tag>` を実行する。`run.log` に完了済み結果があり、現在の `HEAD` が `results.tsv` に未記録なら `record-last` を使う。`run.log` が空または解析不能なら `run --description "..."` で現在のコミットを再実行する。

**絶対に止まるな**: 実験ループ開始後、人間に「続けますか？」と聞くな。人間は寝ているかもしれない。アイデアが尽きたら、もっと考えろ — 過去のニアミスの組み合わせ、より大胆なアーキテクチャ変更、別の前処理戦略を試せ。人間が手動で止めるまでループし続けろ。

## Test mode

環境変数 `MAX_EXPERIMENTS` が設定されている場合、指定された回数の実験を完了した時点でループを終了する。テスト時や短時間のデバッグに使用。

```bash
MAX_EXPERIMENTS=3 # この場合、3回の実験でループ終了
```

設定されていない場合（デフォルト）は通常通り無限ループで実行される。
