# MONAI Bundle (remote GPU) — GRAPHY-Next 公式プラグイン

> **研究用です。診断には使わないでください。** Research use only — not a diagnostic device.

GRAPHY-Next の 2D ビューアから、[MONAI Model Zoo](https://monai.io/model-zoo.html) のセグメンテーションモデル（MONAI Bundle）を
**外部の計算機（Google Colab の GPU、または自分で立てた Jupyter Server）** で動かし、結果を **ROI マネージャ** に読み込みます。
保存するときは **DICOM SEG** にします。

- 必要なもの: GRAPHY-Next 0.4.0 以降（デスクトップ版）
- 権限: `remote-compute`（送るたびに本体が送り先・データ・実行するコードの全文を見せて同意を取ります）
- 送るのは、本体が匿名化したシリーズだけです（プラグインは画素を外へ送る口を持ちません）

## 使い方

1. 環境設定 ＞ 外部の計算機 で **Google でログイン** します。計算機が 1 つも無ければ、最初の実行で Colab の GPU T4 が自動で登録されます。
2. 2D ビューアでシリーズを開き、「解析」メニュー ＞ **MONAI Bundle (remote GPU)**。
3. 一覧からモデルを選んで「実行」。同意画面は 1 回です。
4. 計算機の上で、モデルの説明（`configs/metadata.json`）から「このシリーズに使えるか」を判定し、使えれば推論します。合わなければ推論せずに止まり、理由が出ます。
5. 結果は ROI マネージャに臓器名つきで読み込まれます。保存するラベルを選んで「SEG で保存」。
6. 窓を閉じると、Colab のランタイムを解放するかを聞かれます。

## 選べるモデル（2026-10-03 に Colab の GPU T4 で 8 本とも確認）

| モデル | 対象 | ラベル | GPU 最大 |
|---|---|---|---|
| spleen_ct_segmentation | CT | 脾臓 | 0.9 GB |
| wholeBody_ct_segmentation | CT | 全身 104 臓器 | 8.4 GB |
| swin_unetr_btcv_segmentation | CT | 腹部 13 臓器 | 5.4 GB |
| multi_organ_segmentation | CT | 腹部 7 臓器 | 5.8 GB |
| pancreas_ct_dints_segmentation | CT | 膵臓・膵腫瘍 | 3.4 GB |
| renalStructures_UNEST_segmentation | 造影 CT | 腎臓の構造 | 5.9 GB |
| prostate_mri_anatomy | T2 MR | 前立腺 | 4.0 GB |
| wholeBrainSeg_Large_UNEST_segmentation | T1 MR（MNI 空間） | 脳 133 領域 | 11.5 GB |

一覧に無い Bundle も「その他」から名前で指定できます（Model Zoo の名前、または Hugging Face の `組織/名前`）。

⚠ モデルの説明は作者の自己申告です。止めるのは明らかに合わないとき（モダリティ・チャネル数・出力の種類）だけなので、結果は必ず画像で確かめてください。
⚠ ライセンスはモデルごとに違います（学習データの条件で非商用に限られるものがあります）。結果の画面に出るライセンスを確認してください。

## 計算機の上でしていること

- 匿名化された `inputs/0.npz`（float32 `[z, y, x]`・LPS の幾何）→ NIfTI（RAS）
- Bundle を取得し（`/content/graphy-cache/bundles` に保存・同じランタイムなら再利用）、`monai.bundle.run` で推論（**新しい Python のプロセス**で走らせる）
- 結果の NIfTI を affine で入力の格子に戻し、`labels.npy` と `labels.json` を返す
- Bundle は作られた MONAI の版（多くは 1.4）の書き方のままなので、互換の手当てをする:
  古い入口（`evaluating`）・廃止された引数の除去（結果の `compat` に記録）・DiNTS 系の datalist の形・設定が使う画像の読み手の依存（itk など）
- **パッケージを自分で入れるのは Colab のときだけ**。自分で立てた Jupyter では環境を書き換えず、足りない名前を示して止まります
- PyTorch 2.6 以降の安全な読み込み（`weights_only`）を緩めるのは **MONAI 公式の Bundle だけ**です（任意のリポジトリには許しません）

## 開発

```bash
npm test
# 計算機の上のコードも（偽の MONAI で npz → NIfTI → 結果 → 格子の往復）:
GRAPHY_TEST_PYTHON=/path/to/python [GRAPHY_TEST_PYTHONPATH=<nibabel の場所>] npm test
```

実機の確認は GRAPHY-Next の `automator/src/spike/computeMonaiCheck.ts`（1 本・SEG 保存まで）と
`computeMonaiCatalogCheck.ts`（一覧の全部）。設計は GRAPHY-Next の `fw/remote-compute-design.md` §16〜§18。

## リリース

`plugin.json` の `version` を上げて `v<version>` のタグを push すると、GitHub Actions が zip・sha256・**公式鍵の署名**を作って Release に添付します。
署名の鍵（`secrets.MINISIGN_SECRET_KEY`、パスフレーズがあれば `MINISIGN_PASSWORD`）が無いとリリースは止まります（公式プラグインは署名なしで出しません）。

## ライセンス

MIT（このプラグインのコード）。各 MONAI Bundle のライセンスはそれぞれの Bundle に従います。
