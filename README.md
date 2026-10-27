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
2. 2D ビューアでシリーズを開き、「解析」メニュー ＞ **AI** ＞ **MONAI Bundle (remote GPU)**（本体 0.4.0 では「解析」メニューの直下に出ます）。
3. 一覧からモデルと版（既定は最新）を選んで「実行」。同意画面は 1 回です。一覧は Model Zoo の全 Bundle から作ってあり、このシリーズ（モダリティ）に使えるものだけが選べます。使えないものは理由が出ます。
   「一覧を更新」は、計算機で Model Zoo の最新の一覧を取り直します（画像は送りません）。
4. 計算機の上で、モデルの説明（`configs/metadata.json`）から「このシリーズに使えるか」を判定し、使えれば推論します。合わなければ推論せずに止まり、理由が出ます。
5. 結果は ROI マネージャに臓器名つきで読み込まれます。保存するラベルを選んで「SEG で保存」。
6. 窓を閉じると、Colab のランタイムを解放するかを聞かれます。

## AI アシスタントから使う

アシスタント（システム ＞ AI アシスタント）の「プラグインのツール」で、このプラグインのツールをオンにすると、チャットやロボットモードから臓器名で分割を頼めます（「脾臓を分割して」「左右の腎臓を出して」）。

- `segment_models`: 使えるモデルと、指定できる臓器名（読み取り）
- `segment_run`: 表示中の CT を、臓器名に合う Bundle で分割して ROI マネージャに出す。体積（mL）を返す
- `segment_undo`: `segment_run` が出した結果を消す（本体 0.5 以降。古い本体では使えない旨を返す）

守られること:
- **画像を外部の計算機へ送る前に、本体が毎回同意を取ります**（画面から実行するときと同じ）。取り消すと何も送られず、何も表示されません。
- 臓器名はモデルのラベルに**送る前に**当てます。当たらない名前は、計算機へ送らずに止めます。
- 計算機の環境（Python のパッケージ）が足りないときは、アシスタントは導入せず、足りないものを返します。画面から一度開いて整えてください。
- SEG の保存はアシスタントからはしません。保存は画面の「SEG で保存」です。
- 返答には研究用の注記が付きます。診断には使わないでください。

### スキル（手順の一括実行）
`plugin.json` の `skills[]` で、上のツールをつなぐ手順を 2 件配っています。アシスタント窓の［スキル］に「プラグイン: MONAI…」として出ます（プラグインのツールをオンにすると実行できます）。

- **臓器の分割と体積**: 臓器名を尋ね → 同意のうえ外部の計算機で分割 → ROI マネージャに表示 → 返った体積をパネルに出し → 確認。
- **分割結果の取り消し**: 確認 → このプラグインが出した結果だけを消す。

手順に分岐は無いため、「残さない」ときは確認で［ここで止める］を押したあと、取り消しのスキルを実行します。`plugin:<ツール名>` は、このプラグイン自身の `aiTools` を指します（他のプラグインのツールは指せません）。

## 選べるモデル

一覧は Model Zoo（`Project-MONAI/model-zoo` の `models/model_info.json` と各 Bundle の `configs/metadata.json`）から `node scripts/gen-catalog.mjs` で作り直します（`ui.js` の `ZOO`。取得日は画面にも出ます）。
「このシリーズに使える」かは Bundle の説明（モダリティ・チャネル数・出力の種類）から判定します。
**Colab の GPU T4 で最後まで通したのは下の 8 本だけ**（2026-10-03。ジョブの完走とラベルが返るところまで。結果の正しさは別に検証中）。これ以外は一覧で「未検証」と出ます。

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

### 画像のほかに渡せる入力
- **設定値**（wholeBody_ct_segmentation の `highres`・`sw_overlap`・`sw_batch_size`）: Bundle が「画面に出してよい設定」として持つ値を、型に合わせた部品で出します。既定と違う値だけを計算機へ送ります。範囲が Bundle に書かれていない値は「範囲不明」と出します。Colab（T4）で `highres` の true／false を走らせ、Bundle 側の `pixdim` と `modelname` が切り替わることを確かめました（2026-10-21）。
- **複数チャネル**（brats_mri_segmentation の 4 本: T1c, T1, T2, FLAIR の順。renalStructures_CECT_segmentation の 3 本: artery, vein, excret）: チャネルごとにシリーズを選びます。**プラグインを開く前に、同じ患者のシリーズをタイルで開いておいてください**（開いた時点のタイルしか見えません）。未選択・重複・モダリティ違い・枚数違い・格子違いは、実行を止めて理由を出します。brats は MSD Task01 の 3 症例で完走し、Dice を報告値と比べました（`docs/monai-verification.md`）。renalStructures_CECT は実機で未確認です。
- **ラベル番号と点**（vista3d、spleen_deepedit_annotation）: 点は 1 行 1 点「x, y, z, positive|negative」（患者座標 LPS, mm）で入力欄に書きます。「点 ROI から読み込む」で、置いた点 ROI を座標に直して入力欄へ足せます（ROI のラベルが negative で始まるものは陰性、それ以外は陽性。座標は見て直せます。MSD 脾臓の 1 例に点 ROI を 3 スライス × 4 点置き、出た座標の位置の画素値が、ビューアが示す値と 12/12 一致することを確認しました（0.4.1 で、画素中心の 0.5 の取り違えを直しています）。
  - vista3d: ラベル（132 件から選ぶ）と点の両方が空なら全自動（117 クラス）。点を使うときはラベルは 1 つまで、腎臓・肺・骨（2・20・21）は点と一緒に使えません（Bundle の仕様）。点があると、点の窓の外は「未推論」として背景にし、その数を画面に出します（vista3d は推論していない所を 255 で返すため）。
  - DeepEdit: 陽性は前景（spleen）、陰性は background として渡します。点が無くても走ります。
  - 実機で、vista3d の点を使う経路は Bundle のコード 1 行（`torch.from_numpy(points)`）が今の MONAI で落ちるため、型だけを揃える手当て（`as_tensor`）を計算機の上で当てます（当てた箇所は結果に残ります）。

一覧に無い Bundle も「その他」から名前で指定できます（Model Zoo の名前、または Hugging Face の `組織/名前`）。

⚠ モデルの説明は作者の自己申告です。止めるのは明らかに合わないとき（モダリティ・チャネル数・出力の種類）だけなので、結果は必ず画像で確かめてください。
⚠ ライセンスはモデルごとに違います（学習データの条件で非商用に限られるものがあります）。結果の画面に出るライセンスを確認してください。

## 計算機の上でしていること

- 匿名化された `inputs/0.npz`（float32 `[z, y, x]`・LPS の幾何）→ NIfTI（RAS）
- Bundle を取得し（`/content/graphy-cache/bundles` に保存・同じランタイムなら再利用）、`monai.bundle.run` で推論（**新しい Python のプロセス**で走らせる）
- 結果の NIfTI を affine で入力の格子に戻し、`labels.npy` と `labels.json` を返す
- Bundle は作られた MONAI の版（多くは 1.4）の書き方のままなので、互換の手当てをする:
  古い入口（`evaluating`）・廃止された引数の除去（結果の `compat` に記録）・DiNTS 系の datalist の形・設定が使う画像の読み手の依存（itk など）
- **パッケージを自分で入れるのは Colab のときだけ**。自分で立てた Jupyter・「この PC」では環境を書き換えません。
  **最初の「実行」で、画像を送らない小さな事前チェック（コードだけのジョブ・同意 1 回）が先に走り**、torch・numpy・monai・pytorch-ignite・nibabel の
  有無を表で示し、計算機が実際に使う Python（`sys.executable`）を名指しした `pip install` のコマンド（コピーできる）を出します。
  入れたら［再確認］を押します（本体の再起動は要りません）。通ったあとは、その窓のあいだ事前チェックを省きます。Colab は何も確かめずに通ります。
  Bundle ごとの追加パッケージ（itk など）は Bundle を取ってから分かるので、本処理が止まったときに、同じ枠で名前と Python を示します（入れたらもう一度「実行」）。
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

`plugin.json` の `version` を上げて `v<version>` のタグを push すると、GitHub Actions が zip・sha256・**公式鍵の署名**を作って**下書きの** Release に添付します。下書きの zip を現行の公開版の本体で確かめてから `gh release edit v<version> --draft=false` で公開します。
署名の鍵（`secrets.MINISIGN_SECRET_KEY`、パスフレーズがあれば `MINISIGN_PASSWORD`）が無いとリリースは止まります（公式プラグインは署名なしで出しません）。

## ライセンス

MIT（このプラグインのコード）。各 MONAI Bundle のライセンスはそれぞれの Bundle に従います。
