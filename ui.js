/// <reference path="./graphy-plugin.d.ts" />
// @ts-check
/*
 * MONAI Bundle を外部の計算機（Google Colab の GPU など）で動かす GRAPHY-Next の公式プラグイン（H59・H61・H64・H65）。
 * 設計: GRAPHY-Next の fw/remote-compute-design.md §16〜§18。
 *
 * 流れ（「実行」1 回・同意 1 回）:
 *   1. 一覧から Bundle を選ぶ（表示中のシリーズのモダリティに合わないものは選べない）。一覧に無いものは名前を入れる
 *   2. 「実行」→ 本体が匿名化した npz を送る。計算機の上で Bundle の metadata.json
 *      （network_data_format: 入力のモダリティ・チャネル数、出力のラベル名）を読み、使えるかを判定する。
 *      合わなければ推論せずに止まる（モデルの説明と止めた理由は返る）
 *   3. 使えれば monai.bundle の推論を走らせ、ラベルの volume を返す
 *   4. 下見の画像を見せ、選んだラベルを DICOM SEG（H22）で保存する
 *
 * 計算機の上のコード（PY）は、同意画面に全文が出る。Bundle の名前と上書きの設定だけが差し込まれる。
 */

// 一覧は Model Zoo（Project-MONAI/model-zoo の models/model_info.json と各 Bundle の configs/）から
// scripts/gen-catalog.mjs が生成する。手で書き換えない。プラグインの画面は ui.js しか配られないので、ここに埋め込む。
// <catalog:begin>
export const ZOO = {"fetched":"2026-10-21","source":"https://github.com/Project-MONAI/model-zoo (dev) models/model_info.json","entries":[{"name":"brain_image_synthesis_latent_diffusion_model","versions":["1.0.3","1.0.2","1.0.1"],"version":"1.0.4","ndf":{"inputs":{"image":{"type":"tabular","format":"nii","num_channels":1,"channel_def":{"0":"Gender","1":"Age","2":"Ventricular volume","3":"Brain volume"}}},"outputs":{"pred":{"type":"image","format":"image","channel_def":{"0":"T1w"}}}},"displayable":null,"loadKeys":[]},{"name":"brats_mri_axial_slices_generative_diffusion","versions":["1.1.4","1.1.3","1.1.2","1.1.1","1.1.0","1.0.10","1.0.9","1.0.8","1.0.7","1.0.6","1.0.5","1.0.4","1.0.3","1.0.2","1.0.1","1.0.0"],"version":"1.1.5","ndf":{"inputs":{"latent":{"type":"noise","format":"image","num_channels":1}},"outputs":{"pred":{"type":"feature","format":"image","channel_def":{"0":"image"}}}},"displayable":null,"loadKeys":[]},{"name":"brats_mri_generative_diffusion","versions":["1.1.4","1.1.3","1.1.2","1.1.1","1.1.0","1.0.9","1.0.8","1.0.7","1.0.6","1.0.5","1.0.4","1.0.3","1.0.2","1.0.1","1.0.0"],"version":"1.1.5","ndf":{"inputs":{"latent":{"type":"noise","format":"image","num_channels":8},"condition":{"type":"timesteps","format":"timesteps","num_channels":1}},"outputs":{"pred":{"type":"feature","format":"image","channel_def":{"0":"image"}}}},"displayable":null,"loadKeys":[]},{"name":"brats_mri_segmentation","versions":["0.5.4","0.5.3","0.5.2","0.5.1","0.5.0","0.4.9","0.4.8","0.4.7","0.4.6","0.4.5","0.4.4","0.4.3","0.4.2","0.4.1","0.4.0","0.3.9","0.3.8","0.3.7","0.3.6","0.3.5","0.3.4","0.3.3","0.3.2","0.3.1","0.3.0","0.2.1","0.2.0","0.1.1","0.1.0"],"version":"0.5.4","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","modality":"MR","num_channels":4,"channel_def":{"0":"T1c","1":"T1","2":"T2","3":"FLAIR"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"Tumor core","1":"Whole tumor","2":"Enhancing tumor"}}}},"displayable":null,"loadKeys":["image"]},{"name":"breast_density_classification","versions":["0.1.8","0.1.7","0.1.6","0.1.5","0.1.2","0.1.1","0.1.0"],"version":"0.1.9","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","modality":"Mammogram","num_channels":3,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"labels","channel_def":{"0":"A","1":"B","2":"C","3":"D"}}}},"displayable":null,"loadKeys":["image"]},{"name":"classification_template","versions":["0.0.4","0.0.3","0.0.2","0.0.1"],"version":"0.0.4","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","modality":"none","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"probabilities","format":"classes","channel_def":{"0":"background","1":"circle","2":"triangle","3":"rectangle"}}}},"displayable":null,"loadKeys":[]},{"name":"cxr_image_synthesis_latent_diffusion_model","versions":["1.0.2","1.0.1"],"version":"1.0.3","ndf":{"inputs":{"latent_representation":{"type":"image","format":"magnitude","modality":"CXR","num_channels":3},"timesteps":{"type":"vector","format":"magnitude","num_channels":1},"context":{"type":"vector","format":"magnitude","num_channels":1024}},"outputs":{"pred":{"type":"image","format":"magnitude","channel_def":{"0":"X-ray"}}}},"displayable":null,"loadKeys":[]},{"name":"endoscopic_inbody_classification","versions":["0.5.1","0.5.0","0.4.9","0.4.8","0.4.7","0.4.6","0.4.5","0.4.4","0.4.3","0.4.2","0.4.1","0.4.0","0.3.9","0.3.8","0.3.7","0.3.6","0.3.5","0.3.4","0.3.3","0.3.2","0.3.1","0.3.0","0.2.2","0.2.1","0.2.0"],"version":"0.5.1","ndf":{"inputs":{"image":{"type":"magnitude","format":"RGB","modality":"regular","num_channels":3,"channel_def":{"0":"R","1":"G","2":"B"}}},"outputs":{"pred":{"type":"probabilities","format":"classes","channel_def":{"0":"in body","1":"out body"}}}},"displayable":null,"loadKeys":["image"]},{"name":"endoscopic_tool_segmentation","versions":["0.6.2","0.6.1","0.6.0","0.5.9","0.5.8","0.5.7","0.5.6","0.5.5","0.5.4","0.5.3","0.5.2","0.5.1","0.5.0","0.4.9","0.4.8","0.4.7","0.4.6","0.4.5","0.4.4","0.4.3","0.4.2","0.4.1","0.4.0","0.3.2","0.3.1","0.3.0","0.2.1","0.2.0"],"version":"0.6.2","ndf":{"inputs":{"image":{"type":"magnitude","format":"RGB","modality":"regular","num_channels":3,"channel_def":{"0":"R","1":"G","2":"B"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"tools"}}}},"displayable":null,"loadKeys":["image"]},{"name":"lung_nodule_ct_detection","versions":["0.6.10","0.6.9","0.6.8","0.6.7","0.6.6","0.6.5","0.6.4","0.6.3","0.6.2","0.6.1","0.6.0","0.5.9","0.5.8","0.5.7","0.5.6","0.5.5","0.5.4","0.5.3","0.5.2","0.5.1","0.5.0","0.4.5","0.4.4","0.4.3","0.4.2","0.4.1","0.4.0","0.3.0","0.2.0","0.1.1","0.1.0"],"version":"0.6.11","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","modality":"CT","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"object","format":"dict"}}},"displayable":null,"loadKeys":["image"]},{"name":"maisi_ct_generative","versions":["1.0.2","1.0.1","1.0.0","0.4.6","0.4.5","0.4.4","0.4.3","0.4.2","0.4.1","0.4.0","0.3.6"],"version":"1.0.3","ndf":null,"displayable":null,"loadKeys":[]},{"name":"mednist_ddpm","versions":["1.0.3","1.0.2","1.0.1"],"version":"1.0.3","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","modality":"xray","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"magnitude","channel_def":{"0":"image"}}}},"displayable":null,"loadKeys":[]},{"name":"mednist_gan","versions":["0.4.4","0.4.3","0.4.2","0.4.1","0.4.0","0.3.0","0.2.0","0.1.0"],"version":"0.4.4","ndf":{"inputs":{"latent":{"type":"tuples","format":"latent","num_channels":0,"channel_def":{}}},"outputs":{"pred":{"type":"image","format":"magnitude","channel_def":{"0":"image"}}}},"displayable":null,"loadKeys":[]},{"name":"mednist_reg","versions":["0.0.7","0.0.6","0.0.5","0.0.4","0.0.3","0.0.2","0.0.1"],"version":"0.0.8","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","num_channels":2,"channel_def":{"0":"moving image","1":"fixed image"}}},"outputs":{"pred":{"type":"image","format":"magnitude","channel_def":{"0":"image"}}}},"displayable":null,"loadKeys":[]},{"name":"multi_organ_segmentation","versions":["0.0.6","0.0.5","0.0.4","0.0.3"],"version":"0.0.7","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"CT","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"artery","2":"portal vein","3":"liver","4":"spleen","5":"stomach","6":"gallbladder","7":"pancreas"}}}},"displayable":null,"loadKeys":[]},{"name":"pancreas_ct_dints_segmentation","versions":["0.5.2","0.5.1","0.5.0","0.4.9","0.4.8","0.4.7","0.4.6","0.4.5","0.4.4","0.4.3","0.4.2","0.4.1","0.4.0","0.3.9","0.3.8","0.3.7","0.3.6","0.3.5","0.3.4","0.3.3","0.3.2","0.3.1","0.3.0","0.2.0","0.1.1","0.1.0"],"version":"0.5.4","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"CT","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"pancreas","2":"pancreatic tumor"}}}},"displayable":null,"loadKeys":[]},{"name":"pathology_nuclei_classification","versions":["0.2.2","0.2.1","0.2.0","0.1.9","0.1.8","0.1.7","0.1.6","0.1.5","0.1.4","0.1.3","0.1.2","0.1.1","0.1.0","0.0.9","0.0.8","0.0.7","0.0.6","0.0.5","0.0.4","0.0.3","0.0.2","0.0.1"],"version":"0.2.4","ndf":{"inputs":{"image":{"type":"magnitude","format":"RGB","modality":"regular","num_channels":4,"channel_def":{"0":"R","1":"G","2":"B","3":"Mask"}}},"outputs":{"pred":{"type":"probabilities","format":"classes","channel_def":{"0":"Other","1":"Inflammatory","2":"Epithelial","3":"Spindle-Shaped"}}}},"displayable":null,"loadKeys":["image"]},{"name":"pathology_nuclei_segmentation_classification","versions":["0.2.8","0.2.7","0.2.6","0.2.5","0.2.4","0.2.3","0.2.2","0.2.1","0.2.0","0.1.9","0.1.8","0.1.7","0.1.6","0.1.5","0.1.4","0.1.3","0.1.2","0.1.1","0.1.0"],"version":"0.2.9","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","num_channels":3,"channel_def":{"0":"image"}}},"outputs":{"nucleus_prediction":{"type":"probability","format":"segmentation","channel_def":{"0":"background","1":"nuclei"}},"horizontal_vertical":{"type":"probability","format":"regression","channel_def":{"0":"horizontal distances map","1":"vertical distances map"}},"type_prediction":{"type":"probability","format":"classification","channel_def":{"0":"background","1":"type of nucleus for each pixel"}}}},"displayable":null,"loadKeys":["image"]},{"name":"pathology_nuclick_annotation","versions":["0.2.3","0.2.2","0.2.1","0.2.0","0.1.9","0.1.8","0.1.7","0.1.6","0.1.5","0.1.4","0.1.3","0.1.2","0.1.1","0.1.0","0.0.9","0.0.8","0.0.7","0.0.6","0.0.5","0.0.4","0.0.3","0.0.2","0.0.1"],"version":"0.2.5","ndf":{"inputs":{"image":{"type":"png","format":"RGB","modality":"regular","num_channels":5,"channel_def":{"0":"R","1":"G","2":"B","3":"+ve Signal","4":"-ve Signal"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"Nuclei"}}}},"displayable":null,"loadKeys":["image"]},{"name":"pathology_tumor_detection","versions":["0.6.4","0.6.3","0.6.2","0.6.1","0.6.0","0.5.9","0.5.8","0.5.7","0.5.6","0.5.5","0.5.4","0.5.2","0.5.1","0.5.0","0.4.9","0.4.8","0.4.7","0.4.6","0.4.5","0.4.4","0.4.3","0.4.2","0.4.1","0.4.0","0.3.3","0.3.2","0.3.1","0.3.0","0.2.0","0.1.1","0.1.0"],"version":"0.6.4","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","num_channels":3,"channel_def":{"0":"R","1":"G","2":"B"}}},"outputs":{"pred":{"type":"probability","format":"classification","channel_def":{"0":"metastasis"}}}},"displayable":null,"loadKeys":[]},{"name":"pediatric_abdominal_ct_segmentation","versions":["0.4.6","0.4.5","0.4.4"],"version":"0.4.8","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"CT","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation"}}},"displayable":null,"loadKeys":[]},{"name":"prostate_mri_anatomy","versions":["0.3.6","0.3.5","0.3.4","0.3.3","0.3.2","0.3.1","0.3.0","0.2.0","0.1.1","0.1.0"],"version":"0.3.7","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","modality":"MR","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"labels","channel_def":{"0":"background","1":"central gland","2":"peripheral zone"}}}},"displayable":null,"loadKeys":["image"]},{"name":"renalStructures_CECT_segmentation","versions":["0.2.3","0.2.2","0.2.1","0.2.0","0.1.0"],"version":"0.2.4","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"CT","num_channels":3,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"artery","2":"vein","3":"ureter","4":"neoplasm","5":"parenchyma"}}}},"displayable":null,"loadKeys":["artery","vein","excret"]},{"name":"renalStructures_UNEST_segmentation","versions":["0.2.7","0.2.6","0.2.5","0.2.4","0.2.3","0.2.2","0.2.1","0.2.0","0.1.3","0.1.2","0.1.1","0.1.0"],"version":"0.2.8","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"CT","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"kidney cortex","2":"medulla","3":"pelvicalyceal system"}}}},"displayable":null,"loadKeys":["image"]},{"name":"retinalOCT_RPD_segmentation","versions":["0.0.1"],"version":"0.0.2","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","modality":"OCT","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"dictionary","format":"COCO","channel_def":{"0":"RPD"}}}},"displayable":null,"loadKeys":[]},{"name":"segmentation_template","versions":["0.0.4","0.0.3","0.0.2","0.0.1"],"version":"0.0.4","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","modality":"none","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"category 1","2":"category 2","3":"category 3"}}}},"displayable":null,"loadKeys":[]},{"name":"spleen_ct_segmentation","versions":["0.6.1","0.6.0","0.5.9","0.5.8","0.5.7","0.5.6","0.5.5","0.5.4","0.5.3","0.5.2","0.5.1","0.5.0","0.4.9","0.4.8","0.4.7","0.4.6","0.4.5","0.4.4","0.4.3","0.4.2","0.4.1","0.4.0","0.3.9","0.3.8","0.3.7","0.3.6","0.3.5","0.3.4","0.3.3","0.3.2","0.3.1","0.2.0","0.1.1","0.1.0"],"version":"0.6.1","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"CT","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"spleen"}}}},"displayable":null,"loadKeys":["image"]},{"name":"spleen_deepedit_annotation","versions":["0.5.8","0.5.7","0.5.6","0.5.5","0.5.4","0.5.3","0.5.2","0.5.1","0.5.0","0.4.9","0.4.8","0.4.7","0.4.6","0.4.5","0.4.4","0.4.3","0.4.2","0.4.1","0.4.0","0.3.9","0.3.8","0.3.7","0.3.6","0.3.5","0.3.4","0.3.3","0.3.2","0.3.1","0.3.0","0.2.0","0.1.0"],"version":"0.5.10","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"CT","num_channels":3,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"spleen"}}}},"displayable":null,"loadKeys":["image"]},{"name":"swin_unetr_btcv_segmentation","versions":["0.5.8","0.5.7","0.5.6","0.5.5","0.5.4","0.5.3","0.5.2","0.5.1","0.5.0","0.4.9","0.4.8","0.4.7","0.4.6","0.4.5","0.4.4","0.4.2","0.4.1","0.4.0","0.3.9","0.3.8","0.3.7","0.3.6","0.3.5","0.3.4","0.3.3","0.3.2","0.3.1","0.3.0","0.2.0","0.1.0"],"version":"0.6.0","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"CT","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"spleen","2":"Right Kidney","3":"Left Kideny","4":"Gallbladder","5":"Esophagus","6":"Liver","7":"Stomach","8":"Aorta","9":"IVC","10":"Portal and Splenic Veins","11":"Pancreas","12":"Right adrenal gland","13":"Left adrenal gland"}}}},"displayable":null,"loadKeys":["image"]},{"name":"valve_landmarks","versions":["0.5.2","0.5.1","0.5.0","0.4.3","0.4.2","0.4.1","0.4.0","0.3.0","0.2.0","0.1.0"],"version":"0.5.3","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","modality":"MR","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"tuples","format":"points","channel_def":{"0":"Y Dimension","1":"X Dimension"}}}},"displayable":null,"loadKeys":[]},{"name":"ventricular_short_axis_3label","versions":["0.3.5","0.3.4","0.3.3","0.3.2","0.3.1","0.3.0","0.2.0","0.1.0"],"version":"0.3.5","ndf":{"inputs":{"image":{"type":"image","format":"magnitude","modality":"MR","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"lv_pool","2":"myocardium","3":"rv_pool"}}}},"displayable":null,"loadKeys":["image"]},{"name":"vista2d","versions":["0.4.0","0.3.1","0.3.0","0.2.9","0.2.8","0.2.7","0.2.6","0.2.5","0.2.4","0.2.3","0.2.2","0.2.1"],"version":"0.4.1","ndf":{"inputs":{"image":{"type":"image","format":"RGB","num_channels":3,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation"}}},"displayable":null,"loadKeys":[]},{"name":"vista3d","versions":["0.5.11","0.5.10","0.5.8","0.5.7","0.5.6","0.5.5","0.5.4","0.5.3","0.5.2","0.5.1","0.5.0","0.4.9","0.4.8","0.4.7","0.4.6","0.4.5","0.4.4","0.4.3","0.4.2","0.4.1"],"version":"0.5.13","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"CT","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"liver","2":"kidney","3":"spleen","4":"pancreas","5":"right kidney","6":"aorta","7":"inferior vena cava","8":"right adrenal gland","9":"left adrenal gland","10":"gallbladder","11":"esophagus","12":"stomach","13":"duodenum","14":"left kidney","15":"bladder","16":"prostate or uterus","17":"portal vein and splenic vein","18":"rectum","19":"small bowel","20":"lung","21":"bone","22":"brain","23":"lung tumor","24":"pancreatic tumor","25":"hepatic vessel","26":"hepatic tumor","27":"colon cancer primaries","28":"left lung upper lobe","29":"left lung lower lobe","30":"right lung upper lobe","31":"right lung middle lobe","32":"right lung lower lobe","33":"vertebrae L5","34":"vertebrae L4","35":"vertebrae L3","36":"vertebrae L2","37":"vertebrae L1","38":"vertebrae T12","39":"vertebrae T11","40":"vertebrae T10","41":"vertebrae T9","42":"vertebrae T8","43":"vertebrae T7","44":"vertebrae T6","45":"vertebrae T5","46":"vertebrae T4","47":"vertebrae T3","48":"vertebrae T2","49":"vertebrae T1","50":"vertebrae C7","51":"vertebrae C6","52":"vertebrae C5","53":"vertebrae C4","54":"vertebrae C3","55":"vertebrae C2","56":"vertebrae C1","57":"trachea","58":"left iliac artery","59":"right iliac artery","60":"left iliac vena","61":"right iliac vena","62":"colon","63":"left rib 1","64":"left rib 2","65":"left rib 3","66":"left rib 4","67":"left rib 5","68":"left rib 6","69":"left rib 7","70":"left rib 8","71":"left rib 9","72":"left rib 10","73":"left rib 11","74":"left rib 12","75":"right rib 1","76":"right rib 2","77":"right rib 3","78":"right rib 4","79":"right rib 5","80":"right rib 6","81":"right rib 7","82":"right rib 8","83":"right rib 9","84":"right rib 10","85":"right rib 11","86":"right rib 12","87":"left humerus","88":"right humerus","89":"left scapula","90":"right scapula","91":"left clavicula","92":"right clavicula","93":"left femur","94":"right femur","95":"left hip","96":"right hip","97":"sacrum","98":"left gluteus maximus","99":"right gluteus maximus","100":"left gluteus medius","101":"right gluteus medius","102":"left gluteus minimus","103":"right gluteus minimus","104":"left autochthon","105":"right autochthon","106":"left iliopsoas","107":"right iliopsoas","108":"left atrial appendage","109":"brachiocephalic trunk","110":"left brachiocephalic vein","111":"right brachiocephalic vein","112":"left common carotid artery","113":"right common carotid artery","114":"costal cartilages","115":"heart","116":"left kidney cyst","117":"right kidney cyst","118":"prostate","119":"pulmonary vein","120":"skull","121":"spinal cord","122":"sternum","123":"left subclavian artery","124":"right subclavian artery","125":"superior vena cava","126":"thyroid gland","127":"vertebrae S1","128":"bone lesion","129":"kidney mass","130":"liver tumor","131":"vertebrae L6","132":"airway"}}}},"displayable":null,"loadKeys":["image"]},{"name":"wholeBody_ct_segmentation","versions":["0.2.7","0.2.6","0.2.5","0.2.4","0.2.3","0.2.2","0.2.1","0.2.0","0.1.9","0.1.8","0.1.7","0.1.6","0.1.5","0.1.4","0.1.3","0.1.2","0.1.1","0.1.0"],"version":"0.2.9","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"CT","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"spleen","2":"kidney_right","3":"kidney_left","4":"gallbladder","5":"liver","6":"stomach","7":"aorta","8":"inferior_vena_cava","9":"portal_vein_and_splenic_vein","10":"pancreas","11":"adrenal_gland_right","12":"adrenal_gland_left","13":"lung_upper_lobe_left","14":"lung_lower_lobe_left","15":"lung_upper_lobe_right","16":"lung_middle_lobe_right","17":"lung_lower_lobe_right","18":"vertebrae_L5","19":"vertebrae_L4","20":"vertebrae_L3","21":"vertebrae_L2","22":"vertebrae_L1","23":"vertebrae_T12","24":"vertebrae_T11","25":"vertebrae_T10","26":"vertebrae_T9","27":"vertebrae_T8","28":"vertebrae_T7","29":"vertebrae_T6","30":"vertebrae_T5","31":"vertebrae_T4","32":"vertebrae_T3","33":"vertebrae_T2","34":"vertebrae_T1","35":"vertebrae_C7","36":"vertebrae_C6","37":"vertebrae_C5","38":"vertebrae_C4","39":"vertebrae_C3","40":"vertebrae_C2","41":"vertebrae_C1","42":"esophagus","43":"trachea","44":"heart_myocardium","45":"heart_atrium_left","46":"heart_ventricle_left","47":"heart_atrium_right","48":"heart_ventricle_right","49":"pulmonary_artery","50":"brain","51":"iliac_artery_left","52":"iliac_artery_right","53":"iliac_vena_left","54":"iliac_vena_right","55":"small_bowel","56":"duodenum","57":"colon","58":"rib_left_1","59":"rib_left_2","60":"rib_left_3","61":"rib_left_4","62":"rib_left_5","63":"rib_left_6","64":"rib_left_7","65":"rib_left_8","66":"rib_left_9","67":"rib_left_10","68":"rib_left_11","69":"rib_left_12","70":"rib_right_1","71":"rib_right_2","72":"rib_right_3","73":"rib_right_4","74":"rib_right_5","75":"rib_right_6","76":"rib_right_7","77":"rib_right_8","78":"rib_right_9","79":"rib_right_10","80":"rib_right_11","81":"rib_right_12","82":"humerus_left","83":"humerus_right","84":"scapula_left","85":"scapula_right","86":"clavicula_left","87":"clavicula_right","88":"femur_left","89":"femur_right","90":"hip_left","91":"hip_right","92":"sacrum","93":"face","94":"gluteus_maximus_left","95":"gluteus_maximus_right","96":"gluteus_medius_left","97":"gluteus_medius_right","98":"gluteus_minimus_left","99":"gluteus_minimus_right","100":"autochthon_left","101":"autochthon_right","102":"iliopsoas_left","103":"iliopsoas_right","104":"urinary_bladder"}}}},"displayable":{"highres":true,"sw_overlap":0.25,"sw_batch_size":1},"loadKeys":["image"]},{"name":"wholeBrainSeg_Large_UNEST_segmentation","versions":["0.2.7","0.2.6","0.2.5","0.2.4","0.2.3","0.2.2","0.2.1","0.2.0","0.1.2","0.1.1","0.1.0"],"version":"0.2.8","ndf":{"inputs":{"image":{"type":"image","format":"hounsfield","modality":"MRI","num_channels":1,"channel_def":{"0":"image"}}},"outputs":{"pred":{"type":"image","format":"segmentation","channel_def":{"0":"background","1":"3rd-Ventricle","2":"4th-Ventricle","3":"Right-Accumbens-Area","4":"Left-Accumbens-Area","5":"Right-Amygdala","6":"Left-Amygdala","7":"Brain-Stem","8":"Right-Caudate","9":"Left-Caudate","10":"Right-Cerebellum-Exterior","11":"Left-Cerebellum-Exterior","12":"Right-Cerebellum-White-Matter","13":"Left-Cerebellum-White-Matter","14":"Right-Cerebral-White-Matter","15":"Left-Cerebral-White-Matter","16":"Right-Hippocampus","17":"Left-Hippocampus","18":"Right-Inf-Lat-Vent","19":"Left-Inf-Lat-Vent","20":"Right-Lateral-Ventricle","21":"Left-Lateral-Ventricle","22":"Right-Pallidum","23":"Left-Pallidum","24":"Right-Putamen","25":"Left-Putamen","26":"Right-Thalamus-Proper","27":"Left-Thalamus-Proper","28":"Right-Ventral-DC","29":"Left-Ventral-DC","30":"Cerebellar-Vermal-Lobules-I-V","31":"Cerebellar-Vermal-Lobules-VI-VII","32":"Cerebellar-Vermal-Lobules-VIII-X","33":"Left-Basal-Forebrain","34":"Right-Basal-Forebrain","35":"Right-ACgG--anterior-cingulate-gyrus","36":"Left-ACgG--anterior-cingulate-gyrus","37":"Right-AIns--anterior-insula","38":"Left-AIns--anterior-insula","39":"Right-AOrG--anterior-orbital-gyrus","40":"Left-AOrG--anterior-orbital-gyrus","41":"Right-AnG---angular-gyrus","42":"Left-AnG---angular-gyrus","43":"Right-Calc--calcarine-cortex","44":"Left-Calc--calcarine-cortex","45":"Right-CO----central-operculum","46":"Left-CO----central-operculum","47":"Right-Cun---cuneus","48":"Left-Cun---cuneus","49":"Right-Ent---entorhinal-area","50":"Left-Ent---entorhinal-area","51":"Right-FO----frontal-operculum","52":"Left-FO----frontal-operculum","53":"Right-FRP---frontal-pole","54":"Left-FRP---frontal-pole","55":"Right-FuG---fusiform-gyrus ","56":"Left-FuG---fusiform-gyrus","57":"Right-GRe---gyrus-rectus","58":"Left-GRe---gyrus-rectus","59":"Right-IOG---inferior-occipital-gyrus","60":"Left-IOG---inferior-occipital-gyrus","61":"Right-ITG---inferior-temporal-gyrus","62":"Left-ITG---inferior-temporal-gyrus","63":"Right-LiG---lingual-gyrus","64":"Left-LiG---lingual-gyrus","65":"Right-LOrG--lateral-orbital-gyrus","66":"Left-LOrG--lateral-orbital-gyrus","67":"Right-MCgG--middle-cingulate-gyrus","68":"Left-MCgG--middle-cingulate-gyrus","69":"Right-MFC---medial-frontal-cortex","70":"Left-MFC---medial-frontal-cortex","71":"Right-MFG---middle-frontal-gyrus","72":"Left-MFG---middle-frontal-gyrus","73":"Right-MOG---middle-occipital-gyrus","74":"Left-MOG---middle-occipital-gyrus","75":"Right-MOrG--medial-orbital-gyrus","76":"Left-MOrG--medial-orbital-gyrus","77":"Right-MPoG--postcentral-gyrus","78":"Left-MPoG--postcentral-gyrus","79":"Right-MPrG--precentral-gyrus","80":"Left-MPrG--precentral-gyrus","81":"Right-MSFG--superior-frontal-gyrus","82":"Left-MSFG--superior-frontal-gyrus","83":"Right-MTG---middle-temporal-gyrus","84":"Left-MTG---middle-temporal-gyrus","85":"Right-OCP---occipital-pole","86":"Left-OCP---occipital-pole","87":"Right-OFuG--occipital-fusiform-gyrus","88":"Left-OFuG--occipital-fusiform-gyrus","89":"Right-OpIFG-opercular-part-of-the-IFG","90":"Left-OpIFG-opercular-part-of-the-IFG","91":"Right-OrIFG-orbital-part-of-the-IFG","92":"Left-OrIFG-orbital-part-of-the-IFG","93":"Right-PCgG--posterior-cingulate-gyrus","94":"Left-PCgG--posterior-cingulate-gyrus","95":"Right-PCu---precuneus","96":"Left-PCu---precuneus","97":"Right-PHG---parahippocampal-gyrus","98":"Left-PHG---parahippocampal-gyrus","99":"Right-PIns--posterior-insula","100":"Left-PIns--posterior-insula","101":"Right-PO----parietal-operculum","102":"Left-PO----parietal-operculum","103":"Right-PoG---postcentral-gyrus","104":"Left-PoG---postcentral-gyrus","105":"Right-POrG--posterior-orbital-gyrus","106":"Left-POrG--posterior-orbital-gyrus","107":"Right-PP----planum-polare","108":"Left-PP----planum-polare","109":"Right-PrG---precentral-gyrus","110":"Left-PrG---precentral-gyrus","111":"Right-PT----planum-temporale","112":"Left-PT----planum-temporale","113":"Right-SCA---subcallosal-area","114":"Left-SCA---subcallosal-area","115":"Right-SFG---superior-frontal-gyrus","116":"Left-SFG---superior-frontal-gyrus","117":"Right-SMC---supplementary-motor-cortex","118":"Left-SMC---supplementary-motor-cortex","119":"Right-SMG---supramarginal-gyrus","120":"Left-SMG---supramarginal-gyrus","121":"Right-SOG---superior-occipital-gyrus","122":"Left-SOG---superior-occipital-gyrus","123":"Right-SPL---superior-parietal-lobule","124":"Left-SPL---superior-parietal-lobule","125":"Right-STG---superior-temporal-gyrus","126":"Left-STG---superior-temporal-gyrus","127":"Right-TMP---temporal-pole","128":"Left-TMP---temporal-pole","129":"Right-TrIFG-triangular-part-of-the-IFG","130":"Left-TrIFG-triangular-part-of-the-IFG","131":"Right-TTG---transverse-temporal-gyrus","132":"Left-TTG---transverse-temporal-gyrus"}}}},"displayable":null,"loadKeys":["image"]}]};
// <catalog:end>

/** Colab の T4 で最後まで通した Bundle（ジョブの完走とラベルが返るところまで。結果の正しさは AV-MON-001 の検証表）。 */
export const RAN = {
  date: "2026-10-03",
  names: [
    "spleen_ct_segmentation", "wholeBody_ct_segmentation", "swin_unetr_btcv_segmentation", "multi_organ_segmentation",
    "pancreas_ct_dints_segmentation", "renalStructures_UNEST_segmentation", "prostate_mri_anatomy",
    "wholeBrainSeg_Large_UNEST_segmentation",
  ],
};

/** 表示名（日本語）。一覧に無いものは Bundle 名のまま出す。 */
const JA = {
  spleen_ct_segmentation: "脾臓",
  wholeBody_ct_segmentation: "全身 104 臓器",
  swin_unetr_btcv_segmentation: "腹部 13 臓器（BTCV）",
  multi_organ_segmentation: "腹部 7 臓器",
  pancreas_ct_dints_segmentation: "膵臓・膵腫瘍",
  renalStructures_UNEST_segmentation: "腎臓の構造（造影 CT）",
  prostate_mri_anatomy: "前立腺（T2 MR）",
  wholeBrainSeg_Large_UNEST_segmentation: "脳 133 領域（T1 MR）",
};

/** 判定は通るが、いまの実行コードでは動かない見込みのもの（理由つき）。選べない。 */
export const NOT_YET = {
  spleen_deepedit_annotation: "陽性・陰性のクリック点が要るモデルです。点を渡す画面がまだありません",
  vista3d: "入力がプロンプト（ラベル番号・点）つきの辞書で、いまの実行コードは画像のパスの一覧しか渡せません",
};

/**
 * 設定の LoadImaged が読む入力のキー（label は除く）。計算機側の load_keys と同じ規則。
 * @param {any} inference
 * @returns {string[]}
 */
export function loadKeysOf(inference) {
  const walk = (node) => {
    if (Array.isArray(node)) {
      for (const v of node) { const r = walk(v); if (r.length) return r; }
    } else if (node && typeof node === "object") {
      if (node._target_ === "LoadImaged") {
        let ks = node.keys;
        if (typeof ks === "string" && ks.startsWith("@")) ks = inference[ks.slice(1)];
        ks = typeof ks === "string" ? [ks] : Array.isArray(ks) ? ks : [];
        return ks.filter((k) => typeof k === "string" && k !== "label");
      }
      for (const v of Object.values(node)) { const r = walk(v); if (r.length) return r; }
    }
    return [];
  };
  return inference && typeof inference === "object" ? walk(inference) : [];
}

/** シリーズ名にチャネル名が語として入っているか（"T1" が "T1c" に一致しないように、前後が英数字でないときだけ）。 */
export function channelNameMatches(label, name) {
  const esc = String(name).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp("(^|[^A-Za-z0-9])" + esc + "(?![A-Za-z0-9])", "i").test(String(label));
}
/**
 * 複数チャネルの入力の並び。順序が Bundle に書かれているときだけ出す（取り違えると結果が壊れる）。
 * 1 チャネル（または入力の形式が無い）なら null。順序が分からなければ order = "unknown"。
 * @returns {{ n: number, names: string[], order: "metadata" | "inference" | "unknown" } | null}
 */
export function channelPlan(entry) {
  const image = entry?.ndf?.inputs && (entry.ndf.inputs.image ?? Object.values(entry.ndf.inputs)[0]);
  const n = Number(image?.num_channels ?? 1);
  if (!(n > 1)) return null;
  const def = image.channel_def && typeof image.channel_def === "object" ? image.channel_def : null;
  const byMeta = def ? Array.from({ length: n }, (_, i) => def[String(i)]) : [];
  if (byMeta.length === n && byMeta.every(Boolean) && new Set(byMeta).size === n) return { n, names: byMeta.map(String), order: "metadata" };
  const keys = entry.loadKeys;
  if (Array.isArray(keys) && keys.length === n && new Set(keys).size === n) return { n, names: keys, order: "inference" };
  return { n, names: Array.from({ length: n }, (_, i) => `チャネル ${i}`), order: "unknown" };
}

/**
 * 出力のラベル番号が channel_def の番号と一致しない Bundle の、実際の番号→名前（設定の後処理から読み取ったもの）。
 * brats は予測の 3 チャネル（TC・WT・ET）を、後処理で BraTS の番号（1・2・4）に直して保存する。
 */
export const LABEL_NAMES = {
  brats_mri_segmentation: {
    1: "腫瘍コア（壊死・非造影。BraTS の 1）",
    2: "浮腫（全腫瘍のうちコア以外。BraTS の 2）",
    4: "造影される腫瘍（BraTS の 4）",
  },
};

/** 判定のラベル表を、番号が channel_def と違う Bundle の実際の番号に差し替える。 */
export function applyLabelNames(verdict, bundleName) {
  const m = LABEL_NAMES[bundleName];
  if (!m) return verdict;
  return { ...verdict, labels: Object.entries(m).map(([k, v]) => ({ value: Number(k), name: v })).sort((a, b) => a.value - b.value) };
}

/**
 * Model Zoo の 1 Bundle を一覧の 1 行にする。取得できなかった値は null（0 や空で埋めない）。
 * @param {string} name
 * @param {string[]} versions model_info.json にある版
 * @param {any} metadata configs/metadata.json（取得できなければ null）
 * @param {any} inference configs/inference.json（JSON でない・取得できなければ null）
 */
export function zooEntry(name, versions, metadata, inference) {
  const f = metadata && metadata.network_data_format;
  const pick = (o, keys) => Object.fromEntries(keys.filter((k) => o && o[k] !== undefined).map((k) => [k, o[k]]));
  const side = (m, keys) => (m && typeof m === "object"
    ? Object.fromEntries(Object.entries(m).map(([k, v]) => [k, pick(v, keys)])) : null);
  const dc = inference && typeof inference.displayable_configs === "object" ? inference.displayable_configs : null;
  return {
    name,
    versions: [...versions].sort((a, b) => b.localeCompare(a, undefined, { numeric: true })),
    version: metadata?.version ?? null,
    ndf: f ? { inputs: side(f.inputs, ["type", "format", "modality", "num_channels", "channel_def"]), outputs: side(f.outputs, ["type", "format", "channel_def"]) } : null,
    displayable: dc,
    loadKeys: loadKeysOf(inference),
  };
}

/** 一覧の 1 行を、表示中のシリーズに対して分類する。 */
export function classify(entry, modality) {
  const plan = channelPlan(entry);
  const v = judge({ network_data_format: entry.ndf ?? undefined }, { modality, channels: plan ? plan.n : 1 });
  const ran = RAN.names.includes(entry.name);
  if (!v.ok) return { kind: "stopped", reasons: v.reasons, verdict: v, ran };
  if (NOT_YET[entry.name]) return { kind: "stopped", reasons: [NOT_YET[entry.name]], verdict: v, ran };
  if (plan && plan.order === "unknown") {
    return { kind: "stopped", reasons: [`入力が ${plan.n} チャネルで、並びの意味（どのシリーズをどの順に重ねるか）が Bundle に書かれていません`], verdict: v, ran };
  }
  return { kind: ran ? "ran" : "untested", reasons: [], verdict: v, ran };
}

export const labelOf = (entry) => JA[entry.name] ?? entry.name;

const OTHER = "__other__";
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]*(\/[A-Za-z0-9][A-Za-z0-9_.-]*)?$/;

// ---------------------------------------------------------------------------
// 計算機の上で走る Python（下見と推論で共通）。先頭に GRAPHY（mode・bundle・overrides）が足される。
// ---------------------------------------------------------------------------
export const PY = String.raw`
import glob, json, os, re, subprocess, sys, time
from importlib import metadata as _md

B = GRAPHY['bundle']
OVR = GRAPHY.get('overrides') or {}
T0 = time.time()
STAGES = {}


def stage(name, p):
    STAGES[name] = round(time.time() - T0, 1)
    print('__progress__', p, name, flush=True)


def have(dist):
    try:
        _md.version(dist)
        return True
    except _md.PackageNotFoundError:
        return False


def importable(mod):
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


# パッケージを自分で入れるのは Colab（使い捨てのランタイム）のときだけ。
# 利用者が自分で立てた Jupyter の環境は書き換えない（足りなければ名前を示して止める）
COLAB = bool(os.environ.get('COLAB_RELEASE_TAG')) or importable('google.colab')


def pip(*pk):
    if not COLAB:
        raise RuntimeError('missing-packages: この計算機に ' + ' '.join(pk) + ' を入れてください（pip install ' + ' '.join(pk) + '）')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--disable-pip-version-check', *pk])


# Bundle は実行フォルダの外に置く（同じランタイムなら 2 回目はダウンロードしない）
CACHE = '/content/graphy-cache' if COLAB else os.path.join(os.path.expanduser('~'), '.graphy-cache')

# MONAI 公式の Bundle は、モデルの構造ファイル（DiNTS の search_code_*.pt など）に numpy の配列が入っていて、
# PyTorch 2.6 以降の既定（weights_only=True）では読めない。公式（Model Zoo・Hugging Face の MONAI 組織）に限って
# 従来の読み込みを許す。誰でも置ける任意のリポジトリには許さない（pickle でコードを実行される余地を残さない）
OFFICIAL = '/' not in B['name'] or B['name'].startswith('MONAI/')
if OFFICIAL:
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

stage('setup', 0.02)
# ignite は monai より先に入れる（monai は import の時点で ignite の有無を覚えるので、あとから入れても効かない）。
# Bundle の評価器・ハンドラ（monai.engines / monai.handlers）はほとんどが ignite を使う
missing = [d for d, m in (('pytorch-ignite', 'ignite'), ('monai', 'monai'), ('nibabel', 'nibabel')) if not importable(m)]
if missing:
    pip(*missing)
import numpy as np
import monai
from monai.bundle import ConfigParser
from monai.bundle import download as bundle_download


def fetch():
    name, repo = B['name'], None
    if '/' in name:
        repo, name = name, name.split('/')[-1]
    # 版ごとに分ける（名前だけだと、別の版が残っていたときにそれを使ってしまう）
    bdir = os.path.join(CACHE, 'bundles', B['version']) if B.get('version') else os.path.join(CACHE, 'bundles')
    root = os.path.join(bdir, name)
    meta = os.path.join(root, 'configs', 'metadata.json')
    if os.path.isfile(meta):
        return root
    os.makedirs(bdir, exist_ok=True)
    errors = []
    sources = [('huggingface_hub', repo)] if repo else [('monaihosting', None), ('huggingface_hub', 'MONAI/' + name)]
    for src, rp in sources:
        try:
            kw = dict(name=name, bundle_dir=bdir, source=src, progress=False)
            if B.get('version'):
                kw['version'] = B['version']
            if rp:
                kw['repo'] = rp
            bundle_download(**kw)
            if os.path.isfile(meta):
                return root
            errors.append(src + ': no configs/metadata.json')
        except Exception as e:
            errors.append(src + ': ' + str(e)[:300])
    raise RuntimeError('bundle-download-failed: ' + ' | '.join(errors))


def inference_config(root):
    for n in ('inference.json', 'inference.yaml', 'inference.yml'):
        p = os.path.join(root, 'configs', n)
        if os.path.isfile(p):
            return p
    raise RuntimeError('no-inference-config: configs/inference.(json|yaml) がありません')


def head(path, n=3000):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            return f.read(n)
    except OSError:
        return None


def gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return {'name': torch.cuda.get_device_name(0),
                    'peakMiB': round(torch.cuda.max_memory_allocated() / 1048576)}
    except Exception:
        pass
    return None


def norm(m):
    u = str(m or '').upper()
    return 'MR' if u == 'MRI' else u


def judge(meta, modality, channels=1):
    """このシリーズに使えるか。止めるのは明らかに合わないときだけ（プラグインの judge と同じ規則）。"""
    reasons = []
    fmt = meta.get('network_data_format') or {}
    ins, outs = fmt.get('inputs') or {}, fmt.get('outputs') or {}
    if not ins or not outs:
        reasons.append('入力・出力の形式（network_data_format）が書かれていないので、セグメンテーションのモデルと確認できません')
    if len(ins) > 1:
        reasons.append('入力が %d つ要るモデルです（%s）。1 シリーズでは動かせません' % (len(ins), '・'.join(ins)))
    image = ins.get('image') or (next(iter(ins.values())) if ins else None)
    if image:
        ch = int(image.get('num_channels') or 1)
        if ch != channels:
            reasons.append('入力のチャネル数が %d です（選ばれたシリーズは %d 本）。チャネルの数だけシリーズを選んでください' % (ch, channels))
        want = [norm(x) for x in re.split(r'[/,\s]+', str(image.get('modality') or '')) if x]
        have = norm(modality)
        if want and have not in want:
            reasons.append('このモデルは %s 用です（このシリーズは %s）' % ('・'.join(want), have or '不明'))
        # HU の規則はモダリティが書かれていないときだけ（MR なのに hounsfield と書いた Bundle がある）
        if not want and str(image.get('format') or '').lower() == 'hounsfield' and have != 'CT':
            reasons.append('入力は CT の HU 値を前提にしています')
    pred = outs.get('pred') or (next(iter(outs.values())) if outs else None)
    if pred and str(pred.get('format') or '').lower() not in ('segmentation', 'labels', 'label'):
        reasons.append('出力が「%s」です。このサンプルはセグメンテーションだけを扱います' % pred.get('format'))
    return reasons


def drop_unknown_args(config):
    """
    今の MONAI が受け付けない引数を、部品（_target_）の設定から外す。Bundle は作られた版（多くは 1.4）の書き方のままで、
    新しい版で廃止された引数（例: SwinUNETR の img_size）があると組み立てで止まる。外すのは、同名の候補の
    どれにも無い引数だけ（**kwargs を受ける部品は触らない）。外したものは結果に残す。
    """
    import inspect
    import pydoc
    from monai.bundle.config_item import ComponentLocator
    locator = ComponentLocator()
    dropped = []

    def candidates(name):
        if '.' in name:
            obj = pydoc.locate(name)
            return [obj] if obj is not None else []
        mods = locator.get_component_module_name(name) or []
        if isinstance(mods, str):
            mods = [mods]
        return [o for o in (pydoc.locate(m + '.' + name) for m in mods) if o is not None]

    def walk(node, where):
        if isinstance(node, dict):
            target = node.get('_target_')
            if isinstance(target, str) and target and target[0] not in '$@%':
                params, open_kw = set(), False
                for obj in candidates(target):
                    try:
                        sig = inspect.signature(obj)
                    except (TypeError, ValueError):
                        open_kw = True
                        continue
                    params |= set(sig.parameters)
                    open_kw = open_kw or any(q.kind == q.VAR_KEYWORD for q in sig.parameters.values())
                if params and not open_kw:
                    for k in [k for k in node if not k.startswith('_') and k not in params]:
                        node.pop(k)
                        dropped.append(where + target.split('.')[-1] + '.' + k)
            for k, v in node.items():
                walk(v, where + k + '#')
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, where + str(i) + '#')

    walk(config, '')
    return dropped


def run_id_of(config):
    """推論の入口。新しい Bundle は run、古いものは evaluating。"""
    for k in ('run', 'evaluating', 'inferring'):
        if k in config:
            return k
    raise RuntimeError('no-run-entry: 推論の入口（run / evaluating）が設定にありません')


def load_keys(config):
    """設定の LoadImaged が読む入力のキー（label は除く）。"@名前" は設定の最上位の値で引く。"""
    def keys_of(node):
        if isinstance(node, dict):
            if node.get('_target_') == 'LoadImaged':
                ks = node.get('keys')
                if isinstance(ks, str) and ks.startswith('@'):
                    ks = config.get(ks[1:])
                ks = [ks] if isinstance(ks, str) else list(ks or [])
                return [k for k in ks if isinstance(k, str) and k != 'label']
            for v in node.values():
                r = keys_of(v)
                if r:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = keys_of(v)
                if r:
                    return r
        return []
    return keys_of(config)


def datalist_multi(config, paths):
    """複数チャネル: LoadImaged のキーが 1 つなら、そのキーにパスの並びを渡す（チャネルを重ねて読む）。チャネルごとのキーなら順に割り当てる。"""
    ks = load_keys(config)
    if len(ks) == 1:
        return [{ks[0]: list(paths)}]
    if len(ks) == len(paths):
        return [dict(zip(ks, paths))]
    raise RuntimeError('channel-layout: 入力のキー %s とチャネル数 %d が対応しません' % (ks, len(paths)))


def datalist_key(config):
    """入力の一覧を受ける設定のキー。無ければ datalist（従来どおり）。"""
    for k in ('datalist', 'test_datalist'):
        if k in config:
            return k
    return 'datalist'


def datalist_for(config, image_path):
    """設定のどこかで "@datalist" がそのまま使われていれば dict の一覧、そうでなければパスの一覧。"""
    def raw_use(node):
        if isinstance(node, dict):
            return any(raw_use(v) for v in node.values())
        if isinstance(node, list):
            return any(raw_use(v) for v in node)
        return isinstance(node, str) and node.strip() == '@datalist'
    return [{'image': image_path}] if raw_use(config) else [image_path]


root = fetch()
meta_path = os.path.join(root, 'configs', 'metadata.json')
meta = json.load(open(meta_path, encoding='utf-8'))
inf_path = inference_config(root)
stage('fetched', 0.25)

# モデルの説明は、合わなくても先に書いて返す（プラグインが画面に出す）
N_CH = int(GRAPHY.get('channels') or 1)
zs = [np.load('inputs/%d.npz' % i) for i in range(N_CH)]
metas = [json.loads(bytes(zz['meta.json'])) if 'meta.json' in zz.files else {} for zz in zs]
series_meta = metas[0]
reasons = judge(meta, series_meta.get('modality'), N_CH)
for i in range(1, N_CH):
    if norm(metas[i].get('modality')) != norm(series_meta.get('modality')):
        reasons.append('チャネル %d のモダリティ（%s）がチャネル 0（%s）と違います' % (i, metas[i].get('modality') or '不明', series_meta.get('modality') or '不明'))
parser = ConfigParser()
parser.read_config(inf_path)
keys = list((parser.get() or {}).keys())
json.dump({
    'name': B['name'],
    'metadata': meta,
    'inferenceConfig': os.path.basename(inf_path),
    'hasDatalist': 'datalist' in keys,
    'license': head(os.path.join(root, 'LICENSE')),
    'dataLicense': head(os.path.join(root, 'docs', 'data_license.txt')),
    'bytes': sum(os.path.getsize(p) for p in glob.glob(os.path.join(root, '**'), recursive=True) if os.path.isfile(p)),
    'monai': monai.__version__,
    'modality': series_meta.get('modality'),
    'channels': N_CH,
    'reasons': reasons,
}, open('outputs/bundle.json', 'w', encoding='utf-8'), ensure_ascii=False)
if reasons:
    raise RuntimeError('not-applicable: ' + ' / '.join(reasons))   # 推論はしない

# Bundle が求める追加のパッケージ（torch・numpy・monai は Colab のものを使う）
for dist, ver in (meta.get('optional_packages_version') or {}).items():
    if dist.lower() in ('torch', 'torchvision', 'numpy', 'monai', 'pytorch') or have(dist):
        continue
    try:
        pip(dist + '==' + str(ver))
    except Exception:
        pip(dist)
import nibabel as nib
stage('packages', 0.35)

# npz（本体が匿名化して作ったもの）→ NIfTI。volume は [z, y, x]、origin/direction は LPS
z = zs[0]
vol, sp, org, dr = z['volume'], z['spacing'], z['origin'], z['direction']
if not (np.all(np.isfinite(sp)) and np.all(np.isfinite(org)) and np.all(np.isfinite(dr))):
    raise RuntimeError('no-geometry: このシリーズには患者座標（間隔・位置・向き）がありません')
nz, ny, nx = vol.shape
lps = np.eye(4)
lps[:3, 0] = dr[0] * sp[2]
lps[:3, 1] = dr[1] * sp[1]
lps[:3, 2] = dr[2] * sp[0]
lps[:3, 3] = org
ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps   # NIfTI は RAS
work = os.path.abspath('work')
os.makedirs(os.path.join(work, 'in'), exist_ok=True)
out_dir = os.path.join(work, 'out')
img = nib.Nifti1Image(np.ascontiguousarray(vol.transpose(2, 1, 0)), ras)
img.set_qform(ras, 1)
img.set_sform(ras, 1)
img.header.set_xyzt_units('mm')
image_path = os.path.join(work, 'in', 'image.nii.gz')
nib.save(img, image_path)
image_paths = [image_path]
for i in range(1, N_CH):
    zi = zs[i]
    same = (zi['volume'].shape == vol.shape and np.allclose(zi['spacing'], sp, atol=1e-3)
            and np.allclose(zi['origin'], org, atol=1e-2) and np.allclose(zi['direction'], dr, atol=1e-4))
    if not same:
        raise RuntimeError('grid-mismatch: チャネル %d の格子（大きさ・間隔・位置・向き）がチャネル 0 と違います。同じ格子のシリーズを選んでください' % i)
    imgi = nib.Nifti1Image(np.ascontiguousarray(zi['volume'].transpose(2, 1, 0)), ras)
    imgi.set_qform(ras, 1)
    imgi.set_sform(ras, 1)
    imgi.header.set_xyzt_units('mm')
    pi = os.path.join(work, 'in', 'image_%d.nii.gz' % i)
    nib.save(imgi, pi)
    image_paths.append(pi)
stage('prepared', 0.4)

# 設定が使う画像の読み手に要るパッケージ（metadata の optional_packages_version に書かれていないことがある）
READER_PACKAGES = {'ITKReader': ('itk', 'itk'), 'NrrdReader': ('pynrrd', 'nrrd'), 'PydicomReader': ('pydicom', 'pydicom')}
inf_text = open(inf_path, encoding='utf-8').read()
for reader, (dist, mod) in READER_PACKAGES.items():
    if reader in inf_text and not importable(mod):
        pip(dist)
# datalist の形は Bundle ごとに違う。多くは「パスの一覧」を "$[{'image': i} for i in @datalist]" で包むが、
# DiNTS 系（multi_organ・pancreas）は "@datalist" をそのまま dataset に渡すので {'image': パス} の一覧が要る
cfg0 = parser.get()
kw = dict(config_file=inf_path, meta_file=meta_path, bundle_root=root,
          dataset_dir=os.path.join(work, 'in'), output_dir=out_dir)
kw[datalist_key(cfg0)] = datalist_multi(cfg0, image_paths) if N_CH > 1 else datalist_for(cfg0, image_path)
logging_conf = os.path.join(root, 'configs', 'logging.conf')
if os.path.isfile(logging_conf):
    kw['logging_file'] = logging_conf
kw.update(OVR)
# 上書きが設定のどこに効くかを記録する（参照している設定の解決後の値）。結果の画面と検証でこの値を見る
effective = {}
effective_error = None
if OVR:
    try:
        import copy
        probe = ConfigParser(copy.deepcopy(parser.get()))
        probe.update(pairs=dict(OVR))
        for ck, cv in list((probe.get() or {}).items()):
            if ck not in OVR and any('@' + ok in json.dumps(cv, default=str) for ok in OVR):
                effective[ck] = json.loads(json.dumps(probe.get_parsed_content(ck, instantiate=False), default=str))
    except Exception as e:
        effective_error = type(e).__name__ + ': ' + str(e)[:300]
# 互換の手当て: 今の MONAI が受け付けない引数を外した設定を書いて、それで走らせる
config = parser.get()
dropped = drop_unknown_args(config)
if dropped:
    print('compat: dropped', dropped, flush=True)
    patched = os.path.join(work, 'inference.patched.json')
    json.dump(config, open(patched, 'w', encoding='utf-8'))
    kw['config_file'] = patched
kw['run_id'] = run_id_of(config)
# 推論は新しい Python のプロセスで走らせる。monai は import の時点で ignite・itk などの有無を覚えるので、
# この kernel で入れたパッケージが効くように（この kernel は Bundle の取得と判定にだけ monai を使う）
RUNNER = '''
import json, sys
kw = json.load(open(sys.argv[1], encoding='utf-8'))
try:
    import torch
    cuda = torch.cuda.is_available()
except Exception:
    cuda = False
if cuda:
    torch.cuda.reset_peak_memory_stats()
from monai.bundle import run
run(**kw)
json.dump({'peakMiB': round(torch.cuda.max_memory_allocated() / 1048576) if cuda else None}, open(sys.argv[2], 'w'))
'''
kw_path = os.path.join(work, 'run-args.json')
peak_path = os.path.join(work, 'run-peak.json')
json.dump(kw, open(kw_path, 'w', encoding='utf-8'))
proc = subprocess.run([sys.executable, '-c', RUNNER, kw_path, peak_path], capture_output=True, text=True)
if proc.stdout:
    print(proc.stdout[-4000:], flush=True)
if proc.returncode != 0:
    tail = (proc.stderr or '').strip().splitlines()[-30:]
    print('\n'.join(tail), file=sys.stderr, flush=True)
    last = next((l for l in reversed(tail) if l.strip()), 'unknown error')
    raise RuntimeError('bundle-run-failed: ' + last[:500])
run_peak = json.load(open(peak_path)) if os.path.isfile(peak_path) else {}
stage('inferred', 0.85)

outs = sorted(glob.glob(os.path.join(out_dir, '**', '*.nii*'), recursive=True))
if not outs:
    raise RuntimeError('no-output: output_dir に NIfTI が出ませんでした（datalist / output_dir の上書きに対応していない Bundle かもしれません）')
o = nib.load(outs[0])
lab = np.asanyarray(o.dataobj)
while lab.ndim > 3 and lab.shape[-1] == 1:
    lab = lab[..., 0]
if lab.ndim == 4:
    lab = np.argmax(lab, axis=-1)   # チャネルごとの確率・one-hot
if lab.ndim != 3:
    raise RuntimeError('unexpected-output-shape: ' + str(lab.shape))
lab = np.rint(lab).astype(np.int32)

# 出力の格子 → 入力の格子（同じなら写すだけ。違えば最近傍で取り直す）
m = np.linalg.inv(o.affine) @ ras
resampled = not (lab.shape == (nx, ny, nz) and np.allclose(m, np.eye(4), atol=1e-3))
if resampled:
    # 先に整数へ丸める（端の画素が計算誤差で -1e-12 になり、範囲外として落ちないように）
    res = np.zeros((nx, ny, nz), np.int32)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    for k in range(nz):
        p = m @ np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, k), np.ones(ii.size)])
        q = np.rint(p[:3]).astype(np.int64)
        inside = np.all((q >= 0) & (q < np.array(lab.shape)[:, None]), axis=0)
        v = np.zeros(ii.size, np.int32)
        v[inside] = lab[q[0, inside], q[1, inside], q[2, inside]]
        res[:, :, k] = v.reshape(nx, ny)
    lab = res
zyx = np.ascontiguousarray(lab.transpose(2, 1, 0))
dtype = np.uint8 if zyx.max() < 256 else np.uint16
np.save('outputs/labels.npy', zyx.astype(dtype))
values, counts = np.unique(zyx, return_counts=True)
stage('done', 1.0)
json.dump({
    'name': B['name'],
    'version': meta.get('version'),
    'channelDef': ((meta.get('network_data_format') or {}).get('outputs') or {}).get('pred', {}).get('channel_def'),
    'labels': {str(int(v)): int(c) for v, c in zip(values, counts)},
    'shape': [int(nz), int(ny), int(nx)],
    'geometry': {'spacing': sp.tolist(), 'origin': org.tolist(), 'direction': dr.tolist()},
    'outputFile': os.path.relpath(outs[0], work),
    'resampled': bool(resampled),
    'overrides': OVR,
    'effective': effective,
    'effectiveError': effective_error,
    'outputSpacing': [round(float(z), 4) for z in o.header.get_zooms()[:3]],
    'compat': dropped,
    'runId': kw['run_id'],
    'stages': STAGES,
    'gpu': ({**gpu(), 'peakMiB': run_peak.get('peakMiB')} if gpu() else None),
    'monai': monai.__version__,
}, open('outputs/labels.json', 'w', encoding='utf-8'), ensure_ascii=False)
`;

/**
 * Bundle が `displayable_configs` で公開した設定を、画面の部品に直す。キーは inference.json の
 * 「displayable_configs#名前」（他の設定がこの形で参照する）。範囲は metadata に無いので持たない。
 * @param {Record<string, unknown> | null | undefined} displayable
 * @returns {{ key: string, name: string, kind: "bool" | "int" | "number" | "text", def: unknown }[]}
 */
export function settingFields(displayable) {
  if (!displayable || typeof displayable !== "object") return [];
  return Object.entries(displayable).map(([name, def]) => ({
    key: `displayable_configs#${name}`,
    name,
    def,
    kind: typeof def === "boolean" ? "bool" : typeof def === "number" ? (Number.isInteger(def) ? "int" : "number") : "text",
  }));
}

/**
 * 画面の入力値から、計算機へ渡す上書きを作る。既定と同じ値は渡さない（確かめた既定の経路を変えない）。
 * 数として読めない値は理由つきで返す（0 や空で埋めない）。
 * @param {ReturnType<typeof settingFields>} fields
 * @param {Record<string, unknown>} values 名前 → 入力値（bool は真偽、他は文字列でも可）
 * @returns {{ ok: true, overrides: Record<string, unknown> } | { ok: false, error: string }}
 */
export function settingOverrides(fields, values) {
  const overrides = {};
  for (const f of fields) {
    if (!(f.name in values)) continue;
    let v = values[f.name];
    if (f.kind === "bool") {
      if (typeof v !== "boolean") return { ok: false, error: `${f.name}: 真偽で指定してください` };
    } else if (f.kind === "int" || f.kind === "number") {
      if (typeof v === "string" && v.trim() === "") return { ok: false, error: `${f.name}: 値が空です` };
      v = typeof v === "number" ? v : Number(v);
      if (!Number.isFinite(v)) return { ok: false, error: `${f.name}: 数で指定してください` };
      if (f.kind === "int" && !Number.isInteger(v)) return { ok: false, error: `${f.name}: 整数で指定してください` };
    } else {
      v = String(v);
    }
    if (v !== f.def) overrides[f.key] = v;
  }
  return { ok: true, overrides };
}

/**
 * 計算機へ送るコード。設定は JSON 文字列として埋め込む（JSON の文字列リテラルは Python でもそのまま読める）。
 * @param {{ name: string, version?: string | null }} bundle
 * @param {Record<string, unknown>} [overrides] Bundle の設定の上書き（例: wholeBody の highres）
 * @param {number} [channels] 重ねるシリーズの数（inputs/0.npz … の本数。1〜8）
 */
export function buildScript(bundle, overrides = {}, channels = 1) {
  if (!NAME_RE.test(bundle.name)) throw new Error("bad-bundle-name");
  if (!Number.isInteger(channels) || channels < 1 || channels > 8) throw new Error("bad-channels");
  const cfg = JSON.stringify({ bundle: { name: bundle.name, version: bundle.version ?? null }, overrides, channels });
  return `GRAPHY = __import__('json').loads(${JSON.stringify(cfg)})\n` + PY;
}

export function validBundleName(name) {
  return NAME_RE.test(name);
}

/**
 * 「最新に更新」で計算機に送るコード（画像は送らない）。Model Zoo の model_info.json と各 Bundle の configs/ を取り、
 * 生のまま outputs/zoo.json に書く（一覧への変換は zooEntry。生成スクリプトと同じ関数）。
 */
export const REFRESH_PY = String.raw`
import json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor

RAW = 'https://raw.githubusercontent.com/Project-MONAI/model-zoo/dev/models'


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception:
        return None


info = get(RAW + '/model_info.json')
if not info:
    raise RuntimeError('zoo-fetch-failed: model_info.json を取得できませんでした')
versions = {}
for key in info:
    m = re.match(r'^(.*)_v(\d+\.\d+\.\d+)$', key)
    if m:
        versions.setdefault(m.group(1), []).append(m.group(2))
names = sorted(versions)
with ThreadPoolExecutor(8) as ex:
    metas = list(ex.map(lambda n: get(RAW + '/' + n + '/configs/metadata.json'), names))
    infs = list(ex.map(lambda n: get(RAW + '/' + n + '/configs/inference.json'), names))
json.dump({'entries': [{'name': n, 'versions': versions[n], 'metadata': m, 'inference': i}
                       for n, m, i in zip(names, metas, infs)]},
          open('outputs/zoo.json', 'w', encoding='utf-8'), ensure_ascii=False)
`;

/** 本体がデータを作らなかった理由（よく出るものだけ）。 */
const REFUSALS = {
  "npz-duplicate-positions": "同じ位置のスライスが 2 枚以上あります（撮影が 2 回ぶん混ざったシリーズなど）。1 回ぶんだけのシリーズで試してください",
  "npz-uneven-spacing": "スライスの間隔が揃っていません（欠けたスライスがある）",
  "npz-mixed-geometry": "向きや大きさの違う画像が混ざっています",
  "npz-too-large": "シリーズが大きすぎます",
  "permission-denied": "このプラグインに外部の計算機を使う許可がありません",
  "no-endpoint": "環境設定 ＞ 外部の計算機 で計算機を登録してください",
  "colab-signin-required": "環境設定 ＞ 外部の計算機 で Google にログインしてください（ログインすると Colab の GPU T4 が自動で登録されます）",
  "t4-not-available": "いまのプランでは Colab の GPU T4 が使えません。環境設定 ＞ 外部の計算機 で別の種類を登録してください",
  "colab-not-configured": "このアプリには Colab に接続するための設定が入っていません。Jupyter Server を登録してください",
};
export const explain = (code) => (code && REFUSALS[code] ? `${REFUSALS[code]}（${code}）` : String(code));

// ---------------------------------------------------------------------------
// 純関数（node --test で試す）
// ---------------------------------------------------------------------------

/**
 * numpy の .npy（v1/v2・C 順）を読む。対応は u1/u2/i4/f4。
 * @param {Uint8Array} bytes
 */
export function parseNpy(bytes) {
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  if (bytes.length < 10 || magic.some((b, i) => bytes[i] !== b)) throw new Error("npy-magic");
  const major = bytes[6];
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(bytes.subarray(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*True/.test(header);
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "")
    .split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  if (fortran) throw new Error("npy-fortran-order");
  const n = shape.reduce((a, b) => a * b, 1);
  const off = hstart + hlen;
  const body = bytes.slice(off); // 揃った位置から読むために複写する
  /** @type {Record<string, (b: ArrayBuffer) => ArrayLike<number>>} */
  const make = {
    "|u1": (b) => new Uint8Array(b, 0, n),
    "<u1": (b) => new Uint8Array(b, 0, n),
    "<u2": (b) => new Uint16Array(b, 0, n),
    "<i4": (b) => new Int32Array(b, 0, n),
    "<f4": (b) => new Float32Array(b, 0, n),
  };
  if (!descr || !make[descr]) throw new Error("npy-dtype " + descr);
  return { dtype: descr, shape, data: make[descr](body.buffer) };
}

const normModality = (m) => {
  const u = String(m ?? "").toUpperCase();
  return u === "MRI" ? "MR" : u;
};

/**
 * Bundle の network_data_format から、表示中のシリーズに使えるかを判定する。
 * 止めるのは明らかに合わないときだけ（メタデータは作者の自己申告なので、細かい判定はしない）。
 * @param {any} meta metadata.json
 * @param {{ modality: string, channels?: number }} target channels は重ねるシリーズの数（既定 1）
 * @returns {{ ok: boolean, reasons: string[], warnings: string[], labels: Array<{ value: number, name: string }> }}
 */
export function judge(meta, target) {
  const channels = Number(target.channels ?? 1);
  const reasons = [];
  const warnings = [];
  const fmt = meta && meta.network_data_format;
  const inputs = (fmt && fmt.inputs) || {};
  const outputs = (fmt && fmt.outputs) || {};
  const inKeys = Object.keys(inputs);
  const outKeys = Object.keys(outputs);
  if (!fmt || inKeys.length === 0 || outKeys.length === 0) {
    reasons.push("入力・出力の形式（network_data_format）が書かれていないので、セグメンテーションのモデルと確認できません");
  }
  if (inKeys.length > 1) reasons.push(`入力が ${inKeys.length} つ要るモデルです（${inKeys.join("・")}）。1 シリーズでは動かせません`);
  const image = inputs.image ?? inputs[inKeys[0]];
  if (image) {
    const ch = Number(image.num_channels ?? 1);
    if (ch !== channels) reasons.push(`入力のチャネル数が ${ch} です（選ばれたシリーズは ${channels} 本）。チャネルの数だけシリーズを選んでください`);
    const want = String(image.modality ?? "")
      .split(/[\/,\s]+/).map(normModality).filter(Boolean);
    const have = normModality(target.modality);
    if (want.length > 0 && !want.includes(have)) reasons.push(`このモデルは ${want.join("・")} 用です（表示中は ${have || "不明"}）`);
    // HU の規則はモダリティが書かれていないときだけ（MR なのに hounsfield と書いた Bundle がある）
    if (want.length === 0 && String(image.format ?? "").toLowerCase() === "hounsfield" && have !== "CT") {
      reasons.push("入力は CT の HU 値を前提にしています");
    }
  }
  const pred = outputs.pred ?? outputs[outKeys[0]];
  if (pred && !["segmentation", "labels", "label"].includes(String(pred.format ?? "").toLowerCase())) {
    reasons.push(`出力が「${pred.format}」です。このサンプルはセグメンテーションだけを扱います`);
  }
  const labels = [];
  const def = pred && pred.channel_def;
  if (def && typeof def === "object") {
    for (const [k, v] of Object.entries(def)) {
      const value = Number(k);
      if (Number.isInteger(value) && value > 0) labels.push({ value, name: String(v) });
    }
  } else {
    warnings.push("ラベル名（channel_def）が書かれていません。番号で表示します");
  }
  labels.sort((a, b) => a.value - b.value);
  return { ok: reasons.length === 0, reasons, warnings, labels };
}

/**
 * 計算機から返ったラベル（npz と同じ格子 [z, y, x]）のスライスを、loadVolume の格子のスライスへ対応づける。
 * 患者座標で照らし合わせる（並び順の違いに強く、1 枚ずれていれば必ず止まる）。
 * @param {{ spacing: number[], origin: number[], direction: number[][] }} g  labels.json の geometry
 * @param {[number, number, number]} shapeZyx
 * @param {{ dims: [number, number, number], worldToIndex: number[] }} vol
 * @returns {{ ok: true, kMap: Int32Array } | { ok: false, error: string }}
 */
export function mapSlices(g, shapeZyx, vol) {
  const [nz, ny, nx] = shapeZyx;
  const [vx, vy, vz] = vol.dims;
  if (nx !== vx || ny !== vy || nz !== vz) return { ok: false, error: `grid-mismatch: ${nx}x${ny}x${nz} vs ${vx}x${vy}x${vz}` };
  const [dz, dy, dx] = g.spacing;
  const w2i = vol.worldToIndex;
  const toIndex = (p) => [0, 1, 2].map((r) => w2i[r * 4] * p[0] + w2i[r * 4 + 1] * p[1] + w2i[r * 4 + 2] * p[2] + w2i[r * 4 + 3]);
  const world = (i, j, k) => [0, 1, 2].map((a) =>
    g.origin[a] + i * dx * g.direction[0][a] + j * dy * g.direction[1][a] + k * dz * g.direction[2][a]);
  const kMap = new Int32Array(nz);
  const seen = new Uint8Array(nz);
  for (let k = 0; k < nz; k++) {
    const c0 = toIndex(world(0, 0, k));
    const ci = toIndex(world(nx - 1, 0, k));
    const cj = toIndex(world(0, ny - 1, k));
    const kk = Math.round(c0[2]);
    const tol = 0.25;
    const okPlane = Math.abs(c0[0]) < tol && Math.abs(c0[1]) < tol &&
      Math.abs(ci[0] - (nx - 1)) < tol && Math.abs(ci[1]) < tol &&
      Math.abs(cj[0]) < tol && Math.abs(cj[1] - (ny - 1)) < tol;
    if (!okPlane || Math.abs(c0[2] - kk) > tol || kk < 0 || kk >= nz || seen[kk]) {
      return { ok: false, error: `grid-mismatch at slice ${k}` };
    }
    seen[kk] = 1;
    kMap[k] = kk;
  }
  return { ok: true, kMap };
}

/**
 * 計算機から返ったラベル（npz の並び）を、`loadVolume` の並びの 1 本の volume に写す（同じ型のまま）。
 * 保存（H64）も表示（H65）もこの 1 本を渡す（ラベルごとに volume 大の配列を作らない）。
 * @param {Uint8Array | Uint16Array} labels [z, y, x]
 * @param {Int32Array} kMap  npz の k → 本体の k
 * @param {number} nxy  1 スライスの画素数
 */
export function reorderLabels(labels, kMap, nxy) {
  const out = new /** @type {any} */ (labels.constructor)(labels.length);
  for (let k = 0; k < kMap.length; k++) {
    out.set(labels.subarray(k * nxy, (k + 1) * nxy), kMap[k] * nxy);
  }
  return out;
}

/**
 * ラベルの表（H64・H65 に渡す形）。名前は Bundle の channel_def、無ければ番号。
 * @param {number[]} values
 * @param {Map<number, string>} names
 * @param {string} description
 */
export function labelTable(values, names, description) {
  return values.map((v) => ({ value: v, label: names.get(v) ?? `label ${v}`, color: colorFor(v), description }));
}

/** ラベル番号ごとに見分けやすい色（黄金角で色相を回す）。 */
export function colorFor(value) {
  const h = (value * 137.508) % 360;
  const s = 0.75, l = 0.55;
  const f = (n) => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return /** @type {[number, number, number]} */ ([f(0), f(8), f(4)]);
}

// ---------------------------------------------------------------------------
// 画面
// ---------------------------------------------------------------------------

/** @param {any} host */
export async function activate(host) {
  const target = (host.getTargets?.() ?? []).find((t) => t.kind === "image");
  if (!target) {
    host.notify("MONAI: 画像のタイルを選んでから開いてください");
    return;
  }
  const win = host.openWindow({ title: "MONAI Bundle（外部の計算機）", width: 720, height: 760 });
  const root = win.container;
  root.style.cssText = "font: 13px system-ui, sans-serif; padding: 12px; overflow: auto; display: flex; flex-direction: column; gap: 8px;";
  /** @type {Record<string, any>} */
  const state = (/** @type {any} */ (window).__monaiState = { phase: "idle" });

  const el = (tag, props = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "testid") e.dataset.testid = v;
      else if (k === "style") e.style.cssText = v;
      else e[k] = v;
    }
    for (const c of kids) e.append(c);
    return e;
  };
  const have = normModality(target.modality);
  /** 一覧（同梱のスナップショット。「最新に更新」で置き換わる） */
  let zoo = { entries: ZOO.entries, fetched: ZOO.fetched };
  const entryOf = (name) => zoo.entries.find((e) => e.name === name);
  const picker = el("select", { testid: "monai-bundle", style: "flex: 1; min-width: 0; max-width: 100%" });
  const versionSel = el("select", { testid: "monai-version", title: "Bundle の版（既定は最新）" });
  const refreshBtn = el("button", { textContent: "一覧を更新", testid: "monai-refresh", style: "white-space: nowrap", title: "計算機で Model Zoo の最新の一覧を取得します（画像は送りません）" });
  const listNote = el("div", { testid: "monai-list-note", style: "font-size: 11px; color: #52606d" });
  const group = (label) => el("optgroup", { label });
  const fillPicker = () => {
    const rows = zoo.entries.map((e) => ({ e, c: classify(e, have) }));
    const opt = (e, text, disabled) => el("option", { value: e.name, textContent: text, disabled });
    const mark = (c) => (c.kind === "ran" ? "" : "（未検証）");
    const usable = group("使える");
    const order = (r) => (r.c.kind === "ran" ? RAN.names.indexOf(r.e.name) : 100);
    for (const { e, c } of rows.filter((r) => r.c.kind === "ran" || r.c.kind === "untested").sort((x, y) => order(x) - order(y))) {
      usable.append(opt(e, `${labelOf(e)} — ${e.name}${mark(c)}`, false));
    }
    const stopped = group(`${have || "?"} のこのシリーズには使えない`);
    for (const { e, c } of rows.filter((r) => r.c.kind === "stopped")) {
      stopped.append(opt(e, `${e.name} — ${(c.reasons[0] ?? "").slice(0, 36)}…`, true));
    }
    const prev = picker.value;
    picker.replaceChildren(usable, stopped, el("option", { value: OTHER, textContent: "その他（名前を入れる）" }));
    const first = usable.querySelector("option:not([disabled])");
    picker.value = [...picker.options].some((o) => o.value === prev && !o.disabled) ? prev : (first ? first.value : OTHER);
    const n = usable.children.length;
    listNote.textContent = `Model Zoo ${zoo.entries.length} 本のうち、このシリーズに使えるのは ${n} 本（${zoo.fetched ?? "取得日不明"} 時点）。` +
      `「未検証」は、判定を通っただけで Colab で最後まで通したことはありません。`;
  };
  const fillVersions = () => {
    const e = entryOf(picker.value);
    versionSel.replaceChildren(...(e ? e.versions : []).map((v, i) => el("option", { value: v, textContent: i === 0 ? `${v}（最新）` : v })));
    versionSel.style.display = e ? "" : "none";
  };
  fillPicker();
  fillVersions();
  const settingsBox = el("div", { testid: "monai-settings", style: "display: none; flex-direction: column; gap: 4px; border: 1px solid #ccd; border-radius: 4px; padding: 6px 8px" });
  /** 画面の設定の部品。名前 → 値を読む関数 */
  let settingReaders = {};
  let fields = [];
  const fillSettings = () => {
    fields = settingFields(entryOf(picker.value)?.displayable);
    settingReaders = {};
    settingsBox.style.display = fields.length ? "flex" : "none";
    const rows = [el("div", { style: "font-weight: 600" }, "このモデルの設定（Bundle が公開したもの。範囲は Bundle に書かれていません）")];
    for (const f of fields) {
      const input = f.kind === "bool"
        ? el("input", { type: "checkbox", checked: f.def, testid: `monai-setting-${f.name}` })
        : el("input", { type: "text", value: String(f.def), size: 8, testid: `monai-setting-${f.name}` });
      settingReaders[f.name] = () => (f.kind === "bool" ? input.checked : input.value);
      rows.push(el("label", { style: "display: flex; gap: 8px; align-items: center" }, input, `${f.name}（既定 ${String(f.def)}・範囲不明）`));
    }
    settingsBox.replaceChildren(...rows);
  };
  // 複数チャネルのモデル: チャネルごとにタイル（シリーズ）を選ぶ。並びは Bundle の定義で、取り違えると結果が壊れる
  const channelsBox = el("div", { testid: "monai-channels", style: "display: none; flex-direction: column; gap: 4px; border: 1px solid #ccd; border-radius: 4px; padding: 6px 8px" });
  const channelNote = el("div", { testid: "monai-channels-note", style: "font-size: 11px" });
  /** @type {import("./graphy-plugin").ViewerTarget[]} */
  let tiles = [];
  /** チャネルの数だけの select（チャネル 0 から） */
  let chanSels = [];
  let plan = null;
  /** この実行で、結果を重ねる基準（チャネル 0 のシリーズ） */
  let ref = target;
  const tileLabel = (t) => `${t.seriesLabel}（${t.modality}・${t.sliceCount} 枚）`;
  /** 選ばれたタイル。足りない・重複・モダリティ違いなどの理由も返す */
  const chosenTiles = () => {
    if (!plan) return { tiles: [target], problem: null };
    const picked = chanSels.map((sel) => tiles.find((t) => t.tileId === sel.value) ?? null);
    const missing = picked.map((t, i) => (t ? -1 : i)).filter((i) => i >= 0);
    if (missing.length) return { tiles: picked, problem: `チャネル ${missing.join("・")}（${missing.map((i) => plan.names[i]).join("・")}）のシリーズが未選択です` };
    if (new Set(picked.map((t) => t.seriesUid)).size !== picked.length) return { tiles: picked, problem: "同じシリーズを複数のチャネルに選んでいます" };
    if (new Set(picked.map((t) => normModality(t.modality))).size !== 1) return { tiles: picked, problem: "チャネルのモダリティが揃っていません" };
    if (new Set(picked.map((t) => t.sliceCount)).size !== 1) return { tiles: picked, problem: "チャネルごとの枚数が違います（同じ格子のシリーズが要ります）" };
    return { tiles: picked, problem: null };
  };
  const syncChannelNote = () => {
    const c = chosenTiles();
    channelNote.textContent = c.problem ?? "選択は揃っています。格子（間隔・位置・向き）の一致は、計算機で実行前に確認します。";
    channelNote.style.color = c.problem ? "#8a4b00" : "#1b6e3a";
    runBtn.disabled = state.phase === "running" || !!c.problem;
  };
  const fillChannels = () => {
    plan = channelPlan(entryOf(picker.value));
    channelsBox.style.display = plan ? "flex" : "none";
    chanSels = [];
    if (!plan) { runBtn.disabled = state.phase === "running"; return; }
    const allTargets = host.getTargets?.() ?? [];
    tiles = allTargets.filter((t) => t.kind === "image" && t.patientKey === target.patientKey);
    const rows = [el("div", { style: "font-weight: 600" }, `このモデルは ${plan.n} 本のシリーズを重ねて入力します（並びは ${plan.order === "metadata" ? "Bundle の metadata" : "Bundle の設定（inference.json）"} の定義です）`)];
    plan.names.forEach((nm, i) => {
      const sel = el("select", { testid: `monai-channel-${i}`, style: "flex: 1; min-width: 0" },
        el("option", { value: "", textContent: "— シリーズを選んでください —" }),
        ...tiles.map((t) => el("option", { value: t.tileId, textContent: tileLabel(t) })));
      const hint = tiles.filter((t) => channelNameMatches(t.seriesLabel, nm));
      if (hint.length === 1) sel.value = hint[0].tileId;
      sel.addEventListener("change", syncChannelNote);
      chanSels.push(sel);
      rows.push(el("label", { style: "display: flex; gap: 8px; align-items: center" }, el("span", { style: "width: 6em; font-weight: 600" }, `${i}: ${nm}`), sel));
    });
    const reload = el("button", { textContent: "開いているタイルを読み直す", testid: "monai-channels-reload", style: "align-self: flex-start" });
    reload.addEventListener("click", fillChannels);
    rows.push(channelNote, el("div", { style: "font-size: 11px; color: #52606d" }, `シリーズは、プラグインを開く前に、同じ患者の画像タイルとして開いておいてください（あとから足したタイルは見えません。選べるタイル数は、ビューアのタイル ${allTargets.length} 枚のうち ${tiles.length} 枚）。`), reload);
    channelsBox.replaceChildren(...rows);
    syncChannelNote();
  };
  const nameInput = el("input", { testid: "monai-bundle-name", placeholder: "例: MONAI/spleen_ct_segmentation", style: "flex: 1" });
  const runBtn = el("button", { textContent: "実行", testid: "monai-run" });
  const status = el("div", { testid: "monai-status", style: "color: #52606d; min-height: 1.2em" });
  const info = el("div", { testid: "monai-info" });
  const preview = el("canvas", { testid: "monai-preview", style: "display: none; max-width: 100%; image-rendering: pixelated; border: 1px solid #ccd" });
  const labelBox = el("div", { testid: "monai-labels", style: "display: flex; flex-wrap: wrap; gap: 4px 12px" });
  const saveBtn = el("button", { textContent: "選んだラベルを SEG で保存", testid: "monai-save", disabled: true });
  const result = el("div", { testid: "monai-result" });
  const otherRow = el("div", { style: "display: none; gap: 6px" }, nameInput);
  root.append(
    el("div", {
      testid: "monai-research-only",
      textContent: "研究用です。診断には使わないでください。結果は必ず画像で確かめてから使ってください。",
      style: "font-size: 12px; color: #8a4b00; background: #fdf0e3; border: 1px solid #e0b884; border-radius: 4px; padding: 4px 8px",
    }),
    el("div", {}, `対象: ${target.seriesLabel}（${target.modality}・${target.sliceCount} 枚）`),
    el("div", { style: "display: flex; gap: 6px" }, picker, versionSel, runBtn),
    otherRow,
    settingsBox,
    channelsBox,
    el("div", { style: "display: flex; gap: 6px; align-items: center" }, refreshBtn, listNote),
    el("div", { style: "font-size: 11px; color: #52606d" },
      "「実行」で、匿名化したこのシリーズを送り、計算機の上でモデルの説明（metadata.json）を読んで使えるかを判定してから推論します。" +
      "合わなければ推論せずに止まります。ライセンスは結果と一緒に出ます。"),
    status, info, preview, labelBox, saveBtn, result,
  );
  const syncPicker = () => { otherRow.style.display = picker.value === OTHER ? "flex" : "none"; fillVersions(); fillSettings(); fillChannels(); };
  picker.addEventListener("change", syncPicker);
  syncPicker();
  const setStatus = (t) => { status.textContent = t; };
  const onProgress = (p, m) => setStatus(`${Math.round(p * 100)}% ${m ?? ""}`);
  // phase は結果を読み終えてから idle に戻す（外から「終わったか」を見る印）
  const busy = (b) => { state.phase = b ? "running" : "idle"; runBtn.disabled = b || (!b && !!plan && !!chosenTiles().problem); picker.disabled = b; chanSels.forEach((x) => { x.disabled = b; }); versionSel.disabled = b; refreshBtn.disabled = b; settingsBox.querySelectorAll('input').forEach((i) => { i.disabled = b; }); saveBtn.disabled = b || !state.labels; };
  win.setCloseGuard?.(() => (state.phase === "running" ? "計算の途中です。閉じると結果を受け取れません。" : null));
  // 閉じたら、この窓で計算に使った Colab のランタイムを解放するかを聞く（確認は本体が出す）。
  // 解放しても次の「実行」が自動で確保し直す
  win.onClose(async () => {
    if (!state.usedCompute || !host.compute.status) return;
    for (const e of await host.compute.status()) {
      if (e.kind === "colab" && e.runtime?.allocated) {
        state.release = await host.compute.releaseRuntime(e.id, { ask: true });
      }
    }
  });

  /** @param {any} b @param {ReturnType<typeof judge>} v */
  function showInfo(b, v) {
    const m = b.metadata ?? {};
    const img = m.network_data_format?.inputs?.image ?? {};
    const rows = [
      ["名前", `${b.name}（版 ${m.version ?? "?"}）`],
      ["説明", m.description ?? ""],
      ["入力", `${img.modality ?? "?"}・${img.format ?? "?"}・チャネル ${img.num_channels ?? "?"}・${JSON.stringify(img.spatial_shape ?? "?")}`],
      ["出力", `${v.labels.length} ラベル: ${v.labels.slice(0, 12).map((l) => l.name).join("、")}${v.labels.length > 12 ? " …" : ""}`],
      ["著者", [].concat(m.authors ?? []).join(", ")],
      ["権利", m.copyright ?? ""],
      ["ライセンス", (b.license ?? "（LICENSE ファイルなし）").split("\n").find((l) => l.trim()) ?? ""],
      ["学習データの条件", (b.dataLicense ?? "").split("\n").find((l) => l.trim()) ?? "（記載なし）"],
      ["大きさ", `${(b.bytes / 1048576).toFixed(0)} MB・MONAI ${b.monai}`],
    ];
    const table = el("table", { style: "border-collapse: collapse; font-size: 12px" },
      ...rows.map(([k, val]) => el("tr", {},
        el("th", { textContent: k, style: "text-align: left; padding: 2px 8px 2px 0; vertical-align: top; color: #334e68; white-space: nowrap" }),
        el("td", { textContent: val, style: "padding: 2px 0" }))));
    info.replaceChildren(table);
    for (const w of v.warnings) info.append(el("div", { textContent: "⚠ " + w, style: "color: #8a4b00" }));
    // 止めた理由は計算機の上の判定（bundle.json の reasons）を正とする
    for (const r of b.reasons ?? v.reasons) info.append(el("div", { textContent: "✕ " + r, style: "color: #b42318" }));
  }

  refreshBtn.addEventListener("click", async () => {
    busy(true);
    state.usedCompute = true;
    setStatus("Model Zoo の一覧を取得しています…");
    const r = await host.compute.runJob({ inputs: [], script: REFRESH_PY, timeoutSec: 600 }, { onProgress });
    if (!r.ok) { setStatus(r.cancelled ? "取り消しました" : `失敗: ${explain(r.error)}`); busy(false); return; }
    const raw = r.status === "ok" ? await r.readFile("zoo.json") : null;
    if (!raw) { setStatus(`失敗: ${r.errorName ?? "一覧が返りませんでした"}${r.errorValue ? ": " + r.errorValue : ""}`); busy(false); return; }
    const got = JSON.parse(new TextDecoder().decode(raw)).entries;
    zoo = { entries: got.map((x) => zooEntry(x.name, x.versions, x.metadata, x.inference)), fetched: new Date().toISOString().slice(0, 10) };
    state.zoo = { count: zoo.entries.length, fetched: zoo.fetched };
    fillPicker(); syncPicker();
    setStatus(`一覧を更新しました（${zoo.entries.length} 本）`);
    busy(false);
  });

  runBtn.addEventListener("click", async () => {
    const name = picker.value === OTHER ? nameInput.value.trim() : picker.value;
    if (!validBundleName(name)) { setStatus("名前に使えない文字があります"); return; }
    const values = {};
    for (const f of fields) values[f.name] = settingReaders[f.name]();
    const set = settingOverrides(fields, values);
    if (!set.ok) { setStatus(set.error); return; }
    const chosen = chosenTiles();
    if (chosen.problem) { setStatus(chosen.problem); return; }
    ref = chosen.tiles[0];
    state.overrides = set.overrides;
    state.bundle = null; state.verdict = null; state.labels = null; state.summary = undefined; state.error = undefined;
    info.replaceChildren(); labelBox.replaceChildren(); result.replaceChildren(); preview.style.display = "none";
    busy(true);
    state.usedCompute = true;
    setStatus("送っています…");
    const r = await host.compute.runJob({
      inputs: chosen.tiles.map((t) => ({ studyUid: t.studyUid, seriesUid: t.seriesUid, format: "npz" })),
      script: buildScript({ name, version: picker.value === OTHER || versionSel.value === entryOf(name)?.versions[0] ? null : versionSel.value }, set.overrides, chosen.tiles.length),
      timeoutSec: 3600,
    }, { onProgress });
    if (!r.ok) {
      state.error = r.error;
      setStatus(r.cancelled ? "取り消しました" : `失敗: ${explain(r.error)}`);
      busy(false);
      return;
    }
    // モデルの説明は、合わなかったとき（推論しない）も返る
    const raw = await r.readFile("bundle.json");
    const b = raw ? JSON.parse(new TextDecoder().decode(raw)) : null;
    if (b) {
      state.bundle = b;
      state.verdict = applyLabelNames(judge(b.metadata, { modality: ref.modality, channels: chosen.tiles.length }), b.name);
      showInfo(b, state.verdict);
    }
    if (r.status !== "ok") {
      state.error = `${r.errorName}: ${r.errorValue}`;
      state.traceback = r.traceback;
      state.stderr = r.stderr;
      const notApplicable = /^not-applicable/.test(r.errorValue ?? "");
      setStatus(notApplicable ? "このシリーズには使えないモデルです（推論はしていません）" : `失敗: ${r.errorName}: ${r.errorValue}`);
      busy(false);
      return;
    }
    const [lj, ln] = await Promise.all([r.readFile("labels.json"), r.readFile("labels.npy")]);
    if (!lj || !ln) { setStatus("失敗: 結果が返りませんでした"); busy(false); return; }
    const summary = JSON.parse(new TextDecoder().decode(lj));
    const npy = parseNpy(ln);
    setStatus("ボリュームを読み込んでいます…");
    const vol = await host.loadVolume({ studyUid: ref.studyUid, seriesUid: ref.seriesUid });
    if (!vol) { setStatus("失敗: ボリュームを読めませんでした"); busy(false); return; }
    const mapped = mapSlices(summary.geometry, /** @type {any} */ (npy.shape), vol);
    if (!mapped.ok) { state.error = mapped.error; setStatus(`失敗: ${mapped.error}`); busy(false); return; }
    state.summary = summary;
    state.labels = { data: reorderLabels(/** @type {any} */ (npy.data), mapped.kMap, vol.dims[0] * vol.dims[1]), vol };
    showResult(summary);
    // H65: 結果をそのまま ROI マネージャへ（表示だけ。保存は「SEG で保存」）
    if (host.showLabelVolume) {
      const names = new Map((state.verdict?.labels ?? []).map((l) => [l.value, l.name]));
      const values = state.present.map((x) => x.value);
      state.shown = await host.showLabelVolume(ref.tileId, {
        grid: { dims: vol.dims, ipp: vol.ipp, sliceStep: vol.sliceStep },
        data: state.labels.data,
        table: labelTable(values, names, ""),
        label: `MONAI ${state.bundle?.name ?? ""}`.trim(),
      });
      result.textContent = state.shown.ok
        ? `ROI マネージャに読み込みました（${state.shown.segmentCount} ラベル）。保存するときは下の「SEG で保存」を押してください。`
        : `ROI マネージャに読み込めませんでした: ${state.shown.error}`;
    }
    setStatus(`できました（${summary.stages?.done ?? "?"} 秒${summary.gpu ? "・" + summary.gpu.name + "・最大 " + summary.gpu.peakMiB + " MiB" : ""}）`);
    busy(false);
  });

  /** @param {any} summary */
  function showResult(summary) {
    const { data, vol } = state.labels;
    const [nx, ny, nz] = vol.dims;
    const nxy = nx * ny;
    const names = new Map((state.verdict?.labels ?? []).map((l) => [l.value, l.name]));
    const present = Object.entries(summary.labels).map(([v, c]) => ({ value: Number(v), count: c })).filter((x) => x.value > 0);
    // 下見: ラベルがいちばん多いスライス
    let bestK = 0, best = -1;
    for (let k = 0; k < nz; k++) {
      let c = 0;
      for (let p = k * nxy; p < (k + 1) * nxy; p++) if (data[p] > 0) c++;
      if (c > best) { best = c; bestK = k; }
    }
    const kv = bestK; // data は本体の並びに写してある
    preview.width = nx; preview.height = ny;
    const ctx = /** @type {CanvasRenderingContext2D} */ (preview.getContext("2d"));
    const im = ctx.createImageData(nx, ny);
    // 濃淡の窓: CT は腹部の窓（HU 40±200）、それ以外（MR など）はそのスライスの 2〜98 パーセンタイル
    let lo = 40 - 200, hi = 40 + 200;
    if (String(vol.modality).toUpperCase() !== "CT") {
      const sl = Array.from(vol.data.subarray(kv * nxy, (kv + 1) * nxy)).sort((a, b) => a - b);
      lo = sl[Math.floor(sl.length * 0.02)];
      hi = sl[Math.floor(sl.length * 0.98)];
      if (!(hi > lo)) hi = lo + 1;
    }
    for (let p = 0; p < nxy; p++) {
      const g = Math.max(0, Math.min(255, ((vol.data[kv * nxy + p] - lo) / (hi - lo)) * 255));
      const v = data[bestK * nxy + p];
      const c = v > 0 ? colorFor(v) : null;
      im.data[p * 4] = c ? (g + c[0]) / 2 : g;
      im.data[p * 4 + 1] = c ? (g + c[1]) / 2 : g;
      im.data[p * 4 + 2] = c ? (g + c[2]) / 2 : g;
      im.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    preview.style.display = "block";
    preview.style.width = Math.min(nx, 512) + "px";
    labelBox.replaceChildren(...present.map((x) => {
      const cb = el("input", { type: "checkbox", checked: true, value: String(x.value) });
      const [r, g, b] = colorFor(x.value);
      return el("label", {},
        cb, el("span", { textContent: "■", style: `color: rgb(${r},${g},${b})` }),
        ` ${names.get(x.value) ?? "label " + x.value}（${x.count}）`);
    }));
    state.present = present;
  }

  saveBtn.addEventListener("click", async () => {
    if (!state.labels) return;
    const values = [...labelBox.querySelectorAll("input:checked")].map((c) => Number(/** @type {HTMLInputElement} */ (c).value));
    if (values.length === 0) { setStatus("保存するラベルを選んでください"); return; }
    const { data, vol } = state.labels;
    busy(true);
    const names = new Map((state.verdict?.labels ?? []).map((l) => [l.value, l.name]));
    // H64: ラベルの volume 1 本と表を渡す（選ばなかったラベルは表に入れない＝背景になる）
    const res = await host.saveSegmentation({
      reference: { studyUid: ref.studyUid, seriesUid: ref.seriesUid },
      grid: { dims: vol.dims, spacing: vol.spacing, ipp: vol.ipp, iop: vol.iop, sliceStep: vol.sliceStep },
      seriesDescription: `MONAI ${state.bundle.name}`,
      labels: { data, table: labelTable(values, names, `${state.bundle.name} ${state.summary.version ?? ""}`.trim()) },
    });
    state.saved = res;
    result.textContent = res.ok ? `SEG を保存しました（${values.length} ラベル・${res.seriesInstanceUid}）。`
      : res.cancelled ? "保存を取り消しました" : `保存に失敗しました: ${res.error}`;
    busy(false);
  });
}
