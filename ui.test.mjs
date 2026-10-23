// node --test examples/remote-compute-monai
//
// 計算機の上のコード（PY）も試すときは、numpy・scipy・nibabel のある Python を渡す:
//   GRAPHY_TEST_PYTHON=C:\Users\...\python.exe [GRAPHY_TEST_PYTHONPATH=<nibabel の場所>] node --test examples/remote-compute-monai
// MONAI は偽物（入力の閾値で 2 ラベルを作る）に差し替えるので、確かめるのは「npz → NIfTI → 結果 → npz の格子」の往復。
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { activate, PY_ENV, PY_PROBE, buildProbeScript, installCommand, missingPackages, NOT_YET, PROMPTS, RAN, buildPrompt, parsePoints, promptProblem, roiToLps, vistaLabelOptions, REFRESH_PY, ZOO, channelPlan, loadKeysOf, LABEL_NAMES, settingFields, settingOverrides, buildScript, classify, colorFor, judge, labelTable, mapSlices, parseNpy, reorderLabels, validBundleName, zooEntry } from "./ui.js";

function npy(descr, shape, bytes) {
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`;
  const total = 10 + header.length + 1;
  header += " ".repeat((64 - (total % 64)) % 64) + "\n";
  const out = new Uint8Array(10 + header.length + bytes.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(out.buffer).setUint16(8, header.length, true);
  out.set(new TextEncoder().encode(header), 10);
  out.set(bytes, 10 + header.length);
  return out;
}

test("parseNpy reads uint8 and uint16", () => {
  const a = parseNpy(npy("|u1", [2, 3], new Uint8Array([1, 2, 3, 4, 5, 6])));
  assert.deepEqual(a.shape, [2, 3]);
  assert.deepEqual([...a.data], [1, 2, 3, 4, 5, 6]);
  const b = parseNpy(npy("<u2", [2], new Uint8Array(new Uint16Array([300, 7]).buffer)));
  assert.deepEqual([...b.data], [300, 7]);
  assert.throws(() => parseNpy(new Uint8Array(20)), /npy-magic/);
});

const spleenMeta = {
  version: "0.5.3",
  network_data_format: {
    inputs: { image: { type: "image", format: "hounsfield", modality: "CT", num_channels: 1, spatial_shape: [96, 96, 96] } },
    outputs: { pred: { type: "image", format: "segmentation", num_channels: 2, channel_def: { 0: "background", 1: "spleen" } } },
  },
};

test("judge accepts a CT segmentation bundle for CT and lists the labels", () => {
  const v = judge(spleenMeta, { modality: "CT" });
  assert.equal(v.ok, true);
  assert.deepEqual(v.labels, [{ value: 1, name: "spleen" }]);
});

test("judge refuses obvious mismatches", () => {
  assert.equal(judge(spleenMeta, { modality: "MR" }).ok, false);
  const brats = {
    network_data_format: {
      inputs: { image: { format: "magnitude", modality: "MRI", num_channels: 4 } },
      outputs: { pred: { format: "segmentation", channel_def: { 0: "TC", 1: "WT", 2: "ET" } } },
    },
  };
  const v = judge(brats, { modality: "MR" });
  assert.equal(v.ok, false);
  assert.match(v.reasons.join(), /チャネル数が 4/);
  const cls = { network_data_format: { inputs: { image: { modality: "CT", num_channels: 1 } }, outputs: { pred: { format: "classification" } } } };
  assert.equal(judge(cls, { modality: "CT" }).ok, false);
});

test("judge tolerates how real bundles describe themselves", () => {
  // prostate_mri_anatomy は出力の形式を "labels" と書く
  const prostate = {
    network_data_format: {
      inputs: { image: { format: "magnitude", modality: "MR", num_channels: 1 } },
      outputs: { pred: { format: "labels", channel_def: { 0: "background", 1: "TZ", 2: "PZ" } } },
    },
  };
  assert.equal(judge(prostate, { modality: "MR" }).ok, true);
  // wholeBrainSeg_Large_UNEST_segmentation は MRI なのに format を "hounsfield" と書く
  const brain = {
    network_data_format: {
      inputs: { image: { format: "hounsfield", modality: "MRI", num_channels: 1 } },
      outputs: { pred: { format: "segmentation", channel_def: { 0: "background", 1: "a" } } },
    },
  };
  assert.equal(judge(brain, { modality: "MR" }).ok, true);
  assert.equal(judge(brain, { modality: "CT" }).ok, false);
});

test("the zoo snapshot has every entry with a valid name and versions, and nulls instead of invented values", () => {
  assert.ok(ZOO.entries.length >= 30, "Model Zoo は 35 本（2026-10-20）");
  assert.match(String(ZOO.fetched), /^\d{4}-\d{2}-\d{2}$/);
  assert.equal(new Set(ZOO.entries.map((e) => e.name)).size, ZOO.entries.length);
  for (const e of ZOO.entries) {
    assert.equal(validBundleName(e.name), true, e.name);
    assert.ok(e.versions.length >= 1, e.name);
  }
  assert.equal(ZOO.entries.find((e) => e.name === "maisi_ct_generative").ndf, null);
});

test("every Bundle that already ran on Colab is in the snapshot and usable for its modality", () => {
  for (const n of RAN.names) {
    const e = ZOO.entries.find((x) => x.name === n);
    assert.ok(e, n);
    const mod = e.ndf.inputs.image.modality.toUpperCase().startsWith("MR") ? "MR" : "CT";
    assert.equal(classify(e, mod).kind, "ran", n);
  }
});

// 実物の metadata.json（2026-10-20 取得）に judge を当てた結果。件数が変わったら、中身を見て更新する
test("classify on the real zoo: CT 10 usable, MR 4 usable; generative / classification / point-prompt stop", () => {
  const usable = (mod) => ZOO.entries.filter((e) => ["ran", "untested"].includes(classify(e, mod).kind)).map((e) => e.name).sort();
  assert.deepEqual(usable("CT"), [
    "multi_organ_segmentation", "pancreas_ct_dints_segmentation", "pediatric_abdominal_ct_segmentation",
    "renalStructures_CECT_segmentation", "renalStructures_UNEST_segmentation", "spleen_ct_segmentation", "spleen_deepedit_annotation", "swin_unetr_btcv_segmentation", "vista3d", "wholeBody_ct_segmentation",
  ].sort());
  assert.deepEqual(usable("MR"), ["brats_mri_segmentation", "prostate_mri_anatomy", "ventricular_short_axis_3label", "wholeBrainSeg_Large_UNEST_segmentation"]);
  for (const n of ["maisi_ct_generative", "breast_density_classification", "lung_nodule_ct_detection", "vista2d"]) {
    const c = classify(ZOO.entries.find((e) => e.name === n), "CT");
    assert.equal(c.kind, "stopped", n);
    assert.ok(c.reasons.length >= 1, n);
  }
  // 点・ラベル番号を渡すモデルは、画面から渡せるので選べる（実機は未確認）
  for (const n of Object.keys(PROMPTS)) assert.equal(classify(ZOO.entries.find((e) => e.name === n), "CT").kind, "untested", n);
});

test("zooEntry keeps only what the picker needs and reports missing metadata as null", () => {
  const e = zooEntry("x", ["0.1.0", "0.10.0", "0.2.0"], spleenMeta, { displayable_configs: { highres: true }, other: 1 });
  assert.deepEqual(e.versions, ["0.10.0", "0.2.0", "0.1.0"]);
  assert.deepEqual(e.displayable, { highres: true });
  assert.equal(e.ndf.inputs.image.modality, "CT");
  const none = zooEntry("y", ["1.0.0"], null, null);
  assert.equal(none.ndf, null);
  assert.equal(none.displayable, null);
});

test("judge stops a bundle without network_data_format", () => {
  const v = judge({}, { modality: "CT" });
  assert.equal(v.ok, false);
  assert.match(v.reasons.join(), /network_data_format/);
});

test("REFRESH_PY is valid Python and sends no image", () => {
  const py = process.env.GRAPHY_TEST_PYTHON || "python3";
  const r = spawnSync(py, ["-I", "-c", "import ast,sys; ast.parse(sys.stdin.read())"], { input: REFRESH_PY, encoding: "utf-8" });
  assert.equal(r.status, 0, r.stderr);
  assert.doesNotMatch(REFRESH_PY, /inputs\//);
});

/** npz の幾何（[dz, dy, dx]・LPS）から loadVolume 風の格子を作る。reversed なら k が逆向き。 */
function volGrid(g, nx, ny, nz, reversed) {
  const [dz, dy, dx] = g.spacing;
  const [r0, r1, r2] = g.direction;
  const o = reversed ? g.origin.map((v, a) => v + (nz - 1) * dz * r2[a]) : g.origin;
  const s = reversed ? -dz : dz;
  // index → world（行 4 要素ずつ）
  const m = [
    [r0[0] * dx, r1[0] * dy, r2[0] * s, o[0]],
    [r0[1] * dx, r1[1] * dy, r2[1] * s, o[1]],
    [r0[2] * dx, r1[2] * dy, r2[2] * s, o[2]],
  ];
  // 逆行列（3x3 部分の逆 + 平行移動）
  const a = m.map((r) => r.slice(0, 3));
  const det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  const inv = [
    [(a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det, (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det, (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det],
    [(a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det, (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det, (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det],
    [(a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det, (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det, (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det],
  ];
  const w2i = [];
  for (let r = 0; r < 3; r++) w2i.push(...inv[r], -(inv[r][0] * o[0] + inv[r][1] * o[1] + inv[r][2] * o[2]));
  w2i.push(0, 0, 0, 1);
  return { dims: [nx, ny, nz], worldToIndex: w2i };
}

const c = Math.cos(Math.PI / 6), s = Math.sin(Math.PI / 6);
const geom = { spacing: [2.5, 0.8, 0.7], origin: [-100, -120, 50], direction: [[c, s, 0], [-s, c, 0], [0, 0, 1]] };

test("mapSlices follows the slice order of the host volume", () => {
  const same = mapSlices(geom, [4, 3, 5], volGrid(geom, 5, 3, 4, false));
  assert.equal(same.ok, true);
  assert.deepEqual([...same.kMap], [0, 1, 2, 3]);
  const rev = mapSlices(geom, [4, 3, 5], volGrid(geom, 5, 3, 4, true));
  assert.equal(rev.ok, true);
  assert.deepEqual([...rev.kMap], [3, 2, 1, 0]);
});

test("mapSlices refuses a grid that is off by one slice or a different size", () => {
  const shifted = { ...geom, origin: geom.origin.map((v, a) => v + 2.5 * geom.direction[2][a]) };
  assert.equal(mapSlices(shifted, [4, 3, 5], volGrid(geom, 5, 3, 4, false)).ok, false);
  assert.equal(mapSlices(geom, [4, 3, 6], volGrid(geom, 5, 3, 4, false)).ok, false);
  const flippedX = { ...geom, direction: [geom.direction[0].map((v) => -v), geom.direction[1], geom.direction[2]] };
  assert.equal(mapSlices(flippedX, [4, 3, 5], volGrid(geom, 5, 3, 4, false)).ok, false);
});

test("reorderLabels writes each npz slice into the mapped host slice, keeping the type", () => {
  // 2 スライス × 2 画素。スライスは逆順に対応
  const out = reorderLabels(new Uint16Array([1, 0, 300, 1]), new Int32Array([1, 0]), 2);
  assert.ok(out instanceof Uint16Array);
  assert.deepEqual([...out], [300, 1, 1, 0]);
});

test("labelTable names labels from channel_def and falls back to the number", () => {
  const t = labelTable([1, 5], new Map([[1, "spleen"]]), "d");
  assert.deepEqual(t.map((x) => [x.value, x.label, x.description]), [[1, "spleen", "d"], [5, "label 5", "d"]]);
  assert.equal(t[0].color.length, 3);
});

test("colorFor gives distinct colors for neighbours", () => {
  assert.notDeepEqual(colorFor(1), colorFor(2));
  for (const v of colorFor(7)) assert.ok(v >= 0 && v <= 255);
});

test("buildScript embeds only the bundle settings and passes the code inspector limits", () => {
  assert.throws(() => buildScript({ name: "x'; import os" }), /bad-bundle-name/);
  assert.throws(() => buildScript({ name: "spleen_ct_segmentation" }, {}, 9), /bad-channels/);
  assert.throws(() => buildScript({ name: "spleen_ct_segmentation" }, {}, 0), /bad-channels/);
  assert.equal(validBundleName("MONAI/spleen_ct_segmentation"), true);
  assert.equal(validBundleName("../etc"), false);
  assert.equal(validBundleName(".."), false);
  const code = buildScript({ name: "wholeBody_ct_segmentation" }, { "displayable_configs#highres": false });
  assert.ok(code.length < 64 * 1024);
  // CodeInspector: base64 風の 200 文字以上・数字の 400 文字以上の連なりがあると「埋め込みデータ」とみなされる
  assert.equal(/[A-Za-z0-9+/=_-]{200,}/.test(code), false);
  assert.equal(/[0-9.,\s-]{400,}/.test(code), false);
  assert.match(code.split("\n")[0], /^GRAPHY = __import__\('json'\)\.loads\(".*wholeBody_ct_segmentation.*displayable_configs#highres.*"\)$/);
});

// ---------------------------------------------------------------------------
// PY の往復（偽の MONAI で）
// ---------------------------------------------------------------------------
const PYTHON = process.env.GRAPHY_TEST_PYTHON;

const FAKE_MONAI = {
  "monai/__init__.py": "__version__ = '0.0-fake'\n",
  "ignite/__init__.py": "",
  // 本処理は torch の有無も確かめる（足りなければ missing-packages）。偽物の MONAI と同じく、試験機に torch が無くても動くように置く
  "torch/__init__.py": "class cuda:\n    @staticmethod\n    def is_available():\n        return False\n",
  "monai/bundle/config_item.py": `
class ComponentLocator:
    def get_component_module_name(self, name):
        return ['monai.networks.fake'] if name == 'FakeNet' else None
`,
  "monai/networks/__init__.py": "",
  "monai/networks/fake.py": `
class FakeNet:
    def __init__(self, in_channels, out_channels):
        pass
`,
  "monai/bundle/__init__.py": `
import json, os, shutil
import numpy as np


class ConfigParser:
    def read_config(self, p):
        self.c = json.load(open(p))

    def get(self):
        return self.c


def download(name, bundle_dir, source, progress=True, version=None, repo=None):
    shutil.copytree(os.environ['FAKE_BUNDLE_SRC'], os.path.join(bundle_dir, name))


def run(config_file, meta_file, bundle_root, dataset_dir, output_dir, datalist=None, logging_file=None, run_id='run', **kw):
    cfg = json.load(open(config_file))
    assert run_id in cfg, ('run id', run_id, list(cfg))
    net = cfg.get('network_def')
    if net is not None:
        assert 'img_size' not in net, 'img_size should have been dropped'
        assert net['in_channels'] == 1
    import nibabel as nib
    if datalist is None:
        datalist = kw['test_datalist']
    json.dump(datalist, open(os.path.join(os.getcwd(), 'datalist.json'), 'w'))
    json.dump({k: v for k, v in kw.items() if k in ('input_dict', 'dataset#data')}, open(os.path.join(os.getcwd(), 'prompt-kw.json'), 'w'))
    first = datalist[0]
    if os.environ.get('FAKE_MULTI'):
        assert isinstance(first, dict), ('want dict datalist', datalist)
        first = list(first.values())[0]
        first = first[0] if isinstance(first, list) else first
    if os.environ.get('FAKE_MULTI'):
        pass
    elif os.environ.get('FAKE_DICT_DATALIST'):
        assert isinstance(first, dict) and 'image' in first, ('want dict datalist', datalist)
        first = first['image']
    else:
        assert isinstance(first, str), ('want path datalist', datalist)
    assert (os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD') or '') == (os.environ.get('WANT_NO_WEIGHTS_ONLY') or ''), 'weights_only env'
    img = nib.load(first)
    a = np.asanyarray(img.dataobj)
    lab = (a > 500).astype(np.uint8) + (a > 1500).astype(np.uint8)
    if os.environ.get('FAKE_NOT_INFERRED'):
        lab[0, 0, :] = 255
    aff = img.affine
    mode = os.environ.get('FAKE_MODE', 'same')
    if mode == 'flip':      # 保存の向きだけ x を反転（同じ場所を指す）
        lab = lab[::-1]
        f = np.eye(4); f[0, 0] = -1; f[0, 3] = a.shape[0] - 1
        aff = aff @ f
    elif mode == 'onehot':  # チャネルごとの one-hot
        lab = np.stack([lab == 0, lab == 1, lab == 2], -1).astype(np.float32)
    os.makedirs(os.path.join(output_dir, 'image'), exist_ok=True)
    nib.save(nib.Nifti1Image(lab, aff), os.path.join(output_dir, 'image', 'image_seg.nii.gz'))
`,
};

const MAKE_INPUT = `
import json, os, numpy as np
c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
vol = np.zeros((6, 7, 9), np.float32)   # [z, y, x]
vol[1, 2, 3] = 1000                     # 非対称な目印
vol[4, 5, 7] = 2000
vol[2, 0, 8] = 700
os.makedirs('inputs', exist_ok=True); os.makedirs('outputs', exist_ok=True)
np.savez('inputs/0.npz', volume=vol, spacing=np.array([2.5, 0.8, 0.7]), origin=np.array([-100.0, -120.0, 50.0]),
         direction=np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]]))
np.save('expected.npy', ((vol > 500).astype(np.uint8) + (vol > 1500).astype(np.uint8)))
import zipfile
with zipfile.ZipFile('inputs/0.npz', 'a') as zf:
    zf.writestr('meta.json', json.dumps({'format': 'graphy-npz/1', 'modality': os.environ.get('FAKE_MODALITY', 'CT')}))
for i in range(1, int(os.environ.get('FAKE_NCH', '1'))):
    v2 = vol[:, :, :-1] if os.environ.get('FAKE_BAD_GRID') == str(i) else vol
    np.savez('inputs/%d.npz' % i, volume=v2, spacing=np.array([2.5, 0.8, 0.7]), origin=np.array([-100.0, -120.0, 50.0]),
             direction=np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]]))
    with zipfile.ZipFile('inputs/%d.npz' % i, 'a') as zf:
        zf.writestr('meta.json', json.dumps({'format': 'graphy-npz/1', 'modality': os.environ.get('FAKE_MODALITY_%d' % i, os.environ.get('FAKE_MODALITY', 'CT'))}))
`;

for (const mode of ["same", "flip", "onehot", "dict-datalist", "third-party"]) {
  test(`PY round trip keeps the voxel grid (${mode})`, { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "graphy-monai-"));
    try {
      const lib = path.join(dir, "lib");
      for (const [p, body] of Object.entries(FAKE_MONAI)) {
        fs.mkdirSync(path.dirname(path.join(lib, p)), { recursive: true });
        fs.writeFileSync(path.join(lib, p), body);
      }
      const bundle = path.join(dir, "bundle");
      fs.mkdirSync(path.join(bundle, "configs"), { recursive: true });
      fs.writeFileSync(path.join(bundle, "configs", "metadata.json"), JSON.stringify(spleenMeta));
      // DiNTS 系は "@datalist" をそのまま dataset に渡す
      const dataset = mode === "dict-datalist" ? { data: "@datalist" } : { data: "$[{'image': i} for i in @datalist]" };
      // 古い書き方（evaluating の入口・廃止された引数）も通ること
      const legacy = mode === "flip";
      const cfg = { datalist: [], output_dir: "x", dataset, ...(legacy
        ? { evaluating: ["$@evaluator.run()"], network_def: { _target_: "FakeNet", in_channels: 1, out_channels: 2, img_size: [96, 96, 96] } }
        : { run: ["$@evaluator.run()"] }) };
      fs.writeFileSync(path.join(bundle, "configs", "inference.json"), JSON.stringify(cfg));
      const run = path.join(dir, "run");
      fs.mkdirSync(run);
      const env = {
        ...process.env,
        PYTHONPATH: [lib, process.env.GRAPHY_TEST_PYTHONPATH].filter(Boolean).join(path.delimiter),
        FAKE_BUNDLE_SRC: bundle,
        FAKE_MODE: mode,
        FAKE_DICT_DATALIST: mode === "dict-datalist" ? "1" : "",
        // 公式（名前に / が無い・MONAI/…）だけ従来の torch.load を許す
        WANT_NO_WEIGHTS_ONLY: mode === "third-party" ? "" : "1",
        HOME: dir,
        USERPROFILE: dir,
        PIP_NO_INDEX: "1", PYTHONIOENCODING: "utf-8", // 万一 pip が呼ばれても手元の環境を書き換えない
      };
      delete env.COLAB_RELEASE_TAG;
      delete env.TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD;
      const py = (code) => {
        const r = spawnSync(PYTHON, ["-c", code], { cwd: run, env, encoding: "utf8" });
        assert.equal(r.status, 0, r.stderr);
        return r.stdout;
      };
      py(MAKE_INPUT);
      const out = py(buildScript({ name: mode === "third-party" ? "someone/fake_bundle" : "fake_bundle" }));
      assert.match(out, /__progress__ 1\.0 done/);
      // NIfTI の向き: voxel (i,j,k) の RAS は npz の LPS の x・y を反転したもの
      py(`
import numpy as np, nibabel as nib
z = np.load('inputs/0.npz'); a = nib.load('work/in/image.nii.gz').affine
sp, o, d = z['spacing'], z['origin'], z['direction']
for i, j, k in [(0, 0, 0), (3, 2, 1), (8, 6, 5)]:
    lps = o + i * sp[2] * d[0] + j * sp[1] * d[1] + k * sp[0] * d[2]
    ras = a @ [i, j, k, 1]
    assert np.allclose(ras[:3], [-lps[0], -lps[1], lps[2]]), (i, j, k, ras, lps)
v = nib.load('work/in/image.nii.gz').get_fdata()
assert v[3, 2, 1] == 1000 and v[7, 5, 4] == 2000
`);
      const labels = parseNpy(new Uint8Array(fs.readFileSync(path.join(run, "outputs", "labels.npy"))));
      const expected = parseNpy(new Uint8Array(fs.readFileSync(path.join(run, "expected.npy"))));
      assert.deepEqual(labels.shape, [6, 7, 9]);
      assert.deepEqual([...labels.data], [...expected.data]);
      const summary = JSON.parse(fs.readFileSync(path.join(run, "outputs", "labels.json"), "utf8"));
      assert.equal(summary.resampled, mode === "flip");
      assert.equal(summary.runId, mode === "flip" ? "evaluating" : "run");
      assert.deepEqual(summary.compat, mode === "flip" ? ["network_def#FakeNet.img_size"] : []);
      assert.deepEqual(summary.labels, { 0: 6 * 7 * 9 - 3, 1: 2, 2: 1 });
      assert.deepEqual(summary.channelDef, { 0: "background", 1: "spleen" });
      // 本体の格子（スライスが逆順）へ写しても目印が同じ場所に来る
      const mapped = mapSlices(summary.geometry, labels.shape, volGrid(summary.geometry, 9, 7, 6, true));
      assert.equal(mapped.ok, true);
      const host = reorderLabels(labels.data, mapped.kMap, 63);
      assert.equal(host[(5 - 1) * 63 + 2 * 9 + 3], 1);
      assert.equal(host[(5 - 4) * 63 + 5 * 9 + 7], 2);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
}

test("PY does not install packages outside Colab", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "graphy-monai-"));
  try {
    const env = { ...process.env, PIP_NO_INDEX: "1", PYTHONIOENCODING: "utf-8", HOME: dir, USERPROFILE: dir, PYTHONPATH: path.join(dir, "empty") };
    delete env.COLAB_RELEASE_TAG;
    const r = spawnSync(PYTHON, ["-c", buildScript({ name: "spleen_ct_segmentation" })], { cwd: dir, env, encoding: "utf8" });
    // 手元の Python に MONAI が入っていれば先へ進むので、そのときは「pip を呼んでいない」ことだけ確かめる
    assert.doesNotMatch(r.stdout + r.stderr, /Collecting|Successfully installed/);
    if (r.status !== 0 && /monai|nibabel|torch/.test(r.stderr)) assert.match(r.stderr, /missing-packages/, "足りないものは、import の例外でなく missing-packages で止まる");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("PY stops before inference when the series does not fit, but still returns the model description", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "graphy-monai-"));
  try {
    const lib = path.join(dir, "lib");
    for (const [p, body] of Object.entries(FAKE_MONAI)) {
      fs.mkdirSync(path.dirname(path.join(lib, p)), { recursive: true });
      fs.writeFileSync(path.join(lib, p), body);
    }
    const bundle = path.join(dir, "bundle");
    fs.mkdirSync(path.join(bundle, "configs"), { recursive: true });
    fs.writeFileSync(path.join(bundle, "configs", "metadata.json"), JSON.stringify(spleenMeta));
    fs.writeFileSync(path.join(bundle, "configs", "inference.json"), JSON.stringify({ datalist: [], output_dir: "x" }));
    fs.writeFileSync(path.join(bundle, "LICENSE"), "Apache License\n");
    const run = path.join(dir, "run");
    fs.mkdirSync(run);
    const env = {
      ...process.env,
      PYTHONPATH: [lib, process.env.GRAPHY_TEST_PYTHONPATH].filter(Boolean).join(path.delimiter),
      FAKE_BUNDLE_SRC: bundle,
      FAKE_MODALITY: "MR",
      HOME: dir,
      USERPROFILE: dir,
      PIP_NO_INDEX: "1", PYTHONIOENCODING: "utf-8",
    };
    delete env.COLAB_RELEASE_TAG;
    const py = (code) => spawnSync(PYTHON, ["-c", code], { cwd: run, env, encoding: "utf8" });
    assert.equal(py(MAKE_INPUT).status, 0);
    const r = py(buildScript({ name: "fake_bundle" }));
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, /not-applicable: このモデルは CT 用です（このシリーズは MR）/);
    const b = JSON.parse(fs.readFileSync(path.join(run, "outputs", "bundle.json"), "utf8"));
    assert.equal(b.modality, "MR");
    assert.equal(b.license.trim(), "Apache License");
    assert.deepEqual(b.reasons, ["このモデルは CT 用です（このシリーズは MR）"]);
    assert.equal(fs.existsSync(path.join(run, "outputs", "labels.npy")), false, "推論していない");
    assert.equal(fs.existsSync(path.join(run, "work")), false, "画像を NIfTI にもしていない");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("settingFields builds parts from displayable_configs with the #-path keys", () => {
  const wb = ZOO.entries.find((e) => e.name === "wholeBody_ct_segmentation");
  const f = settingFields(wb.displayable);
  assert.deepEqual(f.map((x) => [x.key, x.kind, x.def]), [
    ["displayable_configs#highres", "bool", true],
    ["displayable_configs#sw_overlap", "number", 0.25],
    ["displayable_configs#sw_batch_size", "int", 1],
  ]);
  assert.deepEqual(settingFields(null), []);
  // wholeBody 以外に displayable_configs を持つ Bundle は無い（35 本の実物）
  assert.deepEqual(ZOO.entries.filter((e) => e.displayable).map((e) => e.name), ["wholeBody_ct_segmentation"]);
});

test("settingOverrides passes only changed values and rejects bad ones", () => {
  const f = settingFields({ highres: true, sw_overlap: 0.25, sw_batch_size: 1 });
  assert.deepEqual(settingOverrides(f, { highres: true, sw_overlap: "0.25", sw_batch_size: "1" }), { ok: true, overrides: {} });
  assert.deepEqual(settingOverrides(f, { highres: false, sw_overlap: "0.5" }), {
    ok: true, overrides: { "displayable_configs#highres": false, "displayable_configs#sw_overlap": 0.5 },
  });
  for (const bad of [{ sw_overlap: "" }, { sw_overlap: "abc" }, { sw_batch_size: "1.5" }, { highres: "false" }, { sw_overlap: "Infinity" }]) {
    assert.equal(settingOverrides(f, bad).ok, false, JSON.stringify(bad));
  }
});

test("channelPlan reads the channel order from metadata, else from the LoadImaged keys, else says unknown", () => {
  const e = (n) => ZOO.entries.find((x) => x.name === n);
  assert.deepEqual(channelPlan(e("brats_mri_segmentation")), { n: 4, names: ["T1c", "T1", "T2", "FLAIR"], order: "metadata" });
  assert.deepEqual(channelPlan(e("renalStructures_CECT_segmentation")), { n: 3, names: ["artery", "vein", "excret"], order: "inference" });
  // DeepEdit の 3 チャネルは、画像 1 + 点の信号 2。画像として重ねるのは 1 本なので複数チャネルの計画は要らない
  assert.equal(channelPlan(e("spleen_deepedit_annotation")), null);
  assert.equal(channelPlan(e("spleen_ct_segmentation")), null);
  // 1 チャネルの Bundle に複数シリーズを渡すと止まる／複数チャネルに 1 本だと止まる
  const brats = e("brats_mri_segmentation");
  assert.equal(judge({ network_data_format: { inputs: { image: { modality: "MR", num_channels: 4 } }, outputs: { pred: { format: "segmentation" } } } }, { modality: "MR", channels: 1 }).ok, false);
  assert.equal(classify(brats, "MR").kind, "untested");
  assert.equal(classify(brats, "CT").kind, "stopped");
});

test("loadKeysOf resolves @refs and drops label", () => {
  assert.deepEqual(loadKeysOf({ image_key: "image", preprocessing: { transforms: [{ _target_: "LoadImaged", keys: "@image_key" }] } }), ["image"]);
  assert.deepEqual(loadKeysOf({ p: { transforms: [{ _target_: "LoadImaged", keys: ["artery", "vein", "excret", "label"] }] } }), ["artery", "vein", "excret"]);
  assert.deepEqual(loadKeysOf({}), []);
  assert.deepEqual(loadKeysOf(null), []);
});

test("brats label names follow the post-processing values 1, 2, 4", () => {
  assert.deepEqual(Object.keys(LABEL_NAMES.brats_mri_segmentation), ["1", "2", "4"]);
});

const multiMeta = (n) => ({
  version: "1.0.0",
  network_data_format: {
    inputs: { image: { type: "image", format: "magnitude", modality: "CT", num_channels: n, channel_def: { 0: "a", 1: "b" } } },
    outputs: { pred: { type: "image", format: "segmentation", channel_def: { 0: "background", 1: "spleen" } } },
  },
});

/** 複数チャネルの往復（偽の MONAI）。入力の形は Bundle の設定の LoadImaged のキーで決まる。 */
function runMulti({ layout, nch = 2, metaCh = 2, badGrid = "", modality1 = "" }) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "graphy-monai-"));
  try {
    const lib = path.join(dir, "lib");
    for (const [p, body] of Object.entries(FAKE_MONAI)) {
      fs.mkdirSync(path.dirname(path.join(lib, p)), { recursive: true });
      fs.writeFileSync(path.join(lib, p), body);
    }
    const bundle = path.join(dir, "bundle");
    fs.mkdirSync(path.join(bundle, "configs"), { recursive: true });
    fs.writeFileSync(path.join(bundle, "configs", "metadata.json"), JSON.stringify(multiMeta(metaCh)));
    const cfg = layout === "one-key"
      ? { image_key: "image", test_datalist: "$load()", output_dir: "x", run: ["$@evaluator.run()"], preprocessing: { transforms: [{ _target_: "LoadImaged", keys: "@image_key" }] } }
      : { datalist: [], output_dir: "x", run: ["$@evaluator.run()"], preprocessing: { transforms: [{ _target_: "LoadImaged", keys: ["a", "b", "label"] }] } };
    fs.writeFileSync(path.join(bundle, "configs", "inference.json"), JSON.stringify(cfg));
    const run = path.join(dir, "run");
    fs.mkdirSync(run);
    const env = {
      ...process.env,
      PYTHONPATH: [lib, process.env.GRAPHY_TEST_PYTHONPATH].filter(Boolean).join(path.delimiter),
      FAKE_BUNDLE_SRC: bundle, FAKE_MODE: "same", FAKE_MULTI: "1", WANT_NO_WEIGHTS_ONLY: "1",
      FAKE_NCH: String(nch), FAKE_BAD_GRID: badGrid, FAKE_MODALITY: "CT", FAKE_MODALITY_1: modality1 || "CT",
      HOME: dir, USERPROFILE: dir, PIP_NO_INDEX: "1", PYTHONIOENCODING: "utf-8",
    };
    delete env.COLAB_RELEASE_TAG;
    delete env.TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD;
    const make = spawnSync(PYTHON, ["-c", MAKE_INPUT], { cwd: run, env, encoding: "utf8" });
    assert.equal(make.status, 0, make.stderr);
    const script = buildScript({ name: "fake_bundle" }, {}, nch);
    const r = spawnSync(PYTHON, ["-c", script], { cwd: run, env, encoding: "utf8" });
    const dl = fs.existsSync(path.join(run, "datalist.json")) ? JSON.parse(fs.readFileSync(path.join(run, "datalist.json"), "utf8")) : null;
    return { r, dl, base: path.basename };
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

test("PY multi-channel: one LoadImaged key gets the list of channel files in order", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const { r, dl } = runMulti({ layout: "one-key" });
  assert.equal(r.status, 0, r.stderr);
  assert.equal(dl.length, 1);
  assert.deepEqual(Object.keys(dl[0]), ["image"]);
  assert.deepEqual(dl[0].image.map((p) => path.basename(p)), ["image.nii.gz", "image_1.nii.gz"]);
});

test("PY multi-channel: one key per channel are assigned in the order of the series", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const { r, dl } = runMulti({ layout: "per-key" });
  assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(Object.keys(dl[0]), ["a", "b"]);
  assert.deepEqual(Object.values(dl[0]).map((p) => path.basename(p)), ["image.nii.gz", "image_1.nii.gz"]);
});

test("PY multi-channel stops when the grids differ, the channel count is wrong or the modalities differ", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const grid = runMulti({ layout: "one-key", badGrid: "1" });
  assert.notEqual(grid.r.status, 0);
  assert.match(grid.r.stderr, /grid-mismatch/);
  assert.equal(grid.dl, null);
  const count = runMulti({ layout: "one-key", nch: 3 });
  assert.notEqual(count.r.status, 0);
  assert.match(count.r.stderr, /not-applicable.*チャネル数が 2/);
  assert.equal(count.dl, null);
  const mod = runMulti({ layout: "one-key", modality1: "MR" });
  assert.notEqual(mod.r.status, 0);
  assert.match(mod.r.stderr, /not-applicable.*モダリティ/);
  assert.equal(mod.dl, null);
  // 正例: 同じ入力は通る（上の負例が、検査ではなく別の理由で落ちていないことの裏）
  assert.equal(runMulti({ layout: "one-key" }).r.status, 0);
});

test("channelNameMatches: T1 は T1c に一致しない（語の境界）", async () => {
  const { channelNameMatches } = await import("./ui.js");
  assert.equal(channelNameMatches("BRATS_001_T1", "T1"), true);
  assert.equal(channelNameMatches("BRATS_001_T1c", "T1"), false, "負例: T1 が T1c に一致してはならない");
  assert.equal(channelNameMatches("BRATS_001_T1c", "T1c"), true);
  assert.equal(channelNameMatches("flair axial", "FLAIR"), true);
  assert.equal(channelNameMatches("T2star", "T2"), false);
});

test("parsePoints: 役割の書き方と、壊れた行は行番号つきで止める", () => {
  const ok = parsePoints("# メモ\n10, 20.5, -30, positive\n\n1 2 3 neg\n4,5,6,+\n7,8,9,0");
  assert.equal(ok.ok, true);
  assert.deepEqual(ok.points.map((p) => p.role), ["positive", "negative", "positive", "negative"]);
  assert.deepEqual(ok.points[1].lps, [1, 2, 3]);
  for (const [text, re] of [["1,2,3", /1 行目.*4 項目/], ["1,2,x,positive", /1 行目.*座標/], ["1,2,3,maybe", /1 行目.*役割/], ["1,2,3,positive\n1,2", /2 行目/]]) {
    const r = parsePoints(text);
    assert.equal(r.ok, false, text);
    assert.match(r.error, re);
  }
});

test("promptProblem: vista3d の組み合わせ（複数ラベル＋点、腎臓・肺・骨＋点、陽性なし）を止める", () => {
  const pos = [{ role: "positive" }];
  assert.equal(promptProblem("vista3d", [3], pos), null);
  assert.equal(promptProblem("vista3d", [1, 3], []), null, "点が無ければ複数ラベルでよい");
  assert.equal(promptProblem("vista3d", [], []), null, "全自動");
  assert.match(promptProblem("vista3d", [1, 3], pos), /1 つまで/);
  assert.match(promptProblem("vista3d", [2], pos), /腎臓・肺・骨/);
  assert.match(promptProblem("vista3d", [20], pos), /腎臓・肺・骨/);
  assert.match(promptProblem("vista3d", [], [{ role: "negative" }]), /陽性/);
  assert.match(promptProblem("spleen_deepedit_annotation", [3], []), /ラベル番号を受け取りません/);
  assert.equal(promptProblem("spleen_ct_segmentation", [3], pos), null, "対象外のモデルはここでは止めない（計算機側が止める）");
});

test("buildPrompt / vistaLabelOptions / roiToLps", () => {
  assert.equal(buildPrompt("vista3d", [], []), null);
  assert.equal(buildPrompt("spleen_ct_segmentation", [1], []), null);
  assert.deepEqual(buildPrompt("vista3d", [3], [{ lps: [1, 2, 3], role: "positive" }]), { labels: [3], points: [{ lps: [1, 2, 3], role: "positive" }] });
  const opts = vistaLabelOptions(ZOO.entries.find((e) => e.name === "vista3d"));
  assert.equal(opts.length, 132);
  assert.deepEqual(opts[2], { value: 3, name: "spleen" });
  // index (col,row,slice) → LPS。indexToWorld は row-major 4x4
  const vol = { dims: [4, 4, 4], indexToWorld: [0.7, 0, 0, -100, 0, 0.8, 0, -120, 0, 0, 2.5, 50, 0, 0, 0, 1] };
  // 画素 (3,2) の中心は、左上隅が 0 の座標で (3.5, 2.5)
  assert.deepEqual(roiToLps({ points: [[3.5, 2.5]], sliceIndex: 1 }, vol).map((v) => +v.toFixed(3)), [-97.9, -118.4, 52.5]);
  // 隅の座標をそのまま index として読む（0.5 の取り違え）と、この値にはならない
  assert.notDeepEqual(roiToLps({ points: [[3, 2]], sliceIndex: 1 }, vol).map((v) => +v.toFixed(3)), [-97.9, -118.4, 52.5]);
});

/** ラベル番号・点つきの往復（偽の MONAI）。cfg は Bundle の設定、prompt は画面から来る入力 */
function runPrompt(cfg, prompt, evaluatorSrc = null, extraEnv = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "graphy-monai-"));
  try {
    const lib = path.join(dir, "lib");
    for (const [p, body] of Object.entries(FAKE_MONAI)) {
      fs.mkdirSync(path.dirname(path.join(lib, p)), { recursive: true });
      fs.writeFileSync(path.join(lib, p), body);
    }
    const bundle = path.join(dir, "bundle");
    fs.mkdirSync(path.join(bundle, "configs"), { recursive: true });
    fs.writeFileSync(path.join(bundle, "configs", "metadata.json"), JSON.stringify(spleenMeta));
    fs.writeFileSync(path.join(bundle, "configs", "inference.json"), JSON.stringify(cfg));
    if (evaluatorSrc !== null) {
      fs.mkdirSync(path.join(bundle, "scripts"), { recursive: true });
      fs.writeFileSync(path.join(bundle, "scripts", "evaluator.py"), evaluatorSrc);
    }
    const run = path.join(dir, "run");
    fs.mkdirSync(run);
    const env = {
      ...process.env,
      PYTHONPATH: [lib, process.env.GRAPHY_TEST_PYTHONPATH].filter(Boolean).join(path.delimiter),
      FAKE_BUNDLE_SRC: bundle, FAKE_MODE: "same", WANT_NO_WEIGHTS_ONLY: "1", FAKE_MODALITY: "CT", ...extraEnv,
      HOME: dir, USERPROFILE: dir, PIP_NO_INDEX: "1", PYTHONIOENCODING: "utf-8",
    };
    delete env.COLAB_RELEASE_TAG;
    delete env.TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD;
    const make = spawnSync(PYTHON, ["-c", MAKE_INPUT], { cwd: run, env, encoding: "utf8" });
    assert.equal(make.status, 0, make.stderr);
    const r = spawnSync(PYTHON, ["-c", buildScript({ name: "fake_bundle" }, {}, 1, prompt)], { cwd: run, env, encoding: "utf8" });
    const f = path.join(run, "prompt-kw.json");
    const kw = fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : null;
    const lj = path.join(run, "outputs", "labels.json");
    const ev = path.join(dir, "bundles", "fake_bundle", "scripts", "evaluator.py");
    return { r, kw, summary: fs.existsSync(lj) ? JSON.parse(fs.readFileSync(lj, "utf8")) : null, evaluator: fs.existsSync(ev) ? fs.readFileSync(ev, "utf8") : null, dir };
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

// 目印のボクセル（x=3, y=2, z=1）の患者座標。MAKE_INPUT の格子（間隔 0.7/0.8/2.5、原点 (-100,-120,50)、x 軸は 30°回転）
const lpsOf = (x, y, z) => {
  const c = Math.cos(Math.PI / 6), s = Math.sin(Math.PI / 6);
  return [-100 + x * 0.7 * c - y * 0.8 * s, -120 + x * 0.7 * s + y * 0.8 * c, 50 + z * 2.5];
};
const VISTA_CFG = { input_dict: "${'image': 'x.nii.gz'}", output_dir: "x", run: ["$@evaluator.run()"], preprocessing: { transforms: [{ _target_: "LoadImaged", keys: "image" }] } };
const DEEPEDIT_CFG = { label_names: { spleen: 1, background: 0 }, datalist: [], output_dir: "x", run: ["$@evaluator.run()"], preprocessing: { transforms: [{ _target_: "AddGuidanceFromPointsDeepEditd", ref_image: "image" }] } };

test("PY prompt (vista3d): ラベルと点が input_dict に、点は元の画像の添字で入る", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const { r, kw, summary } = runPrompt(VISTA_CFG, { labels: [3], points: [{ lps: lpsOf(3, 2, 1), role: "positive" }, { lps: lpsOf(7, 5, 4), role: "negative" }] });
  assert.equal(r.status, 0, r.stderr);
  assert.equal(path.basename(kw.input_dict.image), "image.nii.gz");
  assert.deepEqual(kw.input_dict.label_prompt, [3]);
  assert.deepEqual(kw.input_dict.points, [[3, 2, 1], [7, 5, 4]]);
  assert.deepEqual(kw.input_dict.point_labels, [1, 0]);
  assert.deepEqual(summary.prompt.points.map((p) => p.index), [[3, 2, 1], [7, 5, 4]]);
  // 何も渡さない（全自動）ときは画像だけ
  const auto = runPrompt(VISTA_CFG, null);
  assert.equal(auto.r.status, 0, auto.r.stderr);
  assert.deepEqual(Object.keys(auto.kw.input_dict), ["image"]);
});

test("PY prompt (DeepEdit): 陽性は前景の名前、陰性は background に入る", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const { r, kw } = runPrompt(DEEPEDIT_CFG, { labels: [], points: [{ lps: lpsOf(3, 2, 1), role: "positive" }, { lps: lpsOf(7, 5, 4), role: "negative" }] });
  assert.equal(r.status, 0, r.stderr);
  const d = kw["dataset#data"][0];
  assert.equal(path.basename(d.image), "image.nii.gz");
  assert.deepEqual(d.spleen, [[3, 2, 1]]);
  assert.deepEqual(d.background, [[7, 5, 4]]);
  const none = runPrompt(DEEPEDIT_CFG, null);
  assert.equal(none.r.status, 0, none.r.stderr);
  assert.deepEqual(none.kw["dataset#data"][0].spleen, []);
});

test("PY prompt: 画像の外の点、受け取らないモデル、DeepEdit にラベル番号は止める", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const out = runPrompt(VISTA_CFG, { labels: [], points: [{ lps: lpsOf(50, 2, 1), role: "positive" }] });
  assert.notEqual(out.r.status, 0);
  assert.match(out.r.stderr, /prompt-outside/);
  assert.equal(out.kw, null, "推論に進まない");
  const plain = runPrompt({ datalist: [], output_dir: "x", run: ["$@evaluator.run()"] }, { labels: [3], points: [] });
  assert.notEqual(plain.r.status, 0);
  assert.match(plain.r.stderr, /prompt-unsupported/);
  assert.equal(plain.kw, null);
  const lab = runPrompt(DEEPEDIT_CFG, { labels: [1], points: [] });
  assert.notEqual(lab.r.status, 0);
  assert.match(lab.r.stderr, /prompt-unsupported/);
  // 正例: 同じ設定で点が画像の中なら通る（負例が別の理由で落ちていないことの裏）
  assert.equal(runPrompt(VISTA_CFG, { labels: [], points: [{ lps: lpsOf(3, 2, 1), role: "positive" }] }).r.status, 0);
});

test("PY prompt: vista3d の点だけ evaluator の型を揃える（点が無ければ触らない、一致する行が無ければ何もしない）", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const SRC = "x = 1\n                points = torch.from_numpy(points).to(inputs.device)\n";
  const pts = { labels: [3], points: [{ lps: lpsOf(3, 2, 1), role: "positive" }] };
  const patched = runPrompt(VISTA_CFG, pts, SRC);
  assert.equal(patched.r.status, 0, patched.r.stderr);
  assert.equal(patched.summary.bundleCodePatched.length, 1);
  assert.match(patched.summary.bundleCodePatched[0], /from_numpy.*as_tensor/);
  const noPts = runPrompt(VISTA_CFG, { labels: [3], points: [] }, SRC);
  assert.equal(noPts.r.status, 0, noPts.r.stderr);
  assert.deepEqual(noPts.summary.bundleCodePatched, [], "点が無ければ Bundle のコードは変えない");
  const other = runPrompt(VISTA_CFG, pts, "points = something_else(points)\n");
  assert.equal(other.r.status, 0, other.r.stderr);
  assert.deepEqual(other.summary.bundleCodePatched, [], "一致する行が無ければ何もしない");
});

test("PY prompt: 点があるとき 255（点の窓の外）は背景にして数を返す。点が無ければそのまま", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const pts = { labels: [3], points: [{ lps: lpsOf(3, 2, 1), role: "positive" }] };
  const withPts = runPrompt(VISTA_CFG, pts, null, { FAKE_NOT_INFERRED: "1" });
  assert.equal(withPts.r.status, 0, withPts.r.stderr);
  assert.equal(withPts.summary.labels["255"], undefined, "255 は臓器として返さない");
  assert.equal(withPts.summary.notInferredVoxels, 6, "x=0, y=0 の z 方向 6 ボクセル（偽の出力が lab[0, 0, :] に 255 を置く）");
  const noPts = runPrompt(VISTA_CFG, { labels: [3], points: [] }, null, { FAKE_NOT_INFERRED: "1" });
  assert.equal(noPts.r.status, 0, noPts.r.stderr);
  assert.equal(noPts.summary.labels["255"], 6, "点が無いときは触らない");
  assert.equal(noPts.summary.notInferredVoxels, 0);
});

// --- 画面（最小の偽 DOM）と、事前チェック（依存パッケージの案内） ---
class El {
  constructor(tag) { this.tag = tag; this.kids = []; this.style = {}; this.dataset = {}; this.handlers = {}; this.textContent = ""; this._value = undefined; this.disabled = false; }
  append(...c) { this.kids.push(...c); }
  appendChild(c) { this.kids.push(c); return c; }
  replaceChildren(...c) { this.kids = c; }
  addEventListener(n, f) { this.handlers[n] = f; }
  get value() { return this._value !== undefined ? this._value : (this.tag === "select" ? (this.optionList()[0]?.value ?? "") : ""); }
  set value(v) { this._value = v; }
  optionList() { return this.kids.flatMap((k) => (k.tag === "option" ? [k] : k.tag === "optgroup" ? k.kids.filter((o) => o.tag === "option") : [])); }
  get children() { return this.kids.filter((k) => k instanceof El); }
  get options() { return this.optionList(); }
  querySelector(sel) { return sel === "option:not([disabled])" ? this.kids.find((k) => k.tag === "option" && !k.disabled) ?? null : null; }
  querySelectorAll(tag) { return this.kids.flatMap((k) => (k instanceof El ? [...(k.tag === tag ? [k] : []), ...k.querySelectorAll(tag)] : [])); }
  text() { return this.textContent + this.kids.map((k) => (k instanceof El ? k.text() : String(k))).join(""); }
  find(id) { if (this.dataset.testid === id) return this; for (const k of this.kids) { const f = k instanceof El ? k.find(id) : null; if (f) return f; } return null; }
}
const FULL = { torch: "2.4.0", numpy: "1.26.0", monai: "1.4.0", "pytorch-ignite": "0.5.1", nibabel: "5.2.0" };
const envJson = (installed, extra = {}) => ({ colab: false, python: "/usr/bin/python3", pyVersion: "3.11.9", need: Object.keys(FULL), cuda: false, torchError: null, problem: null, installed: { ...FULL, ...installed }, ...extra });
const probeOutcome = (env) => ({ ok: true, status: "ok", readFile: async (n) => (n === "env.json" ? new TextEncoder().encode(JSON.stringify(env)) : null) });
function fakeMonai(mainOutcome, probes = [probeOutcome(envJson({}))]) {
  globalThis.window = {};
  globalThis.document = { createElement: (t) => new El(t) };
  const calls = [], notes = [];
  const container = new El("div");
  const host = {
    getTargets: () => [{ kind: "image", tileId: "t", studyUid: "S", seriesUid: "R", seriesLabel: "CT", modality: "CT", sliceCount: 100 }],
    notify: (m) => notes.push(m),
    openWindow: () => ({ container, setCloseGuard() {}, onClose() {} }),
    compute: {
      status: async () => [],
      // 事前チェック（inputs が空で、一覧の更新でないもの）は probes を順に返す。最後の 1 件は繰り返す。本処理は mainOutcome
      runJob: async (opts) => {
        calls.push(opts);
        if (opts.inputs.length === 0) return probes.length > 1 ? probes.shift() : probes[0];
        return mainOutcome;
      },
    },
  };
  return { host, calls, notes, container };
}
const failedMain = (errorValue) => ({ ok: true, status: "error", errorName: "RuntimeError", errorValue, readFile: async () => null, traceback: "", stderr: "" });

test("installCommand: 計算機が使う Python を名指しし、空白を含むパスは引用符で囲む", () => {
  assert.equal(installCommand("/usr/bin/python3", ["torch", "monai"]), "/usr/bin/python3 -m pip install torch monai");
  assert.equal(installCommand("C:\\Python311\\python.exe", ["nibabel"]), "C:\\Python311\\python.exe -m pip install nibabel");
  assert.equal(installCommand("C:\\Users\\A B\\py\\python.exe", ["nibabel"]), '"C:\\Users\\A B\\py\\python.exe" -m pip install nibabel');
  assert.equal(installCommand("/usr/bin/python3", []), null);
});

test("missingPackages: 本処理の『名前|Python』を読む。別の形は null", () => {
  assert.deepEqual(missingPackages("missing-packages: monai nibabel|/usr/bin/python3"), { names: ["monai", "nibabel"], python: "/usr/bin/python3" });
  assert.deepEqual(missingPackages("missing-packages: itk|C:\\Users\\A B\\python.exe"), { names: ["itk"], python: "C:\\Users\\A B\\python.exe" });
  assert.equal(missingPackages("not-applicable: x"), null);
  assert.equal(missingPackages("missing-packages: monai"), null);
});

test("本処理の Python: Colab 以外で足りなければ pip を呼ばず、『名前|使われる Python』で止まる", { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" }, () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "graphy-monai-"));
  try {
    const env = { ...process.env, PIP_NO_INDEX: "1", PYTHONIOENCODING: "utf-8", HOME: dir, USERPROFILE: dir, PYTHONPATH: path.join(dir, "empty") };
    delete env.COLAB_RELEASE_TAG;
    const r = spawnSync(PYTHON, ["-c", buildScript({ name: "spleen_ct_segmentation" })], { cwd: dir, env, encoding: "utf8" });
    if (r.status !== 0 && /missing-packages/.test(r.stderr)) {
      const m = /RuntimeError: (missing-packages: .+)/.exec(r.stderr);
      const parsed = missingPackages(m[1]);
      assert.ok(parsed && parsed.names.length && parsed.python, m[1]);
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

const probeRun = (versions, { colab = false, fakeTorch = false, extraEnv = {} } = {}) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "monai-probe-"));
  try {
    // 版の表だけを偽装する（試験機に入っているものに左右されない）。判定・書き出しは本物の PY_PROBE
    const script = buildProbeScript().replace("\nNEED = json.loads", `
_VER = json.loads(${JSON.stringify(JSON.stringify(versions))})
importable = lambda m: _VER.get({'ignite': 'pytorch-ignite'}.get(m, m)) is not None
${colab ? "COLAB = True" : "COLAB = False"}
${fakeTorch ? "import types; _t = types.ModuleType('torch'); _t.cuda = types.SimpleNamespace(is_available=lambda: False); sys.modules['torch'] = _t" : ""}
NEED = json.loads`).replace("def version_of(dist):\n    try:\n        return _md.version(dist)", "def version_of(dist):\n    try:\n        return _VER[dist] if _VER.get(dist) else _md.version(dist)");
    const r = spawnSync(PYTHON, ["-c", script], { cwd: dir, env: { ...process.env, ...extraEnv }, encoding: "utf8" });
    assert.equal(r.status, 0, r.stderr);
    return JSON.parse(fs.readFileSync(path.join(dir, "outputs", "env.json"), "utf-8"));
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
};
const skipPy = { skip: !PYTHON && "GRAPHY_TEST_PYTHON is not set" };

test("プローブ: 揃っていれば problem は null。Python の場所と版を返す", skipPy, () => {
  const e = probeRun(FULL, { fakeTorch: true });
  assert.equal(e.problem, null);
  assert.equal(e.colab, false);
  assert.equal(e.cuda, false);
  assert.ok(e.python && e.pyVersion);
});

test("プローブ: 足りないものは problem に名前を並べる（負例）", skipPy, () => {
  const e = probeRun({ ...FULL, monai: null, nibabel: null }, { fakeTorch: true });
  assert.equal(e.problem, "missing-packages: monai nibabel");
  assert.equal(e.installed.monai, null);
});

test("プローブ: Colab は何も確かめず通す（Colab は本処理が自分で入れる）", skipPy, () => {
  const e = probeRun({}, { colab: true });
  assert.equal(e.colab, true);
  assert.equal(e.problem, null);
  assert.deepEqual(e.installed, {});
});

test("プローブ: 入っているが読み込めない torch は broken-package（負例）", skipPy, () => {
  // 偽の torch を import で例外にする
  const e = probeRun(FULL, { extraEnv: {} });
  // この試験機の Python に本物の torch が無ければ import が失敗する。有れば成功して problem は null
  if (e.torchError) assert.match(e.problem, /^broken-package: torch を読み込めません/);
});

test("画面: 不足なら本処理を出さず、環境の枠にコマンドを出す。入れて［再確認］で通り、実行が出る", async () => {
  const missing = probeOutcome(envJson({ monai: null }, { problem: "missing-packages: monai" }));
  const fixed = probeOutcome(envJson({}));
  const { host, calls, container } = fakeMonai(failedMain("not-applicable: テスト"), [missing, fixed]);
  await activate(host);
  await container.find("monai-run").handlers.click();
  assert.equal(calls.length, 1, "本処理は出さない（事前チェックだけ）");
  assert.deepEqual(calls[0].inputs, []);
  assert.equal(calls[0].script, buildProbeScript());
  assert.equal(container.find("monai-env").style.display, "flex");
  assert.equal(container.find("monai-env-command").value, "/usr/bin/python3 -m pip install monai");
  assert.match(container.find("monai-env-python").text(), /\/usr\/bin\/python3.*3\.11\.9/);
  assert.match(container.find("monai-env").text(), /monai.*✗ 入っていません/);
  assert.equal(container.find("monai-run").disabled, false, "入れ直して押し直せる");
  await container.find("monai-env-recheck").handlers.click();
  assert.equal(container.find("monai-env").style.display, "none");
  assert.match(container.find("monai-status").textContent, /環境は整いました/);
  await container.find("monai-run").handlers.click();
  assert.equal(calls.filter((c) => c.inputs.length === 1).length, 1, "整ったので本処理が 1 回出る");
});

test("画面: 一度通れば 2 回目の実行では事前チェックを省く（同意の窓を増やさない）。Colab は枠なしで通る", async () => {
  const colab = probeOutcome(envJson({}, { colab: true, installed: {} }));
  const { host, calls, container } = fakeMonai(failedMain("not-applicable: テスト"), [colab]);
  await activate(host);
  await container.find("monai-run").handlers.click();
  assert.notEqual(container.find("monai-env").style.display, "flex", "Colab では枠を出さない");
  await container.find("monai-run").handlers.click();
  assert.deepEqual(calls.map((c) => c.inputs.length), [0, 1, 1]);
});

test("画面: 本処理が Bundle 固有の missing-packages で止まったら、同じ枠に Python 名指しのコマンドを出す", async () => {
  const { host, calls, container } = fakeMonai(failedMain("missing-packages: itk pynrrd|/opt/py/bin/python3"));
  await activate(host);
  await container.find("monai-run").handlers.click();
  assert.deepEqual(calls.map((c) => c.inputs.length), [0, 1]);
  assert.equal(container.find("monai-env").style.display, "flex");
  assert.equal(container.find("monai-env-command").value, "/opt/py/bin/python3 -m pip install itk pynrrd");
  assert.equal(container.find("monai-env-recheck"), null, "Bundle 固有の不足は事前チェックでは見えないので、再確認ではなく『もう一度実行』");
  assert.match(container.find("monai-env").text(), /もう一度「実行」/);
});

test("画面: 事前チェックが断られたら理由を出し、本処理は出さない", async () => {
  const { host, calls, container } = fakeMonai(failedMain("x"), [{ ok: false, error: "local-unavailable" }]);
  await activate(host);
  await container.find("monai-run").handlers.click();
  assert.equal(calls.length, 1);
  assert.match(container.find("monai-status").textContent, /失敗/);
});
