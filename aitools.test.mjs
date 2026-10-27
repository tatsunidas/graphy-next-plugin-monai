// node --test aitools.test.mjs
// AI アシスタント向けのツール（segment_models / segment_run / segment_undo）。偽の host で、計算機・ビューアの代わりをする。
import assert from "node:assert/strict";
import { test } from "node:test";

import { ZOO, AI_MODELS, ORGAN_ALIASES, resolveOrgans, pickModel, keepLabels, volumeMl, segmentModels, segmentRun, segmentUndo, aiTools } from "./ui.js";

const entry = (name) => ZOO.entries.find((e) => e.name === name);
const names = (r) => r.labels.map((l) => l.name);

test("resolveOrgans: 日本語・英語・左右。1 語が複数ラベルに当たるとき（肺葉）は全部", () => {
  const whole = entry("wholeBody_ct_segmentation");
  assert.deepEqual(names(resolveOrgans(whole, "脾臓")), ["spleen"]);
  assert.deepEqual(names(resolveOrgans(whole, "spleen, liver")), ["spleen", "liver"]);
  assert.deepEqual(names(resolveOrgans(whole, "右腎臓")), ["kidney_right"]);
  assert.deepEqual(names(resolveOrgans(whole, "left kidney")), ["kidney_left"]);
  assert.deepEqual(names(resolveOrgans(whole, "腎臓")), ["kidney_right", "kidney_left"]);
  assert.equal(resolveOrgans(whole, "肺").labels.length, 5);
  assert.deepEqual(names(resolveOrgans(whole, "右肺")), ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]);
});

test("resolveOrgans: 負例 — 未知・左右のない臓器に左右・空・多すぎる・別のモデルにしかない臓器", () => {
  const whole = entry("wholeBody_ct_segmentation");
  const bad = resolveOrgans(whole, "脾臓, ほげ");
  assert.equal(bad.ok, false);
  assert.equal(bad.error, "organ-not-found");
  assert.deepEqual(bad.missing, ["ほげ"]);
  assert.equal(resolveOrgans(whole, "左脾臓").ok, false);
  assert.equal(resolveOrgans(whole, "").error, "no-organs");
  assert.equal(resolveOrgans(whole, "a1,a2,a3,a4,a5,a6,a7,a8,a9").error, "too-many-organs");
  assert.equal(resolveOrgans(entry("spleen_ct_segmentation"), "肝臓").ok, false);
  assert.equal(resolveOrgans(whole, "ab").ok, false, "2 文字の英字は部分一致に使わない");
});

test("resolveOrgans: swin の channel_def の綴り（Right Kidney / Left Kideny / IVC）にも当たる", () => {
  const swin = entry("swin_unetr_btcv_segmentation");
  assert.deepEqual(names(resolveOrgans(swin, "腎臓")), ["Right Kidney", "Left Kideny"]);
  assert.deepEqual(names(resolveOrgans(swin, "左腎臓")), ["Left Kideny"]);
  assert.deepEqual(names(resolveOrgans(swin, "下大静脈")), ["IVC"]);
});

test("pickModel: 全部の臓器を含むモデルのうちラベル数が最少。指定があればそれだけ", () => {
  const E = ZOO.entries;
  assert.equal(pickModel(E, "脾臓").entry.name, "spleen_ct_segmentation");
  assert.equal(pickModel(E, "脾臓, 右腎臓").entry.name, "swin_unetr_btcv_segmentation");
  assert.equal(pickModel(E, "脾臓, 肺").entry.name, "wholeBody_ct_segmentation");
  assert.equal(pickModel(E, "脾臓", "wholeBody_ct_segmentation").entry.name, "wholeBody_ct_segmentation");
});

test("pickModel: 負例 — 未知のモデル・どのモデルにも無い臓器（候補つき）・モデル指定で足りない臓器", () => {
  const E = ZOO.entries;
  assert.equal(pickModel(E, "脾臓", "no_such_model").error, "unknown-model");
  const nowhere = pickModel(E, "脾臓, ほげ");
  assert.equal(nowhere.error, "organ-not-found");
  assert.deepEqual(nowhere.missing, ["ほげ"]);
  assert.ok(nowhere.available.includes("肝臓"));
  assert.equal(pickModel(E, "肝臓", "spleen_ct_segmentation").error, "organ-not-found");
});

test("pickModel: どのモデルでも当たる語だが 1 つでは足りない組は no-single-model", () => {
  // 膵臓（pancreas モデル・全身・BTCV・multi）と肺（全身だけ）は全身で足りる。足りないのは、別々のモデルにしか無い語の組
  const only = [entry("pancreas_ct_dints_segmentation"), entry("spleen_ct_segmentation")];
  const r = pickModel(only, "膵臓, 脾臓");
  assert.equal(r.error, "no-single-model");
});

test("別名表の語幹は、すべて少なくとも 1 つのモデルのラベルに当たる（死んだ別名を残さない）", () => {
  const E = AI_MODELS.map(entry);
  const dead = Object.keys(ORGAN_ALIASES).filter((k) => !E.some((e) => resolveOrgans(e, k).ok));
  // 脳・心臓・膀胱などは一覧にないモデル（全身 104 臓器など）で当たるもの以外は、ここに挙がる
  assert.deepEqual(dead.filter((k) => ["肝臓", "脾臓", "腎臓", "膵臓", "胆嚢", "胃", "大動脈", "肺"].includes(k)), []);
});

test("keepLabels / volumeMl: 指定したラベルだけが残り、体積は解析値と一致", () => {
  const data = new Uint8Array([0, 1, 1, 2, 3, 3, 3, 0]);
  const { data: out, counts } = keepLabels(data, [1, 3]);
  assert.deepEqual([...out], [0, 1, 1, 0, 3, 3, 3, 0]);
  assert.deepEqual(counts, { 1: 2, 3: 3 });
  assert.equal(volumeMl(1000, [1, 1, 1]), 1);
  assert.equal(volumeMl(10, [10, 10, 10]), 10);
  assert.ok(Math.abs(volumeMl(123456, [0.7, 0.7, 2.5]) - (123456 * 0.7 * 0.7 * 2.5) / 1000) < 1e-9);
});

test("segmentModels: 使えるモデルと臓器名。研究用の注記つき", () => {
  const r = segmentModels();
  assert.equal(r.ok, true);
  assert.equal(r.models.length, AI_MODELS.length);
  assert.ok(r.models.every((m) => m.modality === "CT" && typeof m.organs === "string" && m.organs.length > 0), "臓器名は文字列（結果の深さ上限のため）");
  assert.ok(r.models.find((m) => m.model === "wholeBody_ct_segmentation").organs.includes("spleen"));
  assert.match(r.note, /診断には使わない/);
});

// ---- segment_run の全体（偽の host） ----
const NX = 4, NY = 3, NZ = 5;
const GEOM = { spacing: [10, 10, 10], origin: [0, 0, 0], direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] };
function npy(bytes, shape) {
  let header = `{'descr': '|u1', 'fortran_order': False, 'shape': (${shape.join(", ")}), }`;
  const total = 10 + header.length + 1;
  header += " ".repeat((64 - (total % 64)) % 64) + "\n";
  const out = new Uint8Array(10 + header.length + bytes.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(out.buffer).setUint16(8, header.length, true);
  out.set(new TextEncoder().encode(header), 10);
  out.set(bytes, 10 + header.length);
  return out;
}
/** ラベル 1 を 10 ボクセル、2 を 6、3 を 4 置いた [z, y, x] の配列。 */
function labelBytes() {
  const a = new Uint8Array(NX * NY * NZ);
  let i = 0;
  for (const [v, n] of [[1, 10], [2, 6], [3, 4]]) for (let c = 0; c < n; c++) a[i++] = v;
  return a;
}
function fakeHost(over = {}) {
  const calls = { run: [], show: [], removed: [] };
  const host = {
    getTargets: () => [{ kind: "image", modality: "CT", studyUid: "s", seriesUid: "r", tileId: "t1" }],
    compute: {
      runJob: async (opts, jo) => {
        calls.run.push(opts);
        if (over.runJob) return over.runJob(opts, jo);
        const files = { "labels.json": new TextEncoder().encode(JSON.stringify({ geometry: GEOM, labels: { 1: 10, 2: 6, 3: 4 } })), "labels.npy": npy(labelBytes(), [NZ, NY, NX]) };
        return { ok: true, status: "ok", readFile: async (n) => files[n] ?? null };
      },
    },
    loadVolume: async () => over.vol ?? { dims: [NX, NY, NZ], spacing: [10, 10, 10], ipp: [0, 0, 0], sliceStep: [0, 0, 10], worldToIndex: [0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 1] },
    showLabelVolume: async (tile, req) => { calls.show.push({ tile, req }); return over.show ? over.show : { ok: true, segmentationId: `seg-${calls.show.length}`, segmentCount: req.table.length }; },
    removeLabelVolume: (id) => { calls.removed.push(id); return true; },
  };
  if (over.noRemove) delete host.removeLabelVolume;
  return { host, calls };
}

test("segment_run: 正常系 — 指定ラベルだけが表示され、体積が解析値と一致し、同意は本体の runJob に任せる", async () => {
  const { host, calls } = fakeHost();
  // wholeBody のラベル: 1 spleen, 2 kidney_right, 3 kidney_left。脾臓と右腎臓だけ頼む（3 は計算機が返しても出さない）
  const r = await segmentRun(host, { organs: "脾臓, 右腎臓", model: "wholeBody_ct_segmentation" });
  assert.equal(r.ok, true, JSON.stringify(r));
  assert.deepEqual(r.organs, [{ name: "spleen", voxels: 10, volumeMl: 10 }, { name: "kidney_right", voxels: 6, volumeMl: 6 }]);
  assert.match(r.note, /診断には使わない/);
  assert.equal(calls.run.length, 1, "計算機へのジョブは 1 回（環境確認の別ジョブを送らない）");
  assert.equal(calls.run[0].inputs[0].seriesUid, "r");
  assert.equal(calls.show.length, 1);
  const shown = calls.show[0].req.data;
  assert.equal(shown.filter((v) => v === 3).length, 0, "頼んでいないラベル（左腎臓）は表示しない");
  assert.equal(shown.filter((v) => v === 1).length, 10);
  assert.deepEqual(calls.show[0].req.table.map((t) => t.value), [1, 2]);
  assert.equal(calls.show[0].tile, "t1");
});

test("segment_run: 負例 — 取り消し・環境不足・計算失敗・結果なし・格子不一致・CT 以外・画像なし・未知の臓器は、表示も送信もしない", async () => {
  const ng = async (over, args = { organs: "脾臓" }, hostOver) => {
    const { host, calls } = fakeHost({ ...over });
    if (hostOver) hostOver(host);
    const r = await segmentRun(host, args);
    assert.equal(r.ok, false, JSON.stringify(r));
    if (!over.show) assert.equal(calls.show.length, 0, "何も表示しない");
    return { r, calls };
  };
  assert.equal((await ng({ runJob: async () => ({ ok: false, cancelled: true }) })).r.error, "declined");
  const env = (await ng({ runJob: async () => ({ ok: true, status: "error", errorName: "RuntimeError", errorValue: "missing-packages: torch monai|/usr/bin/python3", readFile: async () => null }) })).r;
  assert.equal(env.error, "environment-missing");
  assert.deepEqual(env.packages, ["torch", "monai"]);
  assert.equal((await ng({ runJob: async () => ({ ok: true, status: "error", errorName: "X", errorValue: "boom", readFile: async () => null }) })).r.error, "compute-failed");
  assert.equal((await ng({ runJob: async () => ({ ok: true, status: "ok", readFile: async () => null }) })).r.error, "no-result");
  assert.equal((await ng({ vol: { dims: [9, 9, 9], spacing: [1, 1, 1], ipp: [0, 0, 0], sliceStep: [0, 0, 1], worldToIndex: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] } })).r.error, "grid-mismatch");
  const mr = await ng({}, { organs: "脾臓" }, (h) => { h.getTargets = () => [{ kind: "image", modality: "MR", studyUid: "s", seriesUid: "r", tileId: "t1" }]; });
  assert.equal(mr.r.error, "model-not-applicable");
  assert.equal(mr.calls.run.length, 0, "CT 以外は送る前に止める");
  const none = await ng({}, { organs: "脾臓" }, (h) => { h.getTargets = () => []; });
  assert.equal(none.r.error, "no-image");
  const unk = await ng({}, { organs: "ほげ" });
  assert.equal(unk.r.error, "organ-not-found");
  assert.equal(unk.calls.run.length, 0, "未知の臓器名は計算機へ送る前に拒否する");
  assert.equal((await ng({ show: { ok: false, error: "viewport is not ready" } })).r.error, "show-failed");
});

test("segment_run: 頼んだ臓器が結果に無ければ nothing-found（空の Mask を出さない）", async () => {
  const empty = new Uint8Array(NX * NY * NZ).fill(0);
  const files = { "labels.json": new TextEncoder().encode(JSON.stringify({ geometry: GEOM, labels: {} })), "labels.npy": npy(empty, [NZ, NY, NX]) };
  const { host, calls } = fakeHost({ runJob: async () => ({ ok: true, status: "ok", readFile: async (n) => files[n] }) });
  const r = await segmentRun(host, { organs: "脾臓" });
  assert.equal(r.error, "nothing-found");
  assert.equal(calls.show.length, 0);
});

test("segment_undo: 出した Mask だけを消す。host が未対応なら unsupported-host。二度目は何も消さない", async () => {
  const { host, calls } = fakeHost();
  const r1 = await segmentRun(host, { organs: "脾臓", model: "wholeBody_ct_segmentation" });
  assert.equal(r1.ok, true);
  const before = calls.show.length;
  const u = segmentUndo(host);
  assert.equal(u.ok, true);
  assert.ok(u.removed >= 1);
  assert.equal(calls.removed.length, u.removed);
  assert.equal(segmentUndo(host).removed, 0, "消し済みを重ねて消さない");
  assert.equal(calls.show.length, before);
  const { host: old } = fakeHost({ noRemove: true });
  assert.deepEqual(segmentUndo(old), { ok: false, error: "unsupported-host" });
});

test("aiTools: plugin.json の宣言と同じ名前で、引数つきで呼べる", async () => {
  assert.deepEqual(Object.keys(aiTools).sort(), ["segment_models", "segment_run", "segment_undo"]);
  assert.equal((await aiTools.segment_models({}, {})).ok, true);
});

test("plugin.json の skills[]: 自分の aiTools だけを指し、入力は定義済みで、ask より前に使わない", async () => {
  const { readFile } = await import("node:fs/promises");
  const manifest = JSON.parse(await readFile(new URL("./plugin.json", import.meta.url), "utf8"));
  const declared = new Set(manifest.aiTools.map((t) => t.name));
  assert.ok(manifest.permissions.includes("ai-tools"));
  assert.deepEqual(manifest.skills.map((s) => s.id), ["organ-volume", "organ-undo"]);
  for (const s of manifest.skills) {
    assert.ok(s.title.ja && s.title.en && s.description.ja && s.description.en, s.id);
    const keys = new Set(s.inputs.map((i) => i.key));
    let asked = new Set();
    for (const st of s.steps) {
      if (st.kind === "ask") {
        assert.ok(keys.has(st.input), `${s.id}: ask of an undefined input`);
        asked.add(st.input);
      } else if (st.kind === "tool") {
        assert.ok(st.id.startsWith("plugin:") && declared.has(st.id.slice(7)), `${s.id}: ${st.id} is not one of this plugin's tools`);
        for (const v of Object.values(st.params)) if (v && typeof v === "object") assert.ok(asked.has(v.$input) || (keys.has(v.$input) && !s.steps.some((x) => x.kind === "ask" && x.input === v.$input)), `${s.id}: input used before it is asked`);
      } else assert.equal(st.kind, "confirm");
    }
  }
});

test("plugin.json の skills[]: 書き込みのツール（segment_run / segment_undo）を含むスキルは確認の手順を持つ", async () => {
  const { readFile } = await import("node:fs/promises");
  const manifest = JSON.parse(await readFile(new URL("./plugin.json", import.meta.url), "utf8"));
  for (const s of manifest.skills) assert.ok(s.steps.some((st) => st.kind === "confirm"), s.id);
});
