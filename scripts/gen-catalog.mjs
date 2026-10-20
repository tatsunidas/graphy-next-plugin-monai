// Model Zoo から ui.js の一覧（ZOO）を作り直す:  node scripts/gen-catalog.mjs
// 取得先: Project-MONAI/model-zoo の dev ブランチ（models/model_info.json と各 Bundle の configs/）。
// 取得できなかった値は null のまま埋め込む（0 や空で埋めない）。
import fs from "node:fs";
import { zooEntry } from "../ui.js";

const RAW = "https://raw.githubusercontent.com/Project-MONAI/model-zoo/dev/models";
const getJson = async (url) => {
  const r = await fetch(url);
  if (!r.ok) return null;
  try { return await r.json(); } catch { return null; }
};

const info = await getJson(`${RAW}/model_info.json`);
if (!info) throw new Error("model_info.json を取得できませんでした");
const byName = new Map();
for (const key of Object.keys(info)) {
  const m = /^(.*)_v(\d+\.\d+\.\d+)$/.exec(key);
  if (!m) throw new Error("model_info.json の名前の形式が想定外です: " + key);
  byName.set(m[1], [...(byName.get(m[1]) ?? []), m[2]]);
}
const names = [...byName.keys()].sort();
const entries = await Promise.all(names.map(async (n) => {
  const [meta, inf] = await Promise.all([getJson(`${RAW}/${n}/configs/metadata.json`), getJson(`${RAW}/${n}/configs/inference.json`)]);
  return zooEntry(n, byName.get(n), meta, inf);
}));
const zoo = { fetched: new Date().toISOString().slice(0, 10), source: "https://github.com/Project-MONAI/model-zoo (dev) models/model_info.json", entries };

const file = new URL("../ui.js", import.meta.url);
const src = fs.readFileSync(file, "utf-8");
const begin = src.indexOf("// <catalog:begin>\n");
const end = src.indexOf("// <catalog:end>");
if (begin < 0 || end < 0) throw new Error("ui.js に <catalog:begin>/<catalog:end> がありません");
const body = `// <catalog:begin>\nexport const ZOO = ${JSON.stringify(zoo)};\n`;
fs.writeFileSync(file, src.slice(0, begin) + body + src.slice(end));
console.log(`${entries.length} Bundle（${entries.filter((e) => !e.ndf).length} 本は metadata を取得できず null）`);
