/* SkyRL Trace Viewer
 * Visualizes per-step training trajectories produced by `dump_training_trajectories`
 * ({export_path}/dumped_trajectories/global_step_{N}.jsonl). Pure client-side: loads
 * either a served manifest.json or files dropped/picked in the browser.
 */
"use strict";

const PAGE_SIZE = 25;
const KNOWN_TAGS = ["think", "propose", "accept", "deal", "reject", "offer", "answer"];

const state = {
  runs: [],          // [{name, source:'manifest'|'memory', steps:[{step, file?, count, trajectories?}], loaded}]
  run: null,         // active run object (with trajectories filled in)
  stepIdx: 0,        // index into run.steps
  view: "step",      // 'step' | 'track'
  page: 0,
  charts: {},
  promptGroups: null, // Map(key -> {prompt, steps: Map(step -> trajs[])}) for track view
};

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

/* ------------------------------------------------------------------ utils */
function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function mean(xs) { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0; }
function fmt(x, d = 3) { return Number.isFinite(x) ? x.toFixed(d) : "–"; }
function djb2(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) h = ((h << 5) + h + str.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}
function rewardColor(r) {
  if (!Number.isFinite(r)) return { fg: "var(--muted)", bg: "var(--panel-2)" };
  if (r > 0.05) return { fg: "var(--good)", bg: "rgba(63,185,80,.12)" };
  if (r < -0.05) return { fg: "var(--bad)", bg: "rgba(248,81,73,.12)" };
  return { fg: "var(--warn)", bg: "rgba(210,153,34,.12)" };
}

/* ------------------------------------------------------------------ parsing */
// Split a chat-template-formatted string into role-tagged messages. Handles
// Qwen/ChatML `<|im_start|>role ... <|im_end|>`; falls back to a single message.
function splitChat(textRaw) {
  const text = String(textRaw ?? "");
  const msgs = [];
  const re = /<\|im_start\|>([a-zA-Z0-9_]+)\s*([\s\S]*?)(?:<\|im_end\|>|$)/g;
  let m, found = false;
  while ((m = re.exec(text)) !== null) {
    found = true;
    msgs.push({ role: m[1].trim(), text: m[2].trim() });
  }
  if (found) return msgs;
  // Llama-style header fallback
  const re2 = /<\|start_header_id\|>([a-zA-Z0-9_]+)<\|end_header_id\|>\s*([\s\S]*?)(?=<\|start_header_id\|>|<\|eot_id\|>|$)/g;
  while ((m = re2.exec(text)) !== null) {
    found = true;
    msgs.push({ role: m[1].trim(), text: m[2].replace(/<\|eot_id\|>/g, "").trim() });
  }
  if (found) return msgs;
  return null;
}

function roleClass(role) {
  const r = (role || "").toLowerCase();
  if (r === "assistant") return "you";
  if (r === "user" || r === "tool") return "them";
  return "system";
}

// Escape then re-highlight <think> blocks and known XML-ish action tags.
function highlightInline(textRaw) {
  let s = esc(textRaw);
  s = s.replace(/&lt;think&gt;([\s\S]*?)&lt;\/think&gt;/g,
    (_, inner) => `<span class="think">&lt;think&gt;${inner}&lt;/think&gt;</span>`);
  const tagRe = new RegExp(`(&lt;\\/?(?:${KNOWN_TAGS.join("|")})(?:[^&]*?)?&gt;)`, "g");
  s = s.replace(tagRe, '<span class="tag">$1</span>');
  return s;
}

function renderConversation(prompt, text) {
  // Prefer parsing the response `text`; prepend the prompt as context if it carries
  // its own role markers, else show it as a single system/context message.
  let msgs = splitChat(text);
  const promptMsgs = splitChat(prompt);
  if (!msgs) {
    msgs = [];
    if (promptMsgs) msgs.push(...promptMsgs);
    else if (prompt) msgs.push({ role: "system", text: prompt });
    msgs.push({ role: "assistant", text: text });
  } else if (promptMsgs) {
    msgs = [...promptMsgs, ...msgs];
  } else if (prompt) {
    msgs = [{ role: "system", text: prompt }, ...msgs];
  }
  return msgs.filter((mm) => (mm.text || "").trim().length).map((mm) => {
    const cls = roleClass(mm.role);
    return `<div class="msg ${cls}"><div class="role">${esc(mm.role)}</div>` +
      `<div class="text">${highlightInline(mm.text)}</div></div>`;
  }).join("");
}

/* ------------------------------------------------------- matched-arm diff */
// A trajectory may carry `pair: {a_label, b_label, a, b}` -- the SAME scenario
// replayed under two conditions (hole vs nohole in the hole atlas). The claim
// those experiments rest on is that everything the model sees BEFORE it acts is
// byte-identical across the pair, and that the two transcripts diverge only
// after a decision. That is a claim about text, so it gets shown as text rather
// than asserted in a caption.

// Line-level LCS. Transcripts are a few hundred lines, so the O(n*m) table is
// cheap and gives a stable alignment -- a greedy scan would mis-pair a block
// that shifted by one line and report the whole tail as changed.
function lcsRows(a, b) {
  const n = a.length, m = b.length;
  const T = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      T[i][j] = a[i] === b[j] ? T[i + 1][j + 1] + 1 : Math.max(T[i + 1][j], T[i][j + 1]);
  const rows = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { rows.push({ l: a[i], r: b[j], t: "same" }); i++; j++; }
    else if (T[i + 1][j] >= T[i][j + 1]) { rows.push({ l: a[i], r: null, t: "del" }); i++; }
    else { rows.push({ l: null, r: b[j], t: "add" }); j++; }
  }
  while (i < n) rows.push({ l: a[i++], r: null, t: "del" });
  while (j < m) rows.push({ l: null, r: b[j++], t: "add" });
  // Pair adjacent del/add runs into change rows, so a modified line shows its
  // two versions on one row instead of as a delete far above an insert.
  const out = [];
  for (let k = 0; k < rows.length; k++) {
    if (rows[k].t === "del") {
      const dels = []; while (k < rows.length && rows[k].t === "del") dels.push(rows[k++].l);
      const adds = []; while (k < rows.length && rows[k].t === "add") adds.push(rows[k++].r);
      k--;
      const len = Math.max(dels.length, adds.length);
      for (let q = 0; q < len; q++) {
        const l = dels[q] ?? null, r = adds[q] ?? null;
        out.push({ l, r, t: l !== null && r !== null ? "change" : (l !== null ? "del" : "add") });
      }
    } else out.push(rows[k]);
  }
  return out;
}

// Word-level highlight inside a changed pair -- this is what makes "exactly
// what differs" legible when a line changes by one number.
function wordMark(l, r) {
  const A = String(l).split(/(\s+)/), B = String(r).split(/(\s+)/);
  let p = 0; while (p < A.length && p < B.length && A[p] === B[p]) p++;
  let s = 0; while (s < A.length - p && s < B.length - p &&
    A[A.length - 1 - s] === B[B.length - 1 - s]) s++;
  const wrap = (arr) => esc(arr.slice(0, p).join("")) +
    `<mark>${esc(arr.slice(p, arr.length - s).join(""))}</mark>` +
    esc(arr.slice(arr.length - s).join(""));
  return [wrap(A), wrap(B)];
}

function renderArmDiff(pair) {
  if (!pair || (pair.a == null && pair.b == null)) return "";
  const a = String(pair.a ?? "").split("\n");
  const b = String(pair.b ?? "").split("\n");
  const rows = lcsRows(a, b);
  const nDiff = rows.filter((r) => r.t !== "same").length;
  const firstIdx = rows.findIndex((r) => r.t !== "same");
  const head =
    `<div class="diffhead">` +
    `<span class="badge">${esc(pair.a_label || "A")}</span>` +
    `<span class="badge">${esc(pair.b_label || "B")}</span>` +
    (nDiff === 0
      ? `<span class="diffok">identical — 0 differing lines</span>`
      : `<span class="diffcount">${nDiff} differing line${nDiff === 1 ? "" : "s"}` +
        `, first at line ${firstIdx + 1}</span>`) +
    (pair.note ? `<span class="muted">${esc(pair.note)}</span>` : "") +
    `</div>`;
  const body = rows.map((r, k) => {
    const first = k === firstIdx && r.t !== "same" ? " first" : "";
    if (r.t === "same")
      return `<div class="drow same"><div class="dl">${esc(r.l)}</div>` +
        `<div class="dr">${esc(r.r)}</div></div>`;
    if (r.t === "change") {
      const [lh, rh] = wordMark(r.l, r.r);
      return `<div class="drow change${first}"><div class="dl">${lh}</div>` +
        `<div class="dr">${rh}</div></div>`;
    }
    if (r.t === "del")
      return `<div class="drow del${first}"><div class="dl">${esc(r.l)}</div>` +
        `<div class="dr empty"></div></div>`;
    return `<div class="drow add${first}"><div class="dl empty"></div>` +
      `<div class="dr">${esc(r.r)}</div></div>`;
  }).join("");
  return head + `<div class="diffgrid">${body}</div>`;
}

function renderRaw(prompt, text) {
  const hi = (s) => esc(s)
    .replace(/(&lt;\|[^&]*?\|&gt;)/g, '<span class="ctrl">$1</span>')
    .replace(/&lt;think&gt;([\s\S]*?)&lt;\/think&gt;/g, '<span class="think">&lt;think&gt;$1&lt;/think&gt;</span>')
    .replace(new RegExp(`(&lt;\\/?(?:${KNOWN_TAGS.join("|")})(?:[^&]*?)?&gt;)`, "g"), '<span class="tag">$1</span>');
  return `<div class="raw"><span class="ctrl">### PROMPT ###\n</span>${hi(prompt)}` +
    `<span class="ctrl">\n\n### RESPONSE ###\n</span>${hi(text)}</div>`;
}

/* ------------------------------------------------------------------ loading */
async function init() {
  wireEvents();
  await loadManifest();
}

async function loadManifest() {
  try {
    const res = await fetch("data/manifest.json", { cache: "no-store" });
    if (!res.ok) throw new Error("no manifest");
    const manifest = await res.json();
    const runs = (manifest.runs || []).map((r) => ({
      name: r.name,
      source: "manifest",
      loaded: false,
      steps: (r.steps || []).map((s) => ({ step: s.step, file: s.file, count: s.count })),
    }));
    if (!runs.length) throw new Error("empty manifest");
    state.runs = runs;
    populateRunSelect();
    await selectRun(0);
  } catch (e) {
    $("#autoloadNote").textContent =
      "No served manifest found — drop files or use the buttons above to begin.";
  }
}

async function selectRun(idx) {
  const run = state.runs[idx];
  if (!run) return;
  if (!run.loaded && run.source === "manifest") {
    setBusy(`Loading ${run.steps.length} steps…`);
    for (const st of run.steps) {
      const res = await fetch("data/" + st.file, { cache: "no-store" });
      st.trajectories = parseJsonl(await res.text());
    }
    run.loaded = true;
  }
  run.steps.sort((a, b) => a.step - b.step);
  state.run = run;
  state.stepIdx = run.steps.length - 1; // default to latest step
  state.promptGroups = buildPromptGroups(run);
  $("#runSelect").value = String(idx);
  showViewer();
  renderEvolution();
  renderFilterOptions();
  setupScrubber();
  renderCurrentStep();
  renderTrackPicker();
}

function parseJsonl(txt) {
  const out = [];
  for (const line of txt.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    try { out.push(JSON.parse(t)); } catch (_) { /* skip malformed */ }
  }
  return out;
}

// Build a run from in-memory File objects (drag-drop / picker).
function ingestFiles(files, runName) {
  const byStep = new Map();
  let pending = files.length;
  if (!pending) return;
  files.forEach((file) => {
    const reader = new FileReader();
    reader.onload = () => {
      const trajs = parseJsonl(reader.result);
      const m = /global_step_(\d+)/.exec(file.name);
      // Prefer the record's own `step`; fall back to filename.
      trajs.forEach((tr) => {
        const step = Number.isFinite(tr.step) ? tr.step : (m ? parseInt(m[1], 10) : 0);
        if (!byStep.has(step)) byStep.set(step, []);
        byStep.get(step).push(tr);
      });
      if (--pending === 0) finalizeIngest(byStep, runName);
    };
    reader.onerror = () => { if (--pending === 0) finalizeIngest(byStep, runName); };
    reader.readAsText(file);
  });
}

function finalizeIngest(byStep, runName) {
  const steps = Array.from(byStep.entries())
    .map(([step, trajectories]) => ({ step, trajectories, count: trajectories.length }))
    .sort((a, b) => a.step - b.step);
  if (!steps.length) { setBusy("No trajectories found in those files."); return; }
  const run = { name: runName, source: "memory", loaded: true, steps };
  // Replace any same-named in-memory run, else append.
  const existing = state.runs.findIndex((r) => r.name === runName);
  if (existing >= 0) state.runs[existing] = run; else state.runs.push(run);
  populateRunSelect();
  selectRun(state.runs.indexOf(run));
}

/* ------------------------------------------------------------------ run UI */
function populateRunSelect() {
  $("#runSelect").innerHTML = state.runs
    .map((r, i) => `<option value="${i}">${esc(r.name)} · ${r.steps.length} steps</option>`)
    .join("");
}
function showViewer() { $("#dropZone").hidden = true; $("#viewer").hidden = false; }
function setBusy(msg) {
  $("#dropZone").hidden = false; $("#viewer").hidden = true;
  $("#autoloadNote").textContent = msg;
}

/* ------------------------------------------------------------------ evolution */
function stepAgg(st) {
  const trs = st.trajectories;
  const rewards = trs.map((t) => Number(t.reward)).filter(Number.isFinite);
  const turns = trs.map((t) => Number(t.turns)).filter(Number.isFinite);
  const tokens = trs.map((t) => Number(t.tokens)).filter(Number.isFinite);
  const stops = {};
  trs.forEach((t) => { const k = t.stop_reason || "unknown"; stops[k] = (stops[k] || 0) + 1; });
  return {
    step: st.step, n: trs.length,
    meanReward: mean(rewards), meanTurns: mean(turns), meanTokens: mean(tokens),
    posFrac: rewards.length ? rewards.filter((r) => r > 0.05).length / rewards.length : 0,
    stops,
  };
}

function renderEvolution() {
  const aggs = state.run.steps.map(stepAgg);
  const labels = aggs.map((a) => a.step);
  $("#evoMeta").textContent =
    `${state.run.name} · ${aggs.length} steps · ${aggs.reduce((s, a) => s + a.n, 0)} trajectories`;

  destroyCharts();
  const grid = "rgba(255,255,255,.06)";
  const baseOpts = (extra = {}) => ({
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    onClick: (_e, els, chart) => {
      const pts = chart.getElementsAtEventForMode(_e, "index", { intersect: false }, false);
      if (pts.length) selectStepByIndex(pts[0].index);
    },
    plugins: { legend: { labels: { color: "#8b949e", boxWidth: 12, font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: "#8b949e", font: { size: 10 } }, grid: { color: grid } },
      y: { ticks: { color: "#8b949e", font: { size: 10 } }, grid: { color: grid } },
      ...extra,
    },
  });

  state.charts.reward = new Chart($("#rewardChart"), {
    type: "line",
    data: { labels, datasets: [{
      label: "mean reward", data: aggs.map((a) => a.meanReward),
      borderColor: "#4f9dff", backgroundColor: "rgba(79,157,255,.15)", fill: true,
      tension: .25, pointRadius: 3, pointHoverRadius: 5,
    }] },
    options: baseOpts(),
  });

  state.charts.len = new Chart($("#lenChart"), {
    type: "line",
    data: { labels, datasets: [
      { label: "turns", data: aggs.map((a) => a.meanTurns), borderColor: "#3fb950",
        backgroundColor: "transparent", yAxisID: "y", tension: .25, pointRadius: 2 },
      { label: "tokens", data: aggs.map((a) => a.meanTokens), borderColor: "#d29922",
        backgroundColor: "transparent", yAxisID: "y1", tension: .25, pointRadius: 2 },
    ] },
    options: baseOpts({
      y: { position: "left", ticks: { color: "#3fb950", font: { size: 10 } }, grid: { color: grid } },
      y1: { position: "right", ticks: { color: "#d29922", font: { size: 10 } }, grid: { display: false } },
    }),
  });

  const stopKeys = Array.from(new Set(aggs.flatMap((a) => Object.keys(a.stops))));
  const palette = ["#4f9dff", "#3fb950", "#d29922", "#f85149", "#a371f7", "#8b949e", "#db61a2"];
  state.charts.stop = new Chart($("#stopChart"), {
    type: "bar",
    data: { labels, datasets: stopKeys.map((k, i) => ({
      label: k, data: aggs.map((a) => (a.stops[k] || 0) / a.n),
      backgroundColor: palette[i % palette.length],
    })) },
    options: { ...baseOpts({
      x: { stacked: true, ticks: { color: "#8b949e", font: { size: 10 } }, grid: { color: grid } },
      y: { stacked: true, max: 1, ticks: { color: "#8b949e", font: { size: 10 } }, grid: { color: grid } },
    }) },
  });
}
function destroyCharts() {
  Object.values(state.charts).forEach((c) => c && c.destroy());
  state.charts = {};
}

/* ------------------------------------------------------------------ scrubber */
function setupScrubber() {
  const sl = $("#stepSlider");
  sl.min = 0; sl.max = state.run.steps.length - 1; sl.value = state.stepIdx;
}
function selectStepByIndex(i) {
  state.stepIdx = Math.max(0, Math.min(state.run.steps.length - 1, i));
  $("#stepSlider").value = state.stepIdx;
  state.page = 0;
  renderCurrentStep();
}

function renderCurrentStep() {
  const st = state.run.steps[state.stepIdx];
  $("#stepLabel").textContent = `step ${st.step} · ${st.trajectories.length} trajectories`;
  renderStatsGrid(stepAgg(st));
  if (state.view === "step") renderList();
}

function renderStatsGrid(a) {
  const rc = rewardColor(a.meanReward);
  const cells = [
    { k: "mean reward", v: fmt(a.meanReward), cls: a.meanReward > 0.05 ? "good" : (a.meanReward < -0.05 ? "bad" : "") },
    { k: "positive-reward %", v: (a.posFrac * 100).toFixed(0) + "%" },
    { k: "mean turns", v: fmt(a.meanTurns, 1) },
    { k: "mean tokens", v: fmt(a.meanTokens, 0) },
    { k: "trajectories", v: String(a.n) },
  ];
  $("#statsGrid").innerHTML = cells
    .map((c) => `<div class="stat"><div class="v ${c.cls || ""}">${c.v}</div><div class="k">${c.k}</div></div>`)
    .join("");
}

/* ------------------------------------------------------------------ filters */
/* `env_key` is "<env>:<consequence>[:<opponent>][:seat1]". Splitting it into
   two dropdowns keeps a roster of 8 games x 3 opponents from becoming one
   24-entry list where you cannot ask "ipd, all opponents" or "grim, all
   games". Pages whose key has no opponent segment are unaffected: `envOf`
   returns the leading env:consequence, `oppOf` returns null, and the opponent
   dropdown shows only "All opponents". */
const SEAT_RE = /^seat\d+$/;
function envOf(t) {
  const k = t.env_key || "unknown";
  const p = k.split(":");
  return p.length >= 2 ? p.slice(0, 2).join(":") : k;
}
function oppOf(t) {
  const p = (t.env_key || "").split(":");
  const rest = p.slice(2).filter((x) => x && !SEAT_RE.test(x));
  return rest.length ? rest.join(":") : null;
}

function renderFilterOptions() {
  const all = state.run.steps.flatMap((s) => s.trajectories);
  const envs = Array.from(new Set(all.map(envOf))).sort();
  const opps = Array.from(new Set(all.map(oppOf).filter(Boolean))).sort();
  const stops = Array.from(new Set(all.map((t) => t.stop_reason || "unknown"))).sort();
  $("#envFilter").innerHTML = `<option value="all">All envs</option>` +
    envs.map((e) => `<option value="${esc(e)}">${esc(e)}</option>`).join("");
  const opp = $("#oppFilter");
  // ALWAYS VISIBLE, even with nothing to offer. Hiding it on pages that
  // predate opponent-split keys made the control look like it was never added
  // -- an absent dropdown and an unimplemented feature are indistinguishable
  // to whoever is looking. Disabled-and-labelled says which one it is.
  if (opps.length) {
    opp.innerHTML = `<option value="all">All opponents (${opps.length})</option>` +
      opps.map((e) => `<option value="${esc(e)}">${esc(e)}</option>`).join("");
    opp.disabled = false;
    opp.title = "Filter by opponent population member";
  } else {
    opp.innerHTML = `<option value="all">No opponent split on this page</option>`;
    opp.disabled = true;
    opp.title = "This page's env_key carries no opponent segment. Re-sample it "
      + "with traces_over_training.py to split by opponent.";
  }
}

function currentFiltered() {
  const st = state.run.steps[state.stepIdx];
  const q = $("#searchBox").value.trim().toLowerCase();
  const envF = $("#envFilter").value;
  const oppF = $("#oppFilter") ? $("#oppFilter").value : "all";
  const stopF = $("#stopFilter").value;
  const minR = parseFloat($("#minReward").value);
  let rows = st.trajectories.map((t, i) => ({ t, i }));
  if (envF !== "all") rows = rows.filter((r) => envOf(r.t) === envF);
  if (oppF !== "all") rows = rows.filter((r) => oppOf(r.t) === oppF);
  if (stopF !== "all") rows = rows.filter((r) => (r.t.stop_reason || "unknown") === stopF);
  if (Number.isFinite(minR)) rows = rows.filter((r) => Number(r.t.reward) >= minR);
  if (q) rows = rows.filter((r) =>
    ((r.t.text || "") + " " + (r.t.prompt || "")).toLowerCase().includes(q));
  const sort = $("#sortBy").value;
  const num = (x) => (Number.isFinite(Number(x)) ? Number(x) : -Infinity);
  const cmp = {
    reward_desc: (a, b) => num(b.t.reward) - num(a.t.reward),
    reward_asc: (a, b) => num(a.t.reward) - num(b.t.reward),
    tokens_desc: (a, b) => num(b.t.tokens) - num(a.t.tokens),
    turns_desc: (a, b) => num(b.t.turns) - num(a.t.turns),
    default: (a, b) => a.i - b.i,
  }[sort] || ((a, b) => a.i - b.i);
  rows.sort(cmp);
  return rows;
}

function renderList() {
  const rows = currentFiltered();
  $("#resultCount").textContent = `${rows.length} shown`;
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  if (state.page >= pages) state.page = 0;
  const slice = rows.slice(state.page * PAGE_SIZE, state.page * PAGE_SIZE + PAGE_SIZE);
  $("#trajList").innerHTML = slice.map((r) => trajCard(r.t, r.i)).join("");
  renderPager(pages);
}

function trajCard(t, idx) {
  const rc = rewardColor(Number(t.reward));
  const preview = (t.text || t.prompt || "").replace(/\s+/g, " ").slice(0, 160);
  return `<div class="traj" data-idx="${idx}">
    <div class="traj-head" data-toggle>
      <span class="idx">#${idx}</span>
      <span class="reward-pill" style="color:${rc.fg};background:${rc.bg}">${fmt(Number(t.reward), 3)}</span>
      <span class="badge">${esc(t.env_key || "env")}</span>
      <span class="badge stop">${esc(t.stop_reason || "—")}</span>
      <span class="badge">${esc(t.turns ?? "?")}t</span>
      <span class="badge">${esc(t.tokens ?? "?")} tok</span>
      <span class="preview">${esc(preview)}</span>
      <span class="spacer"></span>
      <span class="chev">›</span>
    </div>
    <div class="traj-body">
      <div class="tcol-head">
        <span class="seg active" data-seg="convo">Conversation</span>
        <span class="seg" data-seg="raw">Raw</span>
        ${t.pair ? `<span class="seg" data-seg="diff">Arm diff</span>` : ""}
      </div>
      <div class="convo" data-pane="convo">${renderConversation(t.prompt || "", t.text || "")}</div>
      <div data-pane="raw" hidden>${renderRaw(t.prompt || "", t.text || "")}</div>
      ${t.pair ? `<div class="diffpane" data-pane="diff" hidden>${renderArmDiff(t.pair)}</div>` : ""}
    </div>
  </div>`;
}

function renderPager(pages) {
  if (pages <= 1) { $("#pager").innerHTML = ""; return; }
  let html = "";
  for (let p = 0; p < pages; p++) {
    if (pages > 12 && p > 1 && p < pages - 2 && Math.abs(p - state.page) > 1) {
      if (p === 2 || p === pages - 3) html += `<span class="muted">…</span>`;
      continue;
    }
    html += `<button class="btn ${p === state.page ? "active" : ""}" data-page="${p}">${p + 1}</button>`;
  }
  $("#pager").innerHTML = html;
}

/* ------------------------------------------------------------------ track view */
function buildPromptGroups(run) {
  const groups = new Map();
  run.steps.forEach((st) => {
    st.trajectories.forEach((t) => {
      const key = djb2((t.prompt || "").replace(/\s+/g, " ").trim());
      if (!groups.has(key)) groups.set(key, { key, prompt: t.prompt || "", steps: new Map() });
      const g = groups.get(key);
      if (!g.steps.has(st.step)) g.steps.set(st.step, []);
      g.steps.get(st.step).push(t);
    });
  });
  return groups;
}

function renderTrackPicker() {
  // Only groups that recur across >=2 steps are useful for "evolution".
  const groups = Array.from(state.promptGroups.values())
    .map((g) => ({ ...g, coverage: g.steps.size }))
    .filter((g) => g.coverage >= 2)
    .sort((a, b) => b.coverage - a.coverage);
  const sel = $("#promptSelect");
  if (!groups.length) {
    sel.innerHTML = `<option value="">(no prompt appears in 2+ steps)</option>`;
    $("#trackStrip").innerHTML =
      `<p class="muted">Track view needs the same prompt to appear across multiple steps ` +
      `(as with a fixed train/val set). This run doesn't have recurring prompts.</p>`;
    state._trackGroups = [];
    return;
  }
  state._trackGroups = groups;
  sel.innerHTML = groups.map((g, i) => {
    const label = (g.prompt || "").replace(/\s+/g, " ").slice(0, 80);
    return `<option value="${i}">[${g.coverage} steps] ${esc(label)}…</option>`;
  }).join("");
  sel.value = "0";
  renderTrack(0);
}

function renderTrack(gi) {
  const g = state._trackGroups[gi];
  if (!g) return;
  $("#trackMeta").textContent = `appears in ${g.steps.size} steps`;
  const steps = Array.from(g.steps.keys()).sort((a, b) => a - b);
  const promptMsgs = splitChat(g.prompt);
  const promptText = promptMsgs ? promptMsgs.map((m) => `[${m.role}] ${m.text}`).join("\n\n") : g.prompt;
  const cards = steps.map((step) => {
    const trs = g.steps.get(step).slice().sort((a, b) => Number(b.reward) - Number(a.reward));
    const r = mean(trs.map((t) => Number(t.reward)).filter(Number.isFinite));
    const rc = rewardColor(r);
    const bodies = trs.map((t) => {
      const rc2 = rewardColor(Number(t.reward));
      return `<div class="msg you"><div class="role">` +
        `<span class="reward-pill" style="color:${rc2.fg};background:${rc2.bg}">${fmt(Number(t.reward), 3)}</span> ` +
        `${esc(t.stop_reason || "")} · ${esc(t.turns ?? "?")}t</div>` +
        `<div class="text">${highlightInline(responseOnly(t.text || ""))}</div></div>`;
    }).join("");
    return `<div class="track-card">
      <div class="tc-head"><span class="tc-step">step ${step}</span>
        <span class="reward-pill" style="color:${rc.fg};background:${rc.bg}">μ ${fmt(r, 3)}</span>
        <span class="muted">${trs.length} samples</span></div>
      <div class="tc-body">${bodies}</div>
    </div>`;
  }).join("");
  $("#trackStrip").innerHTML =
    `<div class="track-prompt" style="flex-basis:100%"><div class="label">Prompt</div>` +
    `<div class="text">${esc(promptText)}</div></div>` + cards;
}

// For the track strip we only want the model's generated turns, not the templated prompt.
function responseOnly(text) {
  const msgs = splitChat(text);
  if (!msgs) return text;
  const asst = msgs.filter((m) => roleClass(m.role) === "you");
  return (asst.length ? asst : msgs).map((m) => m.text).join("\n\n");
}

/* ------------------------------------------------------------------ events */
function wireEvents() {
  $("#runSelect").addEventListener("change", (e) => selectRun(parseInt(e.target.value, 10)));

  $("#fileInput").addEventListener("change", (e) =>
    ingestFiles(Array.from(e.target.files), "Loaded files"));
  $("#dirInput").addEventListener("change", (e) => {
    const files = Array.from(e.target.files).filter((f) => /\.jsonl?$/.test(f.name));
    const root = files[0]?.webkitRelativePath?.split("/")[0] || "Loaded folder";
    ingestFiles(files, root);
  });

  // drag & drop
  const dz = document.body;
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault(); $("#dropZone").classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault(); if (ev === "drop" || e.target === $("#dropZone")) $("#dropZone").classList.remove("drag");
  }));
  dz.addEventListener("drop", (e) => {
    const files = Array.from(e.dataTransfer.files).filter((f) => /\.jsonl?$/.test(f.name));
    if (files.length) ingestFiles(files, "Dropped files");
  });

  // scrubber
  $("#stepSlider").addEventListener("input", (e) => selectStepByIndex(parseInt(e.target.value, 10)));
  $("#prevStep").addEventListener("click", () => selectStepByIndex(state.stepIdx - 1));
  $("#nextStep").addEventListener("click", () => selectStepByIndex(state.stepIdx + 1));

  // view tabs
  $("#viewTabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab"); if (!btn) return;
    $$("#viewTabs .tab").forEach((b) => b.classList.toggle("active", b === btn));
    state.view = btn.dataset.view;
    $("#stepView").hidden = state.view !== "step";
    $("#trackView").hidden = state.view !== "track";
    if (state.view === "step") renderList();
  });

  // filters
  ["searchBox", "envFilter", "oppFilter", "stopFilter", "sortBy", "minReward"].forEach((id) =>
    $("#" + id).addEventListener("input", () => { state.page = 0; renderList(); }));

  // list interactions (delegated)
  $("#trajList").addEventListener("click", (e) => {
    const seg = e.target.closest(".seg");
    if (seg) {
      const body = seg.closest(".traj-body");
      body.querySelectorAll(".seg").forEach((s) => s.classList.toggle("active", s === seg));
      // Generalised over panes: the diff pane is only present on paired rows, so
      // hardcoding the two original names would leave it stuck open or missing.
      body.querySelectorAll("[data-pane]").forEach((p) => {
        p.hidden = p.dataset.pane !== seg.dataset.seg;
      });
      return;
    }
    const head = e.target.closest("[data-toggle]");
    if (head) head.closest(".traj").classList.toggle("open");
  });
  $("#pager").addEventListener("click", (e) => {
    const b = e.target.closest("[data-page]"); if (!b) return;
    state.page = parseInt(b.dataset.page, 10); renderList();
    window.scrollTo({ top: $("#stepView").offsetTop - 80, behavior: "smooth" });
  });

  // track picker
  $("#promptSelect").addEventListener("change", (e) => renderTrack(parseInt(e.target.value, 10)));
}

init();
