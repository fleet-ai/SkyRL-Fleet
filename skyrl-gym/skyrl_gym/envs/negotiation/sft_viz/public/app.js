"use strict";

const PAGE = 25;

const state = {
  manifest: null,
  dataset: null,      // "casino" | "dnd"
  data: null,         // current dataset payload
  persp: "all",       // "all" | "you" | "them"
  query: "",
  shown: PAGE,
  cache: {},          // dataset -> payload
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
};

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function highlight(text, q) {
  const safe = escapeHtml(text);
  if (!q) return safe;
  try {
    const re = new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
    return safe.replace(re, "<mark>$1</mark>");
  } catch {
    return safe;
  }
}

async function loadDataset(ds) {
  if (state.cache[ds]) return state.cache[ds];
  const r = await fetch(`data/${ds}.json`);
  const payload = await r.json();
  state.cache[ds] = payload;
  return payload;
}

function filtered() {
  const q = state.query.trim().toLowerCase();
  return state.data.conversations.filter((c) => {
    if (state.persp !== "all" && c.perspective !== state.persp) return false;
    if (!q) return true;
    return c.messages.some((m) => m.content.toLowerCase().includes(q));
  });
}

function renderStats() {
  const s = state.data.stats;
  const grid = $("#statsGrid");
  grid.innerHTML = "";
  const cards = [
    { val: s.n_rows.toLocaleString(), lbl: "training rows (all)" },
    { val: state.data.displayed.toLocaleString(), lbl: "loaded in viewer" },
    { val: s.avg_assistant_msgs, lbl: "avg supervised turns", sup: true },
    { val: s.avg_supervised_words, lbl: "avg supervised words / conv", sup: true },
    { val: s.avg_msgs, lbl: "avg total turns" },
    { val: s.supervised_turn_word_p90, lbl: "p90 words / supervised turn" },
  ];
  const p = s.perspectives || {};
  cards.push({ val: `${p.you || 0}/${p.them || 0}`, lbl: "rows: you / them" });
  for (const c of cards) {
    const card = el("div", "stat");
    card.appendChild(el("div", "val" + (c.sup ? " sup" : ""), String(c.val)));
    card.appendChild(el("div", "lbl", c.lbl));
    grid.appendChild(card);
  }
}

function renderHist() {
  const hist = state.data.stats.assistant_turns_hist || {};
  const wrap = $("#turnsHist");
  wrap.innerHTML = "";
  const entries = Object.entries(hist);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  for (const [k, v] of entries) {
    const bar = el("div", "hbar");
    bar.appendChild(el("div", "bc", v));
    const b = el("div", "bar");
    b.style.height = `${(v / max) * 100}%`;
    bar.appendChild(b);
    bar.appendChild(el("div", "bn", k));
    wrap.appendChild(bar);
  }
}

function makeConv(c) {
  const conv = el("div", "conv collapsed");

  const head = el("div", "conv-head");
  const chev = el("span", "chev", "▾");
  head.appendChild(chev);
  const pp = el("span", "pill persp", `assistant = ${c.perspective}`);
  head.appendChild(pp);
  const asstN = c.messages.filter((m) => m.role === "assistant").length;
  head.appendChild(el("span", "pill", `${asstN} supervised turns`));
  head.appendChild(el("span", "spacer"));
  head.appendChild(el("span", "meta", `game #${c.game_index}`));
  head.addEventListener("click", () => conv.classList.toggle("collapsed"));
  conv.appendChild(head);

  const body = el("div", "conv-body");
  const q = state.query.trim();
  for (const m of c.messages) {
    if (m.role === "system") {
      const sb = el("div", "sys-block collapsed");
      const tag = el("div", "sys-tag", "system prompt ▸ (click to expand)");
      tag.addEventListener("click", (e) => {
        e.stopPropagation();
        sb.classList.toggle("collapsed");
        tag.textContent = sb.classList.contains("collapsed")
          ? "system prompt ▸ (click to expand)"
          : "system prompt ▾";
      });
      sb.appendChild(tag);
      const pre = el("pre");
      pre.textContent = m.content;
      sb.appendChild(pre);
      body.appendChild(sb);
      continue;
    }
    const msg = el("div", `msg ${m.role}`);
    const role = el("div", "role");
    role.appendChild(el("span", null, m.role));
    if (m.supervised) {
      const lt = el("span", "losstag", "LOSS");
      role.appendChild(lt);
    }
    msg.appendChild(role);
    const txt = el("div");
    txt.innerHTML = highlight(m.content, q);
    msg.appendChild(txt);
    body.appendChild(msg);
  }
  conv.appendChild(body);
  return conv;
}

function render() {
  const list = filtered();
  const slice = list.slice(0, state.shown);
  const container = $("#convList");
  container.innerHTML = "";
  if (list.length === 0) {
    container.appendChild(el("div", "empty", "No conversations match this filter."));
  }
  for (const c of slice) container.appendChild(makeConv(c));

  $("#resultCount").textContent =
    `${list.length.toLocaleString()} conversations` +
    (state.data.truncated ? ` (sampled from ${state.data.stats.n_rows.toLocaleString()})` : "");
  const more = $("#loadMore");
  more.style.display = slice.length < list.length ? "block" : "none";
  more.textContent = `Load more (${slice.length} / ${list.length})`;
}

function buildDatasetTabs() {
  const tabs = $("#datasetTabs");
  tabs.innerHTML = "";
  for (const d of state.manifest.datasets) {
    const b = el("button", "tab", d.label);
    b.dataset.ds = d.id;
    if (d.id === state.dataset) b.classList.add("active");
    b.addEventListener("click", async () => {
      if (state.dataset === d.id) return;
      state.dataset = d.id;
      [...tabs.children].forEach((c) => c.classList.toggle("active", c.dataset.ds === d.id));
      await switchDataset();
    });
    tabs.appendChild(b);
  }
}

async function switchDataset() {
  state.data = await loadDataset(state.dataset);
  state.shown = PAGE;
  $("#subtitle").textContent =
    `${state.data.label} — assistant turns are supervised (loss); user/system turns are context only.`;
  renderStats();
  renderHist();
  render();
}

function wireControls() {
  $("#searchBox").addEventListener("input", (e) => {
    state.query = e.target.value;
    state.shown = PAGE;
    render();
  });
  $("#onlySupervised").addEventListener("change", (e) => {
    document.body.classList.toggle("dimctx", e.target.checked);
  });
  $("#loadMore").addEventListener("click", () => {
    state.shown += PAGE;
    render();
  });
  document.querySelectorAll("#perspTabs .tab").forEach((t) => {
    t.addEventListener("click", () => {
      document.querySelectorAll("#perspTabs .tab").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      state.persp = t.dataset.persp;
      state.shown = PAGE;
      render();
    });
  });
}

async function main() {
  state.manifest = await (await fetch("data/manifest.json")).json();
  state.dataset = state.manifest.datasets[0].id;
  buildDatasetTabs();
  wireControls();
  await switchDataset();
}

main().catch((e) => {
  document.querySelector("#convList").innerHTML =
    `<div class="empty">Failed to load data: ${e}</div>`;
});
