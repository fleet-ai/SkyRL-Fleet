"use strict";

const ITEM_EMOJI = {
  book: "\uD83D\uDCDA",
  hat: "\uD83C\uDFA9",
  ball: "\u26BD",
  food: "\uD83C\uDF56",
  water: "\uD83D\uDCA7",
  firewood: "\uD83D\uDD25",
};
const PAGE_SIZE = 20;

const state = {
  mode: "data", // 'data' (human dialogues) | 'eval' (model self-play)
  manifest: null,
  dataset: null, // dataset/run descriptor from manifest
  split: null,
  raw: { split: null, games: [], stats: {} },
  filtered: [],
  page: 0,
  charts: {},
};

const $ = (sel) => document.querySelector(sel);
const emojiFor = (name) => ITEM_EMOJI[name] || "\uD83D\uDCE6";

async function init() {
  $("#modeTabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (btn) setMode(btn.dataset.mode);
  });
  $("#datasetTabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    selectDataset(state.manifest.datasets.find((d) => d.id === btn.dataset.id));
  });
  $("#splitTabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (btn) loadSplit(btn.dataset.split);
  });
  const startMode = location.hash.replace("#", "").startsWith("eval") ? "eval" : "data";
  await setMode(startMode);
}

async function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("#modeTabs .tab").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  setupOutcomeFilter(mode);
  const path = mode === "eval" ? "data/eval/manifest.json" : "data/manifest.json";
  state.manifest = await (await fetch(path)).json();
  renderDatasetTabs();
  let ds = state.manifest.datasets[0];
  if (mode === "data") {
    const wanted = location.hash.replace("#", "");
    ds = state.manifest.datasets.find((d) => d.id === wanted) || ds;
  } else {
    const wanted = location.hash.replace("#eval/", "").replace("#eval", "");
    ds = state.manifest.datasets.find((d) => d.id === wanted) || ds;
  }
  selectDataset(ds);
}

function setupOutcomeFilter(mode) {
  const opts =
    mode === "eval"
      ? [["all", "All outcomes"], ["agreement", "Agreement"], ["conflict", "Conflict (overclaim)"],
         ["no_deal", "No deal (no tag)"], ["incomplete", "Incomplete"],
         ["verbal", "\u26A0 Verbal agree, no tag"]]
      : [["all", "All outcomes"], ["agreed", "Agreement only"], ["disagreed", "No agreement"]];
  $("#outcomeFilter").innerHTML = opts.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
}

function renderDatasetTabs() {
  $("#datasetTabs").innerHTML = state.manifest.datasets
    .map((d) => `<button class="tab" data-id="${d.id}">${d.name}</button>`)
    .join("");
}

function selectDataset(ds) {
  state.dataset = ds;
  if (state.mode === "data" && location.hash.replace("#", "") !== ds.id) {
    history.replaceState(null, "", "#" + ds.id);
  }
  $("#datasetBlurb").textContent = ds.blurb;
  document.querySelectorAll("#datasetTabs .tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.id === ds.id);
  });
  $("#splitTabs").style.display = ds.splits.length > 1 ? "" : "none";
  $("#splitTabs").innerHTML = ds.splits
    .map((s, i) => `<button class="tab ${i === 0 ? "active" : ""}" data-split="${s}">${s}</button>`)
    .join("");
  loadSplit(ds.splits[0]);
}

async function loadSplit(split) {
  state.split = split;
  document.querySelectorAll("#splitTabs .tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.split === split);
  });
  $("#gameList").innerHTML = '<div class="loading">Loading ' + state.dataset.name + "\u2026</div>";
  const url = state.mode === "eval" ? `data/eval/${state.dataset.id}.json` : `data/${state.dataset.id}/${split}.json`;
  state.raw = await (await fetch(url)).json();
  state.page = 0;
  renderStats();
  renderCharts();
  applyFilters();
}

function renderStats() {
  const s = state.raw.stats;
  const scoreMax = s.score_max || 10;
  const cards = [
    { label: "Games", value: s.num_games.toLocaleString() },
    { label: "Agreement rate", value: (s.agreement_rate * 100).toFixed(1) + "%" },
    { label: "Avg turns", value: s.avg_turns },
    { label: "Avg score / agent", value: ((s.avg_you_score + s.avg_them_score) / 2).toFixed(1) + " / " + scoreMax },
    { label: "Avg joint score", value: s.avg_joint_score + " / " + (s.joint_max || 2 * scoreMax) },
    ...(s.extra_cards || []),
  ];
  $("#statsGrid").innerHTML = cards
    .map((c) => `<div class="stat-card"><div class="value">${c.value}</div><div class="label">${c.label}</div></div>`)
    .join("");
}

function destroyCharts() {
  Object.values(state.charts).forEach((c) => c && c.destroy());
  state.charts = {};
}

function renderCharts() {
  destroyCharts();
  const s = state.raw.stats;
  const gridColor = "rgba(255,255,255,0.06)";
  const tickColor = "#8b949e";
  const baseOpts = {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: gridColor }, ticks: { color: tickColor } },
      y: { grid: { color: gridColor }, ticks: { color: tickColor } },
    },
  };

  // Turns histogram (cap long tail for readability).
  const cap = 13;
  const turnsEntries = Object.entries(s.turns_hist)
    .map(([k, v]) => [+k, v])
    .sort((a, b) => a[0] - b[0]);
  const tLabels = [], tData = [];
  let tail = 0;
  turnsEntries.forEach(([k, v]) => {
    if (k <= cap) { tLabels.push(k); tData.push(v); } else { tail += v; }
  });
  if (tail) { tLabels.push(cap + 1 + "+"); tData.push(tail); }
  state.charts.turns = new Chart($("#turnsChart"), {
    type: "bar",
    data: { labels: tLabels, datasets: [{ data: tData, backgroundColor: "#58a6ff", borderRadius: 4 }] },
    options: baseOpts,
  });

  // Score distribution over 0..score_max.
  const scoreMax = s.score_max || 10;
  const sLabels = [], sData = [];
  for (let i = 0; i <= scoreMax; i++) { sLabels.push(i); sData.push(s.score_hist[i] || 0); }
  $("#scoreChartTitle").textContent = `Score distribution (per agent / ${scoreMax}, agreed deals)`;
  state.charts.score = new Chart($("#scoreChart"), {
    type: "bar",
    data: { labels: sLabels, datasets: [{ data: sData, backgroundColor: "#3fb950", borderRadius: 4 }] },
    options: baseOpts,
  });

  // Outcome donut. In eval mode show the full failure-reason breakdown.
  let labels, data, colors;
  if (state.mode === "eval" && s.reason_hist) {
    const order = ["agreement", "conflict", "no_deal", "incomplete"];
    const pretty = { agreement: "Agreement", conflict: "Conflict", no_deal: "No deal", incomplete: "Incomplete" };
    const color = { agreement: "#3fb950", conflict: "#f85149", no_deal: "#d29922", incomplete: "#bb8009" };
    labels = []; data = []; colors = [];
    order.forEach((r) => {
      if (s.reason_hist[r]) { labels.push(pretty[r]); data.push(s.reason_hist[r]); colors.push(color[r]); }
    });
  } else {
    labels = ["Agreement", "No deal"];
    data = [s.num_agreed, s.num_games - s.num_agreed];
    colors = ["#3fb950", "#f85149"];
  }
  state.charts.outcome = new Chart($("#outcomeChart"), {
    type: "doughnut",
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0 }] },
    options: {
      responsive: true,
      cutout: "62%",
      plugins: { legend: { position: "bottom", labels: { color: tickColor, boxWidth: 12 } } },
    },
  });
}

function applyFilters() {
  const q = $("#searchBox").value.trim().toLowerCase();
  const outcome = $("#outcomeFilter").value;
  const sortBy = $("#sortBy").value;

  let games = state.raw.games.filter((g) => {
    if (state.mode === "eval") {
      if (outcome === "verbal" && !(g.flags || []).includes("verbal_agreement_no_tag")) return false;
      if (outcome !== "all" && outcome !== "verbal" && g.reason !== outcome) return false;
    } else {
      if (outcome === "agreed" && !(g.agreed && g.valid_alloc)) return false;
      if (outcome === "disagreed" && g.agreed && g.valid_alloc) return false;
    }
    if (q) {
      const text = g.turns.map((t) => t.text).join(" ").toLowerCase();
      if (!text.includes(q)) return false;
    }
    return true;
  });

  const joint = (g) => (g.you_score || 0) + (g.them_score || 0);
  const gap = (g) => Math.abs((g.you_score || 0) - (g.them_score || 0));
  if (sortBy === "turns_desc") games.sort((a, b) => b.num_turns - a.num_turns);
  else if (sortBy === "turns_asc") games.sort((a, b) => a.num_turns - b.num_turns);
  else if (sortBy === "joint_desc") games.sort((a, b) => joint(b) - joint(a));
  else if (sortBy === "gap_desc") games.sort((a, b) => gap(b) - gap(a));

  state.filtered = games;
  state.page = 0;
  renderList();
}

function renderList() {
  const total = state.filtered.length;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  state.page = Math.min(state.page, pages - 1);
  const start = state.page * PAGE_SIZE;
  const slice = state.filtered.slice(start, start + PAGE_SIZE);

  $("#resultCount").textContent = total.toLocaleString() + " games";

  if (!slice.length) {
    $("#gameList").innerHTML = '<div class="loading">No games match these filters.</div>';
    $("#pager").innerHTML = "";
    return;
  }

  $("#gameList").innerHTML = slice.map(gameCard).join("");
  renderPager(pages);
}

function poolHtml(g) {
  return (
    '<div class="pool">' +
    g.counts
      .map((c, i) => {
        const name = g.item_names[i];
        return `<div class="item"><div class="emoji">${emojiFor(name)}</div><div class="cnt">${c}</div><div class="iname">${name}${c === 1 ? "" : "s"}</div></div>`;
      })
      .join("") +
    "</div>"
  );
}

function valueTable(g) {
  const reasonsYou = g.meta ? g.meta.you.reasons : null;
  const reasonsThem = g.meta ? g.meta.them.reasons : null;
  const cell = (v, reason) =>
    reason ? `<td title="${escapeAttr(reason)}" class="has-reason">${v}</td>` : `<td>${v}</td>`;

  let rows =
    "<tr><th>Value</th>" +
    g.item_names.map((n) => `<th>${emojiFor(n)}</th>`).join("") +
    "<th>Max</th></tr>";
  rows +=
    '<tr><td><span class="dot you"></span>You</td>' +
    g.you_values.map((v, i) => cell(v, reasonsYou && reasonsYou[i])).join("") +
    `<td><b>${g.you_max}</b></td></tr>`;
  rows +=
    '<tr><td><span class="dot them"></span>Them</td>' +
    g.them_values.map((v, i) => cell(v, reasonsThem && reasonsThem[i])).join("") +
    `<td><b>${g.them_max}</b></td></tr>`;
  return `<table class="value-table">${rows}</table>`;
}

function allocText(g, alloc) {
  return g.item_names.map((n, i) => `${alloc[i]}${emojiFor(n)}`).join("  ");
}

const REASON_INFO = {
  agreement: { cls: "ok", label: "\u2713 agreement" },
  conflict: { cls: "bad", label: "\u2717 conflict \u2014 both claimed the same items \u2192 0/0" },
  no_deal: { cls: "warn", label: "\u2717 no deal" },
  incomplete: { cls: "warn", label: "\u2717 incomplete \u2014 items left unassigned \u2192 0/0" },
};

function statusBadge(g) {
  if (!g.reason) return "";
  const info = REASON_INFO[g.reason] || { cls: "warn", label: g.reason };
  let extra = "";
  const flags = g.flags || [];
  if (g.reason === "no_deal") {
    if (flags.includes("verbal_agreement_no_tag"))
      extra = ' <span class="flag">verbally agreed, but never sent a valid &lt;deal&gt;</span>';
    else if (flags.includes("one_sided_tag"))
      extra = ' <span class="flag">only one side sent a &lt;deal&gt;</span>';
    else extra = ' <span class="flag">no parseable &lt;deal&gt; emitted</span>';
  }
  return `<div class="status ${info.cls}">${info.label}${extra}</div>`;
}

function evalNoDealOutcome(g) {
  const yp = g.you_alloc ? allocText(g, g.you_alloc) : '<span class="muted">no &lt;deal&gt; sent</span>';
  const tp = g.them_alloc ? allocText(g, g.them_alloc) : '<span class="muted">no &lt;deal&gt; sent</span>';
  return (
    '<div class="outcome disagreed"><div class="otitle">No agreement \u2014 both score 0</div>' +
    `<div class="alloc-line">You claimed: ${yp}</div>` +
    `<div class="alloc-line">Them claimed: ${tp}</div></div>`
  );
}

function outcomeHtml(g) {
  if (!g.agreed || !g.valid_alloc) {
    if (state.mode === "eval") return evalNoDealOutcome(g);
    return '<div class="outcome disagreed"><div class="otitle">Outcome</div><div><span class="tag">No agreement</span></div></div>';
  }
  const max = g.you_max || 10;
  const youPct = (g.you_score / max) * 100;
  const themPct = (g.them_score / max) * 100;
  let html =
    '<div class="outcome agreed"><div class="otitle">Final allocation</div>' +
    `<div class="alloc-line">You take ${allocText(g, g.you_alloc)} &nbsp;·&nbsp; Them take ${allocText(g, g.them_alloc)}</div>` +
    `<div class="score-row"><span class="who"><span class="dot you"></span>You</span><div class="bar"><span class="you" style="width:${youPct}%"></span></div><span class="num">${g.you_score}/${max}</span></div>` +
    `<div class="score-row"><span class="who"><span class="dot them"></span>Them</span><div class="bar"><span class="them" style="width:${themPct}%"></span></div><span class="num">${g.them_score}/${max}</span></div>`;
  if (g.meta) html += subjectiveHtml(g);
  return html + "</div>";
}

function subjectiveHtml(g) {
  const row = (label, side) => {
    const m = g.meta[side];
    const bits = [];
    if (m.satisfaction) bits.push(`${m.satisfaction}`);
    if (m.likeness) bits.push(`likes: ${m.likeness}`);
    return bits.length ? `<div class="subj"><span class="who2">${label}</span>${bits.join(" \u00B7 ")}</div>` : "";
  };
  return '<div class="subjective">' + row("You", "you") + row("Them", "them") + "</div>";
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function escapeAttr(s) {
  return escapeHtml(s).replace(/'/g, "&#39;");
}

function highlightDeal(escaped) {
  // Highlight a complete <deal>...</deal> tag, or a trailing (truncated) one.
  return escaped
    .replace(/(&lt;deal&gt;[\s\S]*?&lt;\/deal&gt;)/g, '<span class="dealtag">$1</span>')
    .replace(/(&lt;deal&gt;(?:(?!&lt;\/deal&gt;)[\s\S])*$)/, '<span class="dealtag trunc">$1</span>');
}

function dialogueHtml(turns) {
  return (
    '<div class="dialogue">' +
    turns
      .map((t) => {
        const ann =
          t.annotations && t.annotations.length
            ? `<div class="anns">${t.annotations.map((a) => `<span class="ann">${a}</span>`).join("")}</div>`
            : "";
        const body = highlightDeal(escapeHtml(t.text));
        return `<div class="msg ${t.speaker}"><div class="speaker">${t.speaker === "you" ? "You" : "Them"}</div>${body}${ann}</div>`;
      })
      .join("") +
    "</div>"
  );
}

function gameCard(g) {
  const reasonCls = g.reason ? ` game-${(REASON_INFO[g.reason] || {}).cls || "warn"}` : "";
  return (
    `<div class="game${reasonCls}">` +
    '<div class="game-side">' +
    poolHtml(g) +
    valueTable(g) +
    outcomeHtml(g) +
    "</div>" +
    '<div class="game-side">' +
    statusBadge(g) +
    `<div class="meta-line">${g.num_turns} turns · ${g.first_speaker === "you" ? "You" : "Them"} opened</div>` +
    dialogueHtml(g.turns) +
    "</div>" +
    "</div>"
  );
}

function renderPager(pages) {
  const p = state.page;
  $("#pager").innerHTML =
    `<button ${p === 0 ? "disabled" : ""} data-go="first">\u00AB</button>` +
    `<button ${p === 0 ? "disabled" : ""} data-go="prev">\u2039 Prev</button>` +
    `<span class="pinfo">Page ${p + 1} of ${pages}</span>` +
    `<button ${p >= pages - 1 ? "disabled" : ""} data-go="next">Next \u203A</button>` +
    `<button ${p >= pages - 1 ? "disabled" : ""} data-go="last">\u00BB</button>`;
  $("#pager").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      const go = b.dataset.go;
      if (go === "first") state.page = 0;
      else if (go === "prev") state.page--;
      else if (go === "next") state.page++;
      else if (go === "last") state.page = pages - 1;
      renderList();
      window.scrollTo({ top: $("#gameList").offsetTop - 20, behavior: "smooth" });
    };
  });
}

let searchTimer;
$("#searchBox").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(applyFilters, 200);
});
$("#outcomeFilter").addEventListener("change", applyFilters);
$("#sortBy").addEventListener("change", applyFilters);

init();
