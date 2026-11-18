const apiForm = document.getElementById("api-config-form");
const apiInput = document.getElementById("api-base-url");
const splitBadge = document.getElementById("split-badge");
const serverModelInfo = document.getElementById("server-model-info");
const statusRegion = document.getElementById("status-region");
const liveAnnouncer = document.getElementById("live-announcer");
const loadingProgress = document.getElementById("loading-progress");
const tripListEl = document.getElementById("trip-list");
const activeTripBadge = document.getElementById("active-trip-badge");
const metricsTbody = document.getElementById("metrics-tbody");
const arcSequenceOutput = document.getElementById("arc-sequence-output");

const toggleGps = document.getElementById("toggle-gps");
const toggleNet = document.getElementById("toggle-network");
const toggleCls = document.getElementById("toggle-cls");
const toggleHmm = document.getElementById("toggle-hmm");

const leafletHost = document.getElementById("leaflet-map");
const fallbackCanvas = document.getElementById("fallback-canvas");

let leafletMap = null;
let layers = {
  gps: null,
  network: null,
  cls: null,
  hmm: null,
};
let lastPayload = null;
let selectedTripId = null;

function getApiBase() {
  return apiInput.value.trim().replace(/\/+$/, "");
}

function setStatus(message, isBusy = false) {
  statusRegion.textContent = message;
  loadingProgress.hidden = !isBusy;
  if (!isBusy && liveAnnouncer) {
    liveAnnouncer.textContent = message;
  }
}

function fmtPct(val) {
  if (val === null || val === undefined || Number.isNaN(val)) return "N/A";
  return `${(val * 100).toFixed(2)}%`;
}

function fmtNum(val, digits = 4) {
  if (val === null || val === undefined || Number.isNaN(val)) return "N/A";
  return Number(val).toFixed(digits);
}

function initLeafletIfAvailable() {
  if (typeof window.L === "undefined") {
    leafletHost.hidden = true;
    fallbackCanvas.hidden = false;
    return false;
  }
  if (!leafletMap) {
    leafletMap = window.L.map(leafletHost).setView([20.0, 0.0], 2);
    window.L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 19,
    }).addTo(leafletMap);
    layers.network = window.L.layerGroup().addTo(leafletMap);
    layers.cls = window.L.layerGroup();
    layers.hmm = window.L.layerGroup().addTo(leafletMap);
    layers.gps = window.L.layerGroup().addTo(leafletMap);
  }
  return true;
}

function syncLayerVisibility() {
  if (leafletMap) {
    const mapPairs = [
      [toggleGps.checked, layers.gps],
      [toggleNet.checked, layers.network],
      [toggleCls.checked, layers.cls],
      [toggleHmm.checked, layers.hmm],
    ];
    for (const [visible, group] of mapPairs) {
      if (!group) continue;
      if (visible && !leafletMap.hasLayer(group)) group.addTo(leafletMap);
      if (!visible && leafletMap.hasLayer(group)) leafletMap.removeLayer(group);
    }
  } else if (lastPayload) {
    renderFallbackCanvas(lastPayload);
  }
}

function renderFallbackCanvas(data) {
  const ctx = fallbackCanvas.getContext("2d");
  const rect = fallbackCanvas.getBoundingClientRect();
  fallbackCanvas.width = Math.max(300, Math.floor(rect.width));
  fallbackCanvas.height = Math.max(240, Math.floor(rect.height));

  ctx.clearRect(0, 0, fallbackCanvas.width, fallbackCanvas.height);
  const pts = data.gps_points || [];
  if (!pts.length) return;

  let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity;
  for (const [la, lo] of pts) {
    if (la < minLat) minLat = la;
    if (la > maxLat) maxLat = la;
    if (lo < minLon) minLon = lo;
    if (lo > maxLon) maxLon = lo;
  }
  const pad = 24;
  const w = fallbackCanvas.width - 2 * pad;
  const h = fallbackCanvas.height - 2 * pad;
  const spanLon = Math.max(1e-5, maxLon - minLon);
  const spanLat = Math.max(1e-5, maxLat - minLat);
  const proj = (la, lo) => [
    pad + ((lo - minLon) / spanLon) * w,
    pad + ((maxLat - la) / spanLat) * h,
  ];

  const drawSegs = (segs, color, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    for (const [a, b] of segs) {
      const [x1, y1] = proj(a[0], a[1]);
      const [x2, y2] = proj(b[0], b[1]);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();
  };

  if (toggleNet.checked) drawSegs(data.local_network_segments || [], "#94a3b8", 1);
  if (toggleCls.checked) drawSegs(data.cls_route_segments || [], "#f59e0b", 2.5);
  if (toggleHmm.checked) drawSegs(data.hmm_route_segments || [], "#dc2626", 3);
  if (toggleGps.checked) {
    ctx.fillStyle = "#2563eb";
    for (const [la, lo] of pts) {
      const [x, y] = proj(la, lo);
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function renderMap(data) {
  lastPayload = data;
  if (!initLeafletIfAvailable()) {
    renderFallbackCanvas(data);
    return;
  }

  layers.network.clearLayers();
  layers.cls.clearLayers();
  layers.hmm.clearLayers();
  layers.gps.clearLayers();

  if (data.local_network_segments?.length) {
    window.L.polyline(data.local_network_segments, {
      color: "#94a3b8",
      weight: 1.5,
      opacity: 0.65,
    }).addTo(layers.network);
  }

  if (data.cls_route_segments?.length) {
    window.L.polyline(data.cls_route_segments, {
      color: "#f59e0b",
      weight: 4,
      opacity: 0.8,
    }).addTo(layers.cls);
  }

  if (data.hmm_route_segments?.length) {
    window.L.polyline(data.hmm_route_segments, {
      color: "#dc2626",
      weight: 3.5,
      opacity: 0.95,
    }).addTo(layers.hmm);
  }

  const pts = data.gps_points || [];
  const stride = Math.max(1, Math.floor(pts.length / 800));
  for (let i = 0; i < pts.length; i += stride) {
    window.L.circleMarker(pts[i], {
      radius: 2.6,
      color: "#2563eb",
      fillColor: "#2563eb",
      fillOpacity: 0.85,
      weight: 0,
    }).addTo(layers.gps);
  }

  syncLayerVisibility();
  if (pts.length) {
    const bounds = window.L.latLngBounds(pts);
    leafletMap.fitBounds(bounds, { padding: [24, 24] });
  }
}

function addComparisonRow(metricLabel, clsText, hmmText, noteText) {
  const tr = document.createElement("tr");
  const th = document.createElement("th");
  th.scope = "row";
  th.textContent = metricLabel;

  const tdCls = document.createElement("td");
  tdCls.textContent = clsText;
  const tdHmm = document.createElement("td");
  tdHmm.textContent = hmmText;
  const tdNote = document.createElement("td");
  tdNote.textContent = noteText;

  tr.append(th, tdCls, tdHmm, tdNote);
  metricsTbody.appendChild(tr);
}

function renderMetrics(data) {
  const ev = data.evaluation;
  const cont = data.continuity;
  const t = data.timings_ms;

  document.getElementById("kpi-hmm-top1").textContent = ev ? fmtPct(ev.hmm_obs_top1_acc) : "N/A";
  document.getElementById("kpi-hmm-seqf1").textContent = ev ? fmtPct(ev.hmm_gapfilled_seq_lcs_f1) : "N/A";
  document.getElementById("kpi-hmm-cont").textContent = fmtPct(cont.hmm_obs_continuity);
  document.getElementById("kpi-miss-rate").textContent = ev ? fmtPct(ev.candidate_miss_rate) : "N/A";
  document.getElementById("kpi-latency").textContent = `${t.total_ms.toFixed(0)} ms`;

  metricsTbody.replaceChildren();
  addComparisonRow(
    "Observation-Level Top-1 Accuracy",
    ev ? fmtPct(ev.cls_obs_top1_acc) : "No ref",
    ev ? fmtPct(ev.hmm_obs_top1_acc) : "No ref",
    "Server-side match against aligned reference route"
  );
  addComparisonRow(
    "Observation Continuity",
    fmtPct(cont.cls_obs_continuity),
    fmtPct(cont.hmm_obs_continuity),
    "Consecutive predictions staying on same or connected arc"
  );
  addComparisonRow(
    "Disjoint Route Jumps",
    String(cont.cls_disjoint_jumps),
    String(cont.hmm_disjoint_jumps),
    "Disconnected transitions in collapsed path"
  );
  addComparisonRow(
    "Sequence LCS F1 (vs .route)",
    ev ? fmtPct(ev.cls_seq_lcs_f1) : "No ref",
    ev ? fmtPct(ev.hmm_gapfilled_seq_lcs_f1) : "No ref",
    "Order-preserving F1 against ground-truth arc sequence"
  );
  addComparisonRow(
    "Arc-Set Jaccard (vs .route)",
    "N/A",
    ev ? fmtPct(ev.hmm_arc_set_jaccard) : "No ref",
    "Unique predicted arc overlap with server .route"
  );
  addComparisonRow(
    "Candidate Classifier Macro F1 / Log Loss",
    ev ? `${fmtNum(ev.cls_candidate_macro_f1, 4)} / ${fmtNum(ev.cls_candidate_log_loss, 4)}` : "No ref",
    "Used as HMM emissions",
    "Binary candidate match quality"
  );
  addComparisonRow(
    "Server Pipeline Timing (ms)",
    `ML: ${t.model_ms} ms`,
    `Total: ${t.total_ms} ms (HMM: ${t.viterbi_ms} ms)`,
    `Cand: ${t.candidates_ms} ms | Feat: ${t.features_ms} ms`
  );

  const arcs = data.hmm_predicted_arc_ids || [];
  const codeEl = document.createElement("code");
  codeEl.textContent = `Total arcs (${arcs.length}): [${arcs.join(", ")}]`;
  arcSequenceOutput.replaceChildren(codeEl);
}

async function runPrediction(tripId) {
  selectedTripId = tripId;
  for (const btn of tripListEl.querySelectorAll("button.trip-btn")) {
    btn.setAttribute("aria-pressed", btn.dataset.tripId === tripId ? "true" : "false");
  }
  activeTripBadge.textContent = `Running ${tripId}...`;
  setStatus(`Sending trip ${tripId} to Laptop 1 API...`, true);

  try {
    const resp = await fetch(`${getApiBase()}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tripId }),
    });
    if (!resp.ok) {
      throw new Error(`API returned HTTP ${resp.status}`);
    }
    const data = await resp.json();
    activeTripBadge.textContent = `Trip ${data.trip_id} (${data.n_gps} pts)`;
    renderMap(data);
    renderMetrics(data);
    setStatus(`Completed trip ${data.trip_id} in ${data.timings_ms.total_ms.toFixed(0)} ms.`, false);
  } catch (err) {
    activeTripBadge.textContent = "Error";
    setStatus(`Prediction failed for ${tripId}: ${err.message}`, false);
  }
}

async function connectAndLoadTrips() {
  setStatus("Connecting to Laptop 1 FastAPI server...", true);
  try {
    const [healthResp, tripsResp] = await Promise.all([
      fetch(`${getApiBase()}/health`),
      fetch(`${getApiBase()}/api/trips`),
    ]);
    if (!healthResp.ok || !tripsResp.ok) {
      throw new Error("API health/trips endpoint failed.");
    }
    const health = await healthResp.json();
    const tripsPayload = await tripsResp.json();

    splitBadge.textContent = `${health.eval_split} (${tripsPayload.trips.length} trips)`;
    serverModelInfo.textContent = `Active model: ${health.model_name} (${health.algorithm_family})`;

    tripListEl.replaceChildren();
    for (const item of tripsPayload.trips) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "trip-btn";
      btn.dataset.tripId = item.trip_id;
      btn.setAttribute("aria-pressed", item.trip_id === selectedTripId ? "true" : "false");

      const leftSpan = document.createElement("span");
      leftSpan.textContent = `Trip ${item.trip_id}`;
      const rightSpan = document.createElement("span");
      rightSpan.className = "trip-meta";
      rightSpan.textContent = `${item.n_gps} pts | ${Math.round(item.duration_s)}s`;

      btn.append(leftSpan, rightSpan);
      btn.addEventListener("click", () => runPrediction(item.trip_id));
      li.appendChild(btn);
      tripListEl.appendChild(li);
    }

    setStatus(`Connected. Loaded ${tripsPayload.trips.length} trips from Laptop 1.`, false);
    if (tripsPayload.trips.length > 0 && !selectedTripId) {
      runPrediction(tripsPayload.trips[0].trip_id);
    }
  } catch (err) {
    splitBadge.textContent = "Disconnected";
    serverModelInfo.textContent = "Check that Laptop 1 FastAPI server is running and reachable.";
    setStatus(`Connection error: ${err.message}`, false);
  }
}

apiForm.addEventListener("submit", (e) => {
  e.preventDefault();
  connectAndLoadTrips();
});

for (const cb of [toggleGps, toggleNet, toggleCls, toggleHmm]) {
  cb.addEventListener("change", syncLayerVisibility);
}

window.addEventListener("DOMContentLoaded", () => {
  initLeafletIfAvailable();
  connectAndLoadTrips();
});
