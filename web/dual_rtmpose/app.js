"use strict";

const CAM_COLORS = ["#00dc00", "#ffb400", "#48aaff", "#ff6f6f"];
const KP_THR = 0.3;
const SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 6], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

const main = document.querySelector("main");
const conn = document.getElementById("conn");

const state = {
  // per-camera latest snapshot (sparse — keyed by camera id)
  bundles: {},
  // per-camera render fps state
  renderTimes: {},
  renderFps: {},
  panes: {},          // { camId: { canvas, stats } }
  serverSeq: 0,
  serverLastMs: 0,
};

// Build (or reuse) the pane for a given camera id.
function ensurePane(camId) {
  if (state.panes[camId]) return state.panes[camId];

  const section = document.createElement("section");
  section.className = "pane";
  section.dataset.cam = String(camId);
  const h2 = document.createElement("h2");
  h2.textContent = `cam${camId}`;
  const canvas = document.createElement("canvas");
  canvas.dataset.cam = String(camId);
  canvas.width = 640;
  canvas.height = 480;
  const stats = document.createElement("pre");
  stats.className = "stats";
  stats.dataset.cam = String(camId);
  stats.textContent = "waiting…";
  section.appendChild(h2);
  section.appendChild(canvas);
  section.appendChild(stats);

  // Insert sorted by cam id
  let inserted = false;
  for (const sib of main.querySelectorAll("section.pane")) {
    if (Number(sib.dataset.cam) > camId) {
      main.insertBefore(section, sib);
      inserted = true;
      break;
    }
  }
  if (!inserted) main.appendChild(section);

  state.panes[camId] = { canvas, stats };
  state.renderTimes[camId] = [];
  state.renderFps[camId] = 0;
  return state.panes[camId];
}

function connect() {
  const wsProto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${wsProto}://${location.host}/ws`);
  ws.onopen = () => {
    conn.textContent = "live";
    conn.className = "conn live";
    setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send("ping");
    }, 5000);
  };
  ws.onclose = () => {
    conn.textContent = "disconnected — retrying";
    conn.className = "conn dead";
    setTimeout(connect, 1500);
  };
  ws.onerror = () => {
    conn.textContent = "error";
    conn.className = "conn dead";
  };
  ws.onmessage = (ev) => {
    let bundle;
    try {
      bundle = JSON.parse(ev.data);
    } catch (e) {
      return;
    }
    state.serverSeq = bundle.seq;
    state.serverLastMs = bundle.ts_ms;
    for (const cam of bundle.cameras || []) {
      ensurePane(cam.id);
      state.bundles[cam.id] = cam;
    }
  };
}

function drawCamera(camId) {
  const pane = state.panes[camId];
  if (!pane) return;
  const canvas = pane.canvas;
  const bundle = state.bundles[camId];
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!bundle) {
    ctx.fillStyle = "#666";
    ctx.font = "16px monospace";
    ctx.fillText("(no data)", 16, 32);
    return;
  }
  if (canvas.width !== bundle.w || canvas.height !== bundle.h) {
    canvas.width = bundle.w;
    canvas.height = bundle.h;
  }
  const color = CAM_COLORS[camId % CAM_COLORS.length];
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  for (const person of bundle.persons || []) {
    if (person.bbox) {
      const [x1, y1, x2, y2] = person.bbox;
      ctx.strokeStyle = "#444";
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
    }
    const kpts = person.kpts || [];
    for (const [a, b] of SKELETON) {
      if (!kpts[a] || !kpts[b]) continue;
      if (kpts[a][2] < KP_THR || kpts[b][2] < KP_THR) continue;
      ctx.beginPath();
      ctx.moveTo(kpts[a][0], kpts[a][1]);
      ctx.lineTo(kpts[b][0], kpts[b][1]);
      ctx.stroke();
    }
    for (const kp of kpts) {
      if (!kp || kp[2] < KP_THR) continue;
      ctx.beginPath();
      ctx.arc(kp[0], kp[1], 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function updateStats(camId) {
  const pane = state.panes[camId];
  if (!pane) return;
  const el = pane.stats;
  const bundle = state.bundles[camId];
  if (!bundle) {
    el.textContent = "waiting…";
    return;
  }
  const s = bundle.stats || {};
  const renderFps = state.renderFps[camId].toFixed(1);
  const latency = bundle.stats && bundle.stats.captured_at_ms && state.serverLastMs
    ? Math.max(0, state.serverLastMs - bundle.stats.captured_at_ms)
    : 0;
  el.textContent =
    `recv_fps        ${(s.recv_fps ?? 0).toFixed(2)}\n` +
    `render_fps      ${renderFps}\n` +
    `recent_pose_fps ${(s.recent_pose_fps ?? 0).toFixed(2)}\n` +
    `avg_pose_fps    ${(s.avg_pose_fps ?? 0).toFixed(2)}\n` +
    `stage_ms        ${(s.stage_ms ?? 0).toFixed(1)}\n` +
    `pending         ${s.pending ?? 0}\n` +
    `processed       ${s.processed ?? 0}\n` +
    `latency_ms      ${latency}\n` +
    `bundle_seq      ${state.serverSeq}`;
}

function renderTick() {
  const now = performance.now();
  for (const camIdStr of Object.keys(state.panes)) {
    const camId = Number(camIdStr);
    drawCamera(camId);
    const times = state.renderTimes[camId];
    times.push(now);
    while (times.length && now - times[0] > 1000) times.shift();
    state.renderFps[camId] = times.length;
    updateStats(camId);
  }
  requestAnimationFrame(renderTick);
}

connect();
requestAnimationFrame(renderTick);
