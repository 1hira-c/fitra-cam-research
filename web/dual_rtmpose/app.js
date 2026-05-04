const CAMERA_IDS = ["cam0", "cam1"];
const PANEL_STATE = new Map();

const connectionStatusEl = document.getElementById("connection-status");
const runnerStatusEl = document.getElementById("runner-status");
const runnerMessageEl = document.getElementById("runner-message");
const bundleSequenceEl = document.getElementById("bundle-sequence");

function getPanelState(cameraId) {
  if (PANEL_STATE.has(cameraId)) {
    return PANEL_STATE.get(cameraId);
  }

  const root = document.querySelector(`[data-camera-id="${cameraId}"]`);
  const canvas = root.querySelector('[data-role="canvas"]');
  const ctx = canvas.getContext("2d");
  const state = {
    root,
    canvas,
    ctx,
    payload: null,
    lastReceivedSequence: null,
    lastReceivedAtMs: 0,
    receiveTimes: [],
    renderTimes: [],
  };
  PANEL_STATE.set(cameraId, state);
  return state;
}

function setMetric(root, field, value) {
  const target = root.querySelector(`[data-field="${field}"]`);
  if (target) {
    target.textContent = value;
  }
}

function formatFps(samples) {
  if (samples.length < 2) {
    return "0.0";
  }
  const durationMs = samples[samples.length - 1] - samples[0];
  if (durationMs <= 0) {
    return "0.0";
  }
  return (((samples.length - 1) * 1000) / durationMs).toFixed(1);
}

function trimOldSamples(samples, nowMs, windowMs = 4000) {
  while (samples.length > 0 && nowMs - samples[0] > windowMs) {
    samples.shift();
  }
}

function drawSkeletonPanel(panelState) {
  const { ctx, canvas, payload, root, renderTimes } = panelState;
  const nowMs = performance.now();
  renderTimes.push(nowMs);
  trimOldSamples(renderTimes, nowMs);

  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, width, height);

  if (!payload) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "28px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("waiting for pose data", width / 2, height / 2);
    setMetric(root, "render-fps", formatFps(renderTimes));
    requestAnimationFrame(() => drawSkeletonPanel(panelState));
    return;
  }

  const frameWidth = payload.frame_size.width || width;
  const frameHeight = payload.frame_size.height || height;
  const scale = Math.min(width / frameWidth, height / frameHeight);
  const offsetX = (width - frameWidth * scale) / 2;
  const offsetY = (height - frameHeight * scale) / 2;
  const edges = payload.skeleton.edges || [];
  const scoreThreshold = payload.skeleton.score_threshold ?? 0;
  const persons = payload.persons || [];
  const palette = ["#38bdf8", "#fb7185", "#a78bfa", "#facc15"];

  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  persons.forEach((person, personIndex) => {
    const color = palette[personIndex % palette.length];
    const keypoints = person.keypoints2d || [];
    const bbox = person.bbox || null;

    if (bbox) {
      const [x1, y1, x2, y2] = bbox;
      ctx.strokeStyle = "rgba(250, 204, 21, 0.65)";
      ctx.lineWidth = 2 / scale;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 4 / scale;
    ctx.lineCap = "round";
    edges.forEach(([startIndex, endIndex]) => {
      const startPoint = keypoints[startIndex];
      const endPoint = keypoints[endIndex];
      if (
        !startPoint ||
        !endPoint ||
        startPoint[2] < scoreThreshold ||
        endPoint[2] < scoreThreshold
      ) {
        return;
      }
      ctx.beginPath();
      ctx.moveTo(startPoint[0], startPoint[1]);
      ctx.lineTo(endPoint[0], endPoint[1]);
      ctx.stroke();
    });

    keypoints.forEach(([x, y, score]) => {
      if (score < scoreThreshold) {
        return;
      }
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(x, y, 6 / scale, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  ctx.restore();

  const latencyMs = Math.max(0, Date.now() - payload.server_publish_ts * 1000);
  const keypointStats = persons.reduce(
    (summary, person) => {
      const stats = person.keypoint_stats || {};
      return {
        visibleCount: summary.visibleCount + (stats.visible_count || 0),
        maxScore: Math.max(summary.maxScore, stats.max_score || 0),
      };
    },
    { visibleCount: 0, maxScore: 0 },
  );
  setMetric(root, "sequence", String(payload.sequence));
  setMetric(root, "persons", String(persons.length));
  setMetric(root, "visible-kpts", String(keypointStats.visibleCount));
  setMetric(root, "max-score", keypointStats.maxScore.toFixed(2));
  setMetric(root, "avg-fps", payload.metrics.avg_fps.toFixed(1));
  setMetric(root, "recent-fps", (payload.metrics.recent_fps || 0).toFixed(1));
  setMetric(root, "avg-publish-fps", payload.metrics.avg_publish_fps.toFixed(1));
  setMetric(root, "stage-total-ms", payload.stage_ms.total.toFixed(1));
  setMetric(
    root,
    "pending-frames",
    String(payload.metrics.pending_frames || payload.metrics.dropped_frames || 0),
  );
  setMetric(root, "failures", String(payload.metrics.failures));
  setMetric(root, "latency-ms", latencyMs.toFixed(1));
  setMetric(root, "render-fps", formatFps(renderTimes));

  requestAnimationFrame(() => drawSkeletonPanel(panelState));
}

function resizeCanvas(panelState) {
  const panelWidth = panelState.root.clientWidth - 32;
  const frameWidth = panelState.payload?.frame_size?.width || 1280;
  const frameHeight = panelState.payload?.frame_size?.height || 720;
  const width = Math.max(320, Math.floor(panelWidth));
  const height = Math.max(180, Math.floor((width * frameHeight) / frameWidth));
  panelState.canvas.width = width;
  panelState.canvas.height = height;
}

function applyBundle(bundle) {
  runnerStatusEl.textContent = bundle.runner_status;
  runnerMessageEl.textContent = bundle.runner_message;
  bundleSequenceEl.textContent = String(bundle.bundle_sequence);

  const cameraPayloads = new Map((bundle.cameras || []).map((camera) => [camera.camera_id, camera]));
  CAMERA_IDS.forEach((cameraId) => {
    const panelState = getPanelState(cameraId);
    const payload = cameraPayloads.get(cameraId) || null;
    if (payload && payload.sequence !== panelState.lastReceivedSequence) {
      panelState.payload = payload;
      panelState.lastReceivedSequence = payload.sequence;
      panelState.lastReceivedAtMs = performance.now();
      panelState.receiveTimes.push(panelState.lastReceivedAtMs);
      trimOldSamples(panelState.receiveTimes, panelState.lastReceivedAtMs);
      resizeCanvas(panelState);
    }
    setMetric(panelState.root, "receive-fps", formatFps(panelState.receiveTimes));
  });
}

function connect() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${window.location.host}/ws/poses`);

  socket.addEventListener("open", () => {
    connectionStatusEl.textContent = "connected";
  });

  socket.addEventListener("message", (event) => {
    const bundle = JSON.parse(event.data);
    applyBundle(bundle);
  });

  socket.addEventListener("close", () => {
    connectionStatusEl.textContent = "reconnecting";
    window.setTimeout(connect, 1000);
  });

  socket.addEventListener("error", () => {
    connectionStatusEl.textContent = "error";
    socket.close();
  });
}

CAMERA_IDS.forEach((cameraId) => {
  const panelState = getPanelState(cameraId);
  resizeCanvas(panelState);
  drawSkeletonPanel(panelState);
});

window.addEventListener("resize", () => {
  CAMERA_IDS.forEach((cameraId) => resizeCanvas(getPanelState(cameraId)));
});

connect();
