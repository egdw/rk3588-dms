const IS_FILE_PREVIEW = window.location.protocol === "file:";
const LOCAL_BACKEND_CANDIDATES = ["https://127.0.0.1:8444", "https://127.0.0.1:8443", "http://127.0.0.1:8000", "http://127.0.0.1:8002"];
let LOCAL_BACKEND = localStorage.getItem("visionSentinelLocalBackend") || LOCAL_BACKEND_CANDIDATES[0];
const API = IS_FILE_PREVIEW ? LOCAL_BACKEND : "";

const labels = [
  { name: "Normal Driving", key: "Safe State", color: "#22c55e", count: 0, accuracy: 94, aliases: ["\u6b63\u5e38\u9a7e\u9a76"] },
  { name: "Eye Closure", key: "Fatigue Eye Closure", color: "#0ea5e9", count: 0, accuracy: 84, aliases: ["\u95ed\u773c\u75b2\u52b3", "\u95ed\u773c/\u7761\u7720"] },
  { name: "Head Down", key: "Head-Down Distraction", color: "#f59e0b", count: 0, accuracy: 81, aliases: ["\u4f4e\u5934\u5206\u5fc3"] },
  { name: "Gaze Offset", key: "Gaze Offset", color: "#8b5cf6", count: 0, accuracy: 80, aliases: ["\u89c6\u7ebf\u504f\u79fb"] },
  { name: "Phone Call", key: "Calling Behavior", color: "#ef4444", count: 0, accuracy: 88, aliases: ["\u6253\u7535\u8bdd", "\u901a\u8bdd\u884c\u4e3a"] },
];

const visibleTextMap = new Map([
  ["\u6b63\u5e38\u9a7e\u9a76", "Normal Driving"],
  ["\u95ed\u773c\u75b2\u52b3", "Eye Closure"],
  ["\u95ed\u773c/\u7761\u7720", "Eye Closure/Sleep"],
  ["\u4f4e\u5934\u5206\u5fc3", "Head-Down Distraction"],
  ["\u89c6\u7ebf\u504f\u79fb", "Gaze Offset"],
  ["\u6253\u7535\u8bdd", "Phone Call"],
  ["\u901a\u8bdd\u884c\u4e3a", "Calling Behavior"],
  ["\u901a\u8bdd", "Phone Call"],
  ["\u5b89\u5168\u72b6\u6001", "Safe State"],
  ["\u75b2\u52b3\u95ed\u773c", "Fatigue Eye Closure"],
  ["\u53f8\u673a\u72b6\u6001\u68c0\u6d4b\u6570\u636e\u96c6", "Driver State Detection Dataset"],
  ["\u53f8\u673a\u5371\u9669\u9a7e\u9a76\u72b6\u6001\u6570\u636e\u96c6", "Driver Risk State Dataset"],
  ["\u53f8\u673a\u72b6\u6001\u89c6\u89c9\u68c0\u6d4b\u6570\u636e", "Driver state vision detection data"],
  ["\u9a7e\u9a76\u8231\u6444\u50cf\u5934", "Cabin Camera"],
  ["\u884c\u8f66\u8bb0\u5f55\u4eea", "Dash Camera"],
  ["\u516c\u5f00\u6570\u636e\u96c6", "Public Dataset"],
  ["\u6a21\u62df\u9a7e\u9a76\u62cd\u6444", "Simulator Capture"],
  ["\u8f7b\u91cf\u68c0\u6d4b\u6a21\u578b", "Light Detection Model"],
  ["\u6807\u51c6\u68c0\u6d4b\u6a21\u578b", "Standard Detection Model"],
  ["\u9ad8\u7cbe\u5ea6\u68c0\u6d4b\u6a21\u578b", "High-Accuracy Detection Model"],
  ["\u73b0\u6210\u72b6\u6001\u68c0\u6d4b\u6a21\u578b", "Ready State Detection Model"],
  ["\u73b0\u6210\u884c\u4e3a\u76ee\u6807\u6a21\u578b", "Ready Action Object Model"],
]);

function localizeText(value) {
  if (value === null || value === undefined) return value;
  return String(value).replace(/[\u4e00-\u9fff/、（）·：，。；]+/g, (text) => {
    let translated = text;
    Array.from(visibleTextMap.entries())
      .sort((a, b) => b[0].length - a[0].length)
      .forEach(([source, target]) => {
        translated = translated.split(source).join(target);
      });
    if (/[\u4e00-\u9fff]/.test(translated)) return "Imported Content";
    return translated;
  });
}

const state = {
  datasets: [],
  activeDatasetId: localStorage.getItem("visionSentinelDatasetId") || "",
  samples: [],
  models: [],
  activeModelId: "",
  activeJobId: "",
  jobTimer: null,
  sampleIndex: 0,
  annotationBoxes: [],
  annotationDraft: null,
  annotationStart: null,
  sampleMediaReady: false,
  modelMetrics: null,
};

const $ = (selector) => document.querySelector(selector);

function installWorkflowControls() {
  const processGrid = document.querySelector(".process-grid");
  if (processGrid && !$("#qualityFilterInput")) {
    const options = document.createElement("div");
    options.className = "process-options";
    options.innerHTML = `
      <label><input id="qualityFilterInput" type="checkbox" checked /> Filter dark and blurry samples</label>
      <label><input id="augmentationInput" type="checkbox" checked /> Generate augmented training samples</label>
      <label>Video FPS <input id="videoFpsInput" type="number" min="0.1" max="10" step="0.1" value="1" /></label>
    `;
    processGrid.after(options);
  }
  const badge = $("#aucBadge");
  if (badge && !$("#evaluateBtn")) {
    badge.classList.remove("done");
    badge.textContent = "Not Evaluated";
    const button = document.createElement("button");
    button.id = "evaluateBtn";
    button.className = "secondary-action compact-action";
    button.type = "button";
    button.innerHTML = `<span class="material-symbols-outlined">analytics</span>Run Evaluation`;
    badge.after(button);
    const selector = document.createElement("select");
    selector.id = "registeredModelSelect";
    selector.className = "compact-model-select";
    selector.setAttribute("aria-label", "Select registered model");
    button.after(selector);
  }
}

installWorkflowControls();

const materialIconFallbacks = {
  radar: "◎", notifications: "●", add_circle: "+", cloud_upload: "⇧", dataset: "▦",
  fact_check: "✓", model_training: "◈", speed: "◔", dns: "▤", warning: "!",
  refresh: "↻", sentiment_dissatisfied: "◡", phone_in_talk: "☎", check_circle: "✓",
  arrow_forward: "→", image: "▧", videocam: "▣", sensors: "◉", verified_user: "◇",
  visibility_off: "⊘", face_5: "◉", smoking_rooms: "≈", restaurant: "▥",
  airline_seat_recline_normal: "⌁", center_focus_strong: "⌾", memory: "▦", save: "▣",
  skip_previous: "◀", skip_next: "▶", face: "◉", face_6: "◉", bedtime: "◔",
  visibility: "◎", tune: "☷", auto_fix_high: "✦", analytics: "▥", ios_share: "⇧",
  support_agent: "?",
  delete_sweep: "×",
};

function installFilePreviewIconFallbacks() {
  if (!IS_FILE_PREVIEW) return;
  document.documentElement.classList.add("material-icons-fallback");
  document.querySelectorAll(".material-symbols-outlined").forEach((icon) => {
    const name = icon.textContent.trim();
    icon.dataset.iconName = name;
    icon.textContent = materialIconFallbacks[name] || "•";
  });
}

async function redirectFilePreviewToBackend() {
  if (!IS_FILE_PREVIEW) return false;
  for (const backendUrl of [LOCAL_BACKEND, ...LOCAL_BACKEND_CANDIDATES]) {
    try {
      const response = await fetch(`${backendUrl}/api/health`, { cache: "no-store" });
      if (!response.ok) continue;
      LOCAL_BACKEND = backendUrl;
      localStorage.setItem("visionSentinelLocalBackend", backendUrl);
      window.location.replace(`${backendUrl}/index.html${window.location.hash || ""}`);
      return true;
    } catch {
      // Try the next known local development endpoint.
    }
  }
  return false;
}

const backendStatus = $("#backendStatus");
const createDatasetBtn = $("#createDatasetBtn");
const fileInput = $("#fileInput");
const refreshBtn = $("#refreshBtn");
const processBtn = $("#processBtn");
const trainBtn = $("#trainBtn");
const saveLabel = $("#saveLabel");
const exportBtn = $("#exportBtn");
const evaluateBtn = $("#evaluateBtn");
const registeredModelSelect = $("#registeredModelSelect");
const prevFrame = $("#prevFrame");
const nextFrame = $("#nextFrame");
const clearBoxes = $("#clearBoxes");
const annotationCanvas = $("#annotationCanvas");
const samplePreviewImage = $("#samplePreviewImage");
const samplePreviewVideo = $("#samplePreviewVideo");
const annotationHint = $("#annotationHint");

const activeDatasetName = $("#activeDatasetName");
const activeDatasetMeta = $("#activeDatasetMeta");
const activeDatasetId = $("#activeDatasetId");
const totalSamples = $("#totalSamples");
const annotatedSamples = $("#annotatedSamples");
const dashboardAnnotationRate = $("#dashboardAnnotationRate");
const dashboardAnnotationBar = $("#dashboardAnnotationBar");
const modelCount = $("#modelCount");
const jobProgress = $("#jobProgress");
const jobStatusText = $("#jobStatusText");
const datasetList = $("#datasetList");
const classList = $("#classList");
const sampleList = $("#sampleList");
const sampleCursor = $("#sampleCursor");
const labelSelect = $("#labelSelect");
const jobLog = $("#jobLog");
const activeJobId = $("#activeJobId");
const progressBar = $("#progressBar");
const toast = $("#toast");

let curveData = [];
let confusionMatrixData = null;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2800);
}

function showPage(pageName) {
  const target = pageName || "dashboard";
  document.querySelectorAll("[data-page]").forEach((page) => {
    page.classList.toggle("active", page.dataset.page === target);
  });
  document.querySelectorAll("[data-page-target]").forEach((link) => {
    const isActive = link.dataset.pageTarget === target;
    link.classList.toggle("active", isActive);
    if (isActive) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value || 0);
}

function translateJobType(type) {
  return (
    {
      process: "Dataset Processing",
      train: "Model Training",
      evaluate: "Result Evaluation",
      export: "Model Export",
    }[type] || type || "Job"
  );
}

function translateJobStatus(status) {
  return (
    {
      queued: "Queued",
      running: "Running",
      succeeded: "Completed",
      failed: "Failed",
    }[status] || status || "Waiting"
  );
}

function translateMediaType(type) {
  return (
    {
      image: "Image",
      video: "Video",
      other: "File",
    }[type] || type || "File"
  );
}

function translateSampleStatus(status) {
  return (
    {
      uploaded: "Uploaded",
      annotated: "Annotated",
      processed: "Processed",
    }[status] || status || "Pending"
  );
}

function normalizeModelName(name) {
  return (
    {
      "Light Detection Model": "yolov8n.pt",
      "Standard Detection Model": "yolov8s.pt",
      "High-Accuracy Detection Model": "yolov8m.pt",
    }[name] || name
  );
}

function metricPercent(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value) * 100)}%` : "--";
}

function renderModelMetrics(metrics) {
  state.modelMetrics = metrics && Object.keys(metrics).length ? metrics : null;
  $("#precisionScore").textContent = metricPercent(metrics?.precision);
  $("#recallScore").textContent = metricPercent(metrics?.recall);
  $("#f1Score").textContent = metricPercent(metrics?.f1);
  $("#detailMapScore").textContent = metricPercent(metrics?.map5095 ?? metrics?.map50);
  $("#aucBadge").textContent = metrics?.source === "ultralytics-test"
    ? "Real Test Metrics"
    : metrics?.source === "ultralytics"
      ? "Training Validation Metrics"
      : "Not Evaluated";
  $("#aucBadge").classList.toggle("done", Boolean(metrics?.source));
  curveData = (metrics?.history || [])
    .filter((item) => Number.isFinite(item.loss) || Number.isFinite(item.map50))
    .map((item) => ({ epoch: item.epoch, loss: item.loss, accuracy: item.map50 }));
  renderLossChart();
  renderRocChart();
  renderMatrix();
}

function translateExportFormat(format) {
  return (
    {
      pt: "Weight File",
      onnx: "Universal Model",
      engine: "Inference Engine",
      tflite: "Mobile Model",
    }[format] || format
  );
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API}${path}`, {
      headers: options.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    const startCommand = "npm start";
    throw new Error(IS_FILE_PREVIEW
      ? `This is a local file preview and the backend is not connected. Please run: ${startCommand}`
      : `Backend service is not connected. Please run: ${startCommand}`);
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(localizeText(payload.error) || `Request failed: ${response.status}`);
  }
  return payload;
}

async function checkBackend() {
  try {
    const health = await requestJson("/api/health");
    backendStatus.className = "status-chip online";
    backendStatus.innerHTML = `<span class="status-dot"></span>Backend Online`;
    backendStatus.title = health.capabilities?.ultralytics_training
      ? "Dataset processing, training, evaluation, export, and browser ONNX inference are available"
      : "Dataset processing and browser ONNX inference are available; real training requires Ultralytics in the project virtual environment";
    return true;
  } catch (error) {
    backendStatus.className = "status-chip offline";
    backendStatus.innerHTML = `<span class="status-dot"></span>Backend Offline`;
    jobLog.textContent = "Please start the backend with: npm start";
    return false;
  }
}

async function createDataset() {
  const payload = {
    name: $("#datasetName").value.trim() || "Driver Risk State Dataset",
    source: $("#datasetSource").value,
    description: $("#datasetDescription").value.trim(),
  };
  const result = await requestJson("/api/datasets", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.activeDatasetId = result.id;
  localStorage.setItem("visionSentinelDatasetId", result.id);
  showToast("Dataset created");
  await refreshAll();
}

async function loadDatasets() {
  const result = await requestJson("/api/datasets");
  state.datasets = result.items || [];
  if (!state.activeDatasetId && state.datasets[0]) {
    state.activeDatasetId = state.datasets[0].id;
    localStorage.setItem("visionSentinelDatasetId", state.activeDatasetId);
  }
  if (state.activeDatasetId && !state.datasets.some((item) => item.id === state.activeDatasetId)) {
    state.activeDatasetId = state.datasets[0]?.id || "";
  }
  renderDatasets();
}

async function loadSummary() {
  if (!state.activeDatasetId) {
    renderSummary(null);
    return;
  }
  const summary = await requestJson(`/api/datasets/${state.activeDatasetId}/summary`);
  renderSummary(summary);
}

async function loadSamples() {
  if (!state.activeDatasetId) {
    state.samples = [];
    renderSamples();
    return;
  }
  const result = await requestJson(`/api/datasets/${state.activeDatasetId}/samples`);
  state.samples = result.items || [];
  state.sampleIndex = Math.min(state.sampleIndex, Math.max(state.samples.length - 1, 0));
  renderSamples();
  await loadCurrentSample();
}

async function loadModels() {
  const result = await requestJson("/api/models");
  state.models = result.items || [];
  if (!state.models.some((model) => model.id === state.activeModelId)) {
    state.activeModelId = state.models[0]?.id || "";
  }
  modelCount.textContent = formatNumber(state.models.length);
  registeredModelSelect.innerHTML = state.models.length
    ? state.models.map((model) => `<option value="${model.id}">${localizeText(model.name)}</option>`).join("")
    : `<option value="">No models</option>`;
  registeredModelSelect.value = state.activeModelId;
  renderModelMetrics(state.models.find((model) => model.id === state.activeModelId)?.metrics || null);
}

async function loadJobs() {
  const result = await requestJson("/api/jobs");
  const jobs = result.items || [];
  const latest = jobs[0];
  if (!state.activeJobId && latest) {
    renderJob(latest);
  }
}

function renderDatasets() {
  datasetList.innerHTML =
    state.datasets
      .map(
        (dataset) => `
          <button class="list-row" type="button" data-dataset-id="${dataset.id}">
            <span>
              <strong>${localizeText(dataset.name)}</strong>
              <small>${localizeText(dataset.source)} · ${new Date(dataset.created_at * 1000).toLocaleString("en-US")}</small>
            </span>
            <span class="pill">${dataset.id === state.activeDatasetId ? "Active" : "Select"}</span>
          </button>
        `,
      )
      .join("") || `<div class="list-row"><span><strong>No datasets</strong><small>Click "Create Dataset" to begin.</small></span></div>`;

  datasetList.querySelectorAll("[data-dataset-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.activeDatasetId = button.dataset.datasetId;
      localStorage.setItem("visionSentinelDatasetId", state.activeDatasetId);
      await refreshAll();
    });
  });
}

function renderSummary(summary) {
  if (!summary) {
    activeDatasetName.textContent = "No dataset selected";
    activeDatasetMeta.textContent = "Start the backend to create datasets and upload samples.";
    activeDatasetId.textContent = "Dataset ID: -";
    totalSamples.textContent = "0";
    annotatedSamples.textContent = "0";
    if (dashboardAnnotationRate) dashboardAnnotationRate.textContent = "0%";
    if (dashboardAnnotationBar) dashboardAnnotationBar.style.width = "0%";
    labels.forEach((label) => (label.count = 0));
    renderClasses();
    return;
  }

  activeDatasetName.textContent = localizeText(summary.dataset.name);
  activeDatasetMeta.textContent = `${localizeText(summary.dataset.source)} · ${localizeText(summary.dataset.description || "Driver state vision detection data")}`;
  activeDatasetId.textContent = `Dataset ID: ${summary.dataset.id}`;
  totalSamples.textContent = formatNumber(summary.total_samples);
  annotatedSamples.textContent = formatNumber(summary.annotated_samples);
  const annotationRate = summary.total_samples
    ? Math.round((summary.annotated_samples / summary.total_samples) * 100)
    : 0;
  if (dashboardAnnotationRate) dashboardAnnotationRate.textContent = `${annotationRate}%`;
  if (dashboardAnnotationBar) dashboardAnnotationBar.style.width = `${annotationRate}%`;

  labels.forEach((label) => {
    label.count = label.aliases.reduce((total, alias) => total + (summary.labels?.[alias] || 0), summary.labels?.[label.name] || 0);
  });
  renderClasses();
}

function renderClasses() {
  const maxCount = Math.max(...labels.map((item) => item.count), 1);
  classList.innerHTML = labels
    .map(
      (item) => `
        <article class="class-row">
          <span>
            <strong>${item.name}</strong>
            <small>${formatNumber(item.count)} annotated samples · ${item.key}</small>
          </span>
          <span class="pill" style="color:${item.color}">${item.accuracy}%</span>
          <div class="bar"><i style="width:${Math.max(6, Math.round((item.count / maxCount) * 100))}%; background:${item.color}"></i></div>
        </article>
      `,
    )
    .join("");
}

function renderOptions() {
  labelSelect.innerHTML = labels.map((item) => `<option value="${item.name}">${item.name}</option>`).join("");
  const modelSelect = $("#modelSelect");
  modelSelect.innerHTML = `
    <option value="Driver-Monitoring-System/models/soham/best.pt">Ready State Detection Weights (Local)</option>
    <option value="Driver-Monitoring-System/models/chaitanya/best.pt">Ready Action Object Weights (Local)</option>
    <option value="Light Detection Model">Light Detection Model (requires pretrained weights)</option>
    <option value="Standard Detection Model">Standard Detection Model (requires pretrained weights)</option>
    <option value="High-Accuracy Detection Model">High-Accuracy Detection Model (requires pretrained weights)</option>
  `;
}

function renderSamples() {
  sampleCursor.textContent = `Sample ${state.samples.length ? state.sampleIndex + 1 : 0} / ${state.samples.length}`;
  sampleList.innerHTML =
    state.samples
      .slice(0, 8)
      .map(
        (sample, index) => `
          <button class="sample-row" type="button" data-sample-index="${index}">
            <span>
              <strong>${sample.filename}</strong>
              <small>${translateSampleStatus(sample.status)} · ${translateMediaType(sample.media_type)}</small>
            </span>
            <span class="pill">${localizeText(sample.label) || "Unlabeled"}</span>
          </button>
        `,
      )
      .join("") || `<div class="sample-row"><span><strong>No samples</strong><small>Uploaded images and videos will appear here.</small></span></div>`;

  sampleList.querySelectorAll("[data-sample-index]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.sampleIndex = Number(button.dataset.sampleIndex);
      renderSamples();
      await loadCurrentSample();
    });
  });
}

function currentSample() {
  return state.samples[state.sampleIndex] || null;
}

function activeSampleMedia() {
  if (samplePreviewImage?.classList.contains("active")) return samplePreviewImage;
  if (samplePreviewVideo?.classList.contains("active")) return samplePreviewVideo;
  return null;
}

function annotationGeometry() {
  const stage = $("#detectionStage");
  const media = activeSampleMedia();
  if (!stage || !media || !state.sampleMediaReady) return null;
  const sourceWidth = media instanceof HTMLVideoElement ? media.videoWidth : media.naturalWidth;
  const sourceHeight = media instanceof HTMLVideoElement ? media.videoHeight : media.naturalHeight;
  if (!sourceWidth || !sourceHeight) return null;
  const stageWidth = stage.clientWidth;
  const stageHeight = stage.clientHeight;
  const scale = Math.min(stageWidth / sourceWidth, stageHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return { stage, width, height, offsetX: (stageWidth - width) / 2, offsetY: (stageHeight - height) / 2 };
}

function drawAnnotationBoxes() {
  if (!annotationCanvas) return;
  const geometry = annotationGeometry();
  const stage = $("#detectionStage");
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, stage?.clientWidth || 1);
  const height = Math.max(1, stage?.clientHeight || 1);
  if (annotationCanvas.width !== Math.round(width * pixelRatio) || annotationCanvas.height !== Math.round(height * pixelRatio)) {
    annotationCanvas.width = Math.round(width * pixelRatio);
    annotationCanvas.height = Math.round(height * pixelRatio);
  }
  const context = annotationCanvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!geometry) return;
  const boxes = state.annotationDraft ? [...state.annotationBoxes, state.annotationDraft] : state.annotationBoxes;
  boxes.forEach((box, index) => {
    const x = geometry.offsetX + (box.x - box.w / 2) * geometry.width;
    const y = geometry.offsetY + (box.y - box.h / 2) * geometry.height;
    const boxWidth = box.w * geometry.width;
    const boxHeight = box.h * geometry.height;
    context.strokeStyle = index === boxes.length - 1 && state.annotationDraft ? "#f59e0b" : "#0ea5e9";
    context.fillStyle = "rgba(14, 165, 233, 0.12)";
    context.lineWidth = 2;
    context.fillRect(x, y, boxWidth, boxHeight);
    context.strokeRect(x, y, boxWidth, boxHeight);
    const label = box.label || labelSelect.value;
    context.font = "700 12px system-ui, sans-serif";
    const labelWidth = context.measureText(label).width + 14;
    context.fillStyle = "#0ea5e9";
    context.fillRect(x, Math.max(0, y - 24), labelWidth, 24);
    context.fillStyle = "#ffffff";
    context.fillText(label, x + 7, Math.max(16, y - 7));
  });
}

function annotationPoint(event) {
  const geometry = annotationGeometry();
  if (!geometry) return null;
  const rect = geometry.stage.getBoundingClientRect();
  const x = (event.clientX - rect.left - geometry.offsetX) / geometry.width;
  const y = (event.clientY - rect.top - geometry.offsetY) / geometry.height;
  return { x: Math.min(Math.max(x, 0), 1), y: Math.min(Math.max(y, 0), 1) };
}

async function loadCurrentSample() {
  const sample = currentSample();
  if (!sample || !samplePreviewImage || !samplePreviewVideo) {
    state.annotationBoxes = [];
    state.sampleMediaReady = false;
    drawAnnotationBoxes();
    return;
  }
  state.sampleMediaReady = false;
  state.annotationDraft = null;
  samplePreviewImage.classList.remove("active");
  samplePreviewVideo.classList.remove("active");
  samplePreviewVideo.pause();
  samplePreviewVideo.removeAttribute("src");
  $("#detectionCanvas")?.classList.remove("active");
  $("#cameraVideo")?.classList.remove("active");
  $("#detectionPlaceholder").hidden = true;

  const annotationResult = await requestJson(`/api/samples/${sample.id}/annotation`);
  const annotation = annotationResult.annotation;
  state.annotationBoxes = Array.isArray(annotation?.boxes) ? annotation.boxes : [];
  if (annotation?.label) {
    const matchedLabel = labels.find((item) => item.name === annotation.label || item.aliases.includes(annotation.label));
    if (matchedLabel) labelSelect.value = matchedLabel.name;
  }
  if (annotation?.task_type) {
    const mode = document.querySelector(`input[name="labelMode"][value="${annotation.task_type}"]`);
    if (mode) mode.checked = true;
  }

  const fileUrl = `/api/samples/${sample.id}/file?time=${Date.now()}`;
  if (String(sample.media_type).startsWith("video/")) {
    samplePreviewVideo.onloadedmetadata = () => {
      state.sampleMediaReady = true;
      samplePreviewVideo.classList.add("active");
      annotationCanvas.classList.add("active");
      annotationCanvas.classList.toggle("drawing-enabled", document.querySelector("input[name='labelMode']:checked")?.value === "detection");
      drawAnnotationBoxes();
    };
    samplePreviewVideo.src = fileUrl;
    samplePreviewVideo.load();
  } else {
    samplePreviewImage.onload = () => {
      state.sampleMediaReady = true;
      samplePreviewImage.classList.add("active");
      annotationCanvas.classList.add("active");
      annotationCanvas.classList.toggle("drawing-enabled", document.querySelector("input[name='labelMode']:checked")?.value === "detection");
      drawAnnotationBoxes();
    };
    samplePreviewImage.src = fileUrl;
  }
  annotationHint.textContent = annotation
    ? `${state.annotationBoxes.length} detection boxes loaded; drag to edit`
    : "Drag on the sample frame to draw detection boxes, or switch to image classification";
}

async function uploadSamples(files) {
  if (!state.activeDatasetId) {
    await createDataset();
  }
  if (!files?.length) return;
  const body = new FormData();
  Array.from(files).forEach((file) => body.append("files", file));
  const result = await requestJson(`/api/datasets/${state.activeDatasetId}/upload`, {
    method: "POST",
    body,
  });
  showToast(`${result.saved.length} samples uploaded`);
  await refreshAll();
}

async function saveAnnotation() {
  if (!state.activeDatasetId) {
    showToast("Create a dataset first");
    return;
  }
  const sample = state.samples[state.sampleIndex];
  if (!sample) {
    showToast("Upload and select a dataset sample first");
    return;
  }
  const label = labelSelect.value;
  const taskType = document.querySelector("input[name='labelMode']:checked")?.value || "detection";
  if (taskType === "detection" && !state.annotationBoxes.length) {
    showToast("Draw at least one detection box on the sample frame first");
    return;
  }
  await requestJson("/api/annotations", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: state.activeDatasetId,
      sample_id: sample?.id,
      task_type: taskType,
      label,
      boxes: taskType === "detection" ? state.annotationBoxes.map((box) => ({ ...box, label })) : [],
    }),
  });
  showToast(`Annotation saved: ${label}`);
  await refreshAll();
}

async function startJob(path, payload) {
  const result = await requestJson(path, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.activeJobId = result.job_id;
  activeJobId.textContent = result.job_id;
  pollJob(result.job_id);
  return result.job_id;
}

async function startProcess() {
  if (!state.activeDatasetId) {
    showToast("Create a dataset first");
    return;
  }
  const train = Number($("#trainSplit").value);
  const val = Number($("#validSplit").value);
  const test = Number($("#testSplit").value);
  await startJob("/api/process-jobs", {
    dataset_id: state.activeDatasetId,
    options: {
      target_size: Number($("#imageSizeInput").value) || 640,
      quality_filter: $("#qualityFilterInput").checked,
      augment: $("#augmentationInput").checked,
      video_fps: Number($("#videoFpsInput").value) || 1,
      splits: { train, val, test },
    },
  });
  showToast("Dataset processing job started");
}

async function startTraining() {
  if (!state.activeDatasetId) {
    showToast("Create a dataset first");
    return;
  }
  await startJob("/api/train-jobs", {
    dataset_id: state.activeDatasetId,
    model: normalizeModelName($("#modelSelect").value),
    image_size: Number($("#imageSizeInput").value),
    epochs: Number($("#epochsInput").value),
    batch_size: Number($("#batchInput").value),
    learning_rate: Number($("#lrInput").value),
    pretrained: $("#pretrainedInput").checked,
  });
  showToast("Model training job started");
}

async function exportModel() {
  await loadModels();
  if (!state.activeModelId) {
    showToast("No model is available for export. Complete training first.");
    return;
  }
  const format = document.querySelector("input[name='exportFormat']:checked")?.value || "pt";
  await startJob("/api/export-jobs", { model_id: state.activeModelId, format });
  showToast(`Export job started: ${translateExportFormat(format)}`);
}

async function evaluateModel() {
  await loadModels();
  if (!state.activeModelId) {
    showToast("No model is available for evaluation");
    return;
  }
  if (!state.activeDatasetId) {
    showToast("Select a processed dataset first");
    return;
  }
  await startJob("/api/evaluate-jobs", {
    model_id: state.activeModelId,
    dataset_id: state.activeDatasetId,
  });
  showToast("Real test-set evaluation job started");
}

async function pollJob(jobId) {
  window.clearInterval(state.jobTimer);
  async function tick() {
    const job = await requestJson(`/api/jobs/${jobId}`);
    renderJob(job);
    if (["succeeded", "failed"].includes(job.status)) {
      window.clearInterval(state.jobTimer);
      await loadModels();
      await loadSummary();
    }
  }
  await tick();
  state.jobTimer = window.setInterval(tick, 1200);
}

function renderJob(job) {
  if (!job) return;
  state.activeJobId = job.id;
  activeJobId.textContent = job.id;
  const progress = job.progress || 0;
  jobProgress.textContent = `${progress}%`;
  jobStatusText.textContent = `${translateJobType(job.type)} · ${translateJobStatus(job.status)}`;
  progressBar.style.width = `${progress}%`;
  jobLog.textContent = [job.message, job.log].filter(Boolean).map(localizeText).join("\n\n") || "Job running...";
}

function pointsFor(data, valueKey, min, max, invert = true) {
  const width = 540;
  const height = 210;
  const pad = 30;
  if (!data.length) return "";
  const xMax = Math.max(...data.map((item) => item.epoch), 1);
  const range = Math.max(max - min, 0.000001);
  return data
    .filter((item) => Number.isFinite(item[valueKey]))
    .map((item) => {
      const x = pad + (item.epoch / xMax) * (width - pad * 2);
      const normalized = (item[valueKey] - min) / range;
      const y = invert
        ? height - pad - normalized * (height - pad * 2)
        : pad + normalized * (height - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function renderLossChart() {
  if (!curveData.length) {
    $("#lossChart").innerHTML = `<div class="empty-chart">Real training loss and mAP50 curves will appear here</div>`;
    return;
  }
  const losses = curveData.map((item) => item.loss).filter(Number.isFinite);
  const lossMin = losses.length ? Math.min(...losses) : 0;
  const lossMax = losses.length ? Math.max(...losses) : 1;
  $("#lossChart").innerHTML = `
    <svg viewBox="0 0 540 210" aria-hidden="true">
      <line x1="30" y1="180" x2="510" y2="180" stroke="#9aa6b5" />
      <line x1="30" y1="30" x2="30" y2="180" stroke="#9aa6b5" />
      <text class="chart-label" x="32" y="24">Metric Value</text>
      <text class="chart-label" x="448" y="198">Training Epoch</text>
      <polyline points="${pointsFor(curveData, "loss", lossMin, lossMax)}" fill="none" stroke="#ef4444" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
      <polyline points="${pointsFor(curveData, "accuracy", 0, 1)}" fill="none" stroke="#0ea5e9" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  `;
}

function renderRocChart() {
  const artifact = state.modelMetrics?.artifacts?.pr_curve;
  $("#rocChart").innerHTML = artifact
    ? `<img class="evaluation-artifact" src="${artifact}" alt="Real precision-recall curve" />`
    : `<div class="empty-chart">Run a real evaluation to show the precision-recall curve</div>`;
}

function renderMatrix() {
  const artifact = state.modelMetrics?.artifacts?.confusion_matrix;
  $("#confusionMatrix").innerHTML = artifact
    ? `<img class="evaluation-artifact" src="${artifact}" alt="Real class confusion matrix" />`
    : `<div class="empty-chart">Run a real evaluation to show the confusion matrix</div>`;
}

async function refreshAll() {
  const online = await checkBackend();
  if (!online) return;
  await loadDatasets();
  await Promise.all([loadSummary(), loadSamples(), loadModels(), loadJobs()]);
}

function bindEvents() {
  document.querySelectorAll("[data-page-target]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      const pageName = link.dataset.pageTarget;
      history.replaceState(null, "", `#${pageName}`);
      showPage(pageName);
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });

  window.addEventListener("hashchange", () => {
    showPage(window.location.hash.replace("#", "") || "dashboard");
    window.scrollTo({ top: 0 });
  });


  createDatasetBtn.addEventListener("click", () => createDataset().catch((error) => showToast(error.message)));
  refreshBtn.addEventListener("click", () => refreshAll().catch((error) => showToast(error.message)));
  fileInput.addEventListener("change", (event) => uploadSamples(event.target.files).catch((error) => showToast(error.message)));
  document.querySelectorAll(".upload-action input:not(#fileInput)").forEach((input) => {
    input.addEventListener("change", (event) => uploadSamples(event.target.files).catch((error) => showToast(error.message)));
  });
  saveLabel.addEventListener("click", () => saveAnnotation().catch((error) => showToast(error.message)));
  processBtn.addEventListener("click", () => startProcess().catch((error) => showToast(error.message)));
  trainBtn.addEventListener("click", () => startTraining().catch((error) => showToast(error.message)));
  exportBtn.addEventListener("click", () => exportModel().catch((error) => showToast(error.message)));
  evaluateBtn.addEventListener("click", () => evaluateModel().catch((error) => showToast(error.message)));
  registeredModelSelect.addEventListener("change", () => {
    state.activeModelId = registeredModelSelect.value;
    renderModelMetrics(state.models.find((model) => model.id === state.activeModelId)?.metrics || null);
  });

  document.querySelectorAll(".box").forEach((box) => {
    box.addEventListener("click", () => {
      document.querySelectorAll(".box").forEach((item) => item.classList.remove("selected"));
      box.classList.add("selected");
    });
  });

  prevFrame.addEventListener("click", async () => {
    state.sampleIndex = Math.max(0, state.sampleIndex - 1);
    renderSamples();
    await loadCurrentSample();
  });

  nextFrame.addEventListener("click", async () => {
    state.sampleIndex = Math.min(Math.max(state.samples.length - 1, 0), state.sampleIndex + 1);
    renderSamples();
    await loadCurrentSample();
  });

  clearBoxes.addEventListener("click", () => {
    state.annotationBoxes = [];
    state.annotationDraft = null;
    drawAnnotationBoxes();
    annotationHint.textContent = "Detection boxes cleared; drag to draw again";
  });

  annotationCanvas.addEventListener("pointerdown", (event) => {
    if (document.querySelector("input[name='labelMode']:checked")?.value !== "detection") return;
    const point = annotationPoint(event);
    if (!point) return;
    state.annotationStart = point;
    state.annotationDraft = { label: labelSelect.value, x: point.x, y: point.y, w: 0.001, h: 0.001 };
    annotationCanvas.setPointerCapture(event.pointerId);
    drawAnnotationBoxes();
  });

  annotationCanvas.addEventListener("pointermove", (event) => {
    if (!state.annotationStart) return;
    const point = annotationPoint(event);
    if (!point) return;
    state.annotationDraft = {
      label: labelSelect.value,
      x: (state.annotationStart.x + point.x) / 2,
      y: (state.annotationStart.y + point.y) / 2,
      w: Math.abs(point.x - state.annotationStart.x),
      h: Math.abs(point.y - state.annotationStart.y),
    };
    drawAnnotationBoxes();
  });

  annotationCanvas.addEventListener("pointerup", (event) => {
    if (!state.annotationStart || !state.annotationDraft) return;
    if (state.annotationDraft.w >= 0.01 && state.annotationDraft.h >= 0.01) {
      state.annotationBoxes.push(state.annotationDraft);
      annotationHint.textContent = `${state.annotationBoxes.length} detection boxes drawn`;
    }
    state.annotationStart = null;
    state.annotationDraft = null;
    if (annotationCanvas.hasPointerCapture(event.pointerId)) annotationCanvas.releasePointerCapture(event.pointerId);
    drawAnnotationBoxes();
  });

  document.querySelectorAll("input[name='labelMode']").forEach((input) => {
    input.addEventListener("change", () => {
      annotationCanvas.classList.toggle("drawing-enabled", input.checked && input.value === "detection");
      annotationHint.textContent = input.value === "classification" && input.checked
        ? "Image classification mode does not require detection boxes"
        : "Drag on the sample frame to draw detection boxes";
    });
  });
  labelSelect.addEventListener("change", drawAnnotationBoxes);
  new ResizeObserver(drawAnnotationBoxes).observe($("#detectionStage"));

  [
    ["#trainSplit", "#trainSplitLabel"],
    ["#validSplit", "#validSplitLabel"],
    ["#testSplit", "#testSplitLabel"],
  ].forEach(([inputSelector, labelSelector]) => {
    const input = $(inputSelector);
    const label = $(labelSelector);
    input.addEventListener("input", () => {
      label.textContent = `${input.value}%`;
    });
  });
}

async function boot() {
  installFilePreviewIconFallbacks();
  showPage(window.location.hash.replace("#", "") || "dashboard");
  window.requestAnimationFrame(() => window.scrollTo({ top: 0, left: 0 }));
  bindEvents();
  renderOptions();
  renderClasses();
  renderSamples();
  renderLossChart();
  renderRocChart();
  renderMatrix();
  const redirected = await redirectFilePreviewToBackend();
  if (redirected) return;
  refreshAll().catch((error) => showToast(error.message));
}

boot();
