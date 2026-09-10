(() => {
  "use strict";

  const INPUT_SIZE = 640;
  const CONFIDENCE_THRESHOLD = 0.25;
  const PHONE_CONFIDENCE_THRESHOLD = 0.2;
  const PHONE_ROI_INPUT_SIZE = 640;
  const PHONE_ROI_EXPAND = 2.6;
  const PHONE_ROI_ROTATIONS_DEGREES = [-22, 22];
  const PHONE_ROI_MIN_INTERVAL_MS = 1600;
  const PHONE_ROI_RESULT_HOLD_MS = 2400;
  const IOU_THRESHOLD = 0.45;
  const CAMERA_INTERVAL_MS = 500;
  const FACE_TRACK_INTERVAL_MS = 150;
  const CAMERA_MODEL_START_DELAY_MS = 600;
  const CAMERA_RENDER_MAX_WIDTH = 960;
  const CAMERA_RENDER_MAX_HEIGHT = 540;
  const CAMERA_CAPTURE_WIDTH = 320;
  const CAMERA_CAPTURE_HEIGHT = 180;
  const CAMERA_CAPTURE_MAX_WIDTH = 424;
  const CAMERA_CAPTURE_MAX_HEIGHT = 240;
  const CAMERA_CAPTURE_FPS = 8;
  const FACE_BOX_SMOOTHING_ALPHA = 0.58;
  const FACE_BOX_HOLD_MS = 700;
  const FACE_BOX_RESET_SHIFT = 0.72;
  const MAX_FACE_CANDIDATES = 1;
  const MIN_PRIMARY_FACE_SCORE = 0.18;
  const MIN_FACE_IMAGE_EVIDENCE = 0.26;
  const CAMERA_OVERLAY_MIRRORED = true;
  const OVERLAY_CALIBRATION_STORAGE_KEY = "visionSentinelOverlayCalibration";
  const DEFAULT_OVERLAY_CALIBRATION = Object.freeze({ offsetX: 0, offsetY: 0, scaleX: 1, scaleY: 1 });
  const ALERT_CONFIRM_MS = 1500;
  const GAZE_OFF_ALERT_CONFIRM_MS = 1800;
  // 闭眼报警:所有来源(关键点路径 / ONNX 模型)的 drowsy 统一要求在报警端
  // 持续满 1 秒才触发,即"完全闭眼/疲劳状态持续超过 1 秒才报警";
  // 关键点路径另有"完全闭眼"严格阈值(见 EYE_FULLY_CLOSED_CONFIDENCE)挡住眯眼误报。
  const DROWSY_ALERT_CONFIRM_MS = 1000;
  const ALERT_MISS_GRACE_MS = 700;
  const PHONE_MISS_GRACE_MS = 2200;
  const ALERT_HOLD_MS = 3000;
  const ALERT_AUDIO_COOLDOWN_MS = 4500;
  const HEAD_DOWN_CONFIDENCE_THRESHOLD = 0.56;
  const SIXDREPNET_STATUS_PATH = "/api/head-pose/6drepnet/status";
  const SIXDREPNET_INFERENCE_PATH = "/api/head-pose/6drepnet";
  const SIXDREPNET_MIN_INTERVAL_MS = 650;
  const SIXDREPNET_FACE_CROP_SIZE = 256;
  const SIXDREPNET_HEAD_DOWN_PITCH_DEGREES = 16;
  const SIXDREPNET_HEAD_DOWN_RANGE_DEGREES = 18;
  const EYE_CLOSED_CONFIDENCE_THRESHOLD = 0.52;
  const EYE_CLOSED_CONFIRM_MS = 300;
  // 报警判定使用更严格的"完全闭眼"阈值:0.52~0.75 之间的眯眼/小眼睛处于灰区,
  // 关键点路径不会因此产生 drowsy 检测(计时器随时归零)。
  const EYE_FULLY_CLOSED_CONFIDENCE = 0.75;
  // soham 模型的 "Drowsy" 类是整图分类,对正常驾驶误报率高:
  // 模型来源的疲劳检测要求更高置信度,且关键点眼部状态明确为睁眼时直接抑制。
  const DROWSY_MODEL_CONFIDENCE_THRESHOLD = 0.6;
  const GAZE_OFF_CONFIDENCE_THRESHOLD = 0.62;
  const GAZE_OFF_RELEASE_THRESHOLD = 0.34;
  const GAZE_OFF_CONFIRM_MS = 400;
  const GAZE_OFF_HOLD_MS = 500;
  const GAZE_EYE_CLOSED_SUPPRESS_THRESHOLD = 0.38;
  const GAZE_OFF_HEAD_YAW_START_DEGREES = 15;
  const GAZE_OFF_HEAD_YAW_FULL_DEGREES = 32;
  const GAZE_OFF_HEAD_PITCH_DOWN_START_DEGREES = 18;
  const GAZE_OFF_HEAD_PITCH_UP_START_DEGREES = 20;
  const GAZE_OFF_HEAD_PITCH_FULL_DEGREES = 36;
  const HEAD_POSE_MODEL = {
    // 已本地化到 vendor/mediapipe/,不再依赖外网 CDN(网络不稳定会导致模型加载失败)
    bundle: "/vendor/mediapipe/vision_bundle.mjs",
    wasmPath: "/vendor/mediapipe/wasm",
    modelPath: "/vendor/mediapipe/face_landmarker.task",
  };
  // 跨域隔离(SAB 可用)时用 WebGPU 加速版,否则退回纯 WASM 版
  //(jsep/webgpu 构建的 wasm 为 pthreads 构建,没有 SharedArrayBuffer 无法初始化)。
  const ONNX_ISOLATED = typeof SharedArrayBuffer !== "undefined" && window.crossOriginIsolated === true;
  const ONNX_RUNTIME = {
    script: ONNX_ISOLATED ? "/vendor/ort/ort.webgpu.min.js" : "/vendor/ort/ort.wasm.min.js",
    wasmPath: "/vendor/ort/",
    executionProviders: ONNX_ISOLATED ? ["webgpu", "wasm"] : ["wasm"],
    initPromise: null,
  };
  const FACIAL_FEATURE_POINTS = [
    10, 21, 54, 67, 103, 109, 127, 132, 136, 148, 149, 150, 152, 172, 176, 234, 251, 284, 297, 323, 332, 338, 356,
    361, 365, 377, 378, 379, 389, 397, 454, 1, 2, 4, 5, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278,
    294, 327, 328, 344, 6, 33, 46, 52, 53, 55, 65, 66, 70, 105, 107, 133, 144, 145, 153, 154, 155, 157, 158, 159,
    160, 161, 163, 173, 190, 246, 249, 263, 276, 282, 283, 285, 295, 296, 300, 334, 336, 362, 373, 374, 380, 381,
    382, 384, 385, 386, 387, 388, 390, 398, 414, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 13, 14, 17,
    37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
    312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
  ];
  const EYE_PREVIEW_POINTS = {
    left: {
      outline: [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
      iris: [468, 469, 470, 471, 472],
    },
    right: {
      outline: [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
      iris: [473, 474, 475, 476, 477],
    },
  };
  const FACE_SURFACE_POINTS = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 48, 115, 220, 45, 275, 344, 278, 294, 205, 425, 50, 280, 187, 411, 200, 152];
  const MODEL_PATHS = {
    soham: "Driver-Monitoring-System/public/static/models/soham_best.onnx",
    chaitanya: "Driver-Monitoring-System/public/static/models/chaitanya_best.onnx",
    coco: "Driver-Monitoring-System/public/static/models/yolov8n_coco.onnx",
  };
  const MODEL_NAMES = {
    ensemble: "Dual Model Fusion",
    soham: "State Detection Model",
    chaitanya: "Action Object Model",
    coco: "COCO Phone Model",
  };
  const PHONE_OBJECT_MODEL = "coco";
  const MODEL_CLASSES = {
    coco: [
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
      "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
      "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
      "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
      "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
      "hair drier", "toothbrush",
    ],
    chaitanya: ["Cigarette", "Drinking", "Eating", "Phone", "Seatbelt"],
    soham: ["Distracted", "Drinking", "Drowsy", "Eating", "PhoneUse", "SafeDriving", "Seatbelt", "Smoking"],
  };
  const UNIFIED_CLASSES = {
    Cigarette: "smoking",
    Smoking: "smoking",
    Drinking: "drinking",
    Eating: "eating",
    Phone: "phone",
    PhoneUse: "phone",
    Seatbelt: "seatbelt",
    Distracted: "distracted",
    Drowsy: "drowsy",
    SafeDriving: "safe",
    "cell phone": "phone",
  };
  const CLASS_META = {
    safe: { label: "Normal Driving", color: "#22c55e", risk: false },
    phone: { label: "Phone Call", color: "#ef4444", risk: true },
    drowsy: { label: "Eye Closure", color: "#f59e0b", risk: true },
    head_down: { label: "Head Down", color: "#dc2626", risk: true },
    gaze_off: { label: "Gaze Offset", color: "#f43f5e", risk: true },
  };
  const ACTIVE_DETECTION_KEYS = new Set(["safe", "drowsy", "head_down", "gaze_off", "phone"]);
  const FACE_ALARM_KEYS = new Set(["drowsy", "head_down", "gaze_off"]);
  const ALARM_DETECTION_KEYS = new Set(["drowsy", "head_down", "gaze_off", "phone"]);
  const SECONDARY_STATUS_KEYS = new Set(["phone"]);
  const ANNOTATION_LABELS = {
    safe: "Normal Driving",
    phone: "Phone Call",
    drowsy: "Eye Closure",
    head_down: "Head Down",
    gaze_off: "Gaze Offset",
  };
  const ALERT_AUDIO_SOURCES = {
    drowsy: "严禁疲劳驾驶，请立即休息.mp3",
    gaze_off: "视线已偏离，请专注驾驶.mp3",
    phone: "禁止拨打电话，安全第一.mp3",
    head_down: "请立即抬头，观察前方.mp3",
  };
  const ALERT_AUDIO_PRIORITY = {
    drowsy: 4,
    head_down: 3,
    gaze_off: 2,
    phone: 1,
  };
  const MANUAL_CONTROL_PATH = "/api/manual-control";
  const MANUAL_CONTROL_POLL_MS = 800;
  // ---------- 原生 NPU 推理模式(页面/样式零改动, 只换推理来源) ----------
  // 打开方式: index.html?infer=native#/live-detection (缺省仍为浏览器推理, 随时可回退)
  const NATIVE_INFER_MODE = (new URLSearchParams(window.location.search).get("infer")
    || window.VISION_SENTINEL_INFER_MODE || "").toLowerCase() === "native";
  const NATIVE_SERVICE_BASE = (new URLSearchParams(window.location.search).get("nativeBase")
    || window.VISION_SENTINEL_NATIVE_BASE
    || `http://${window.location.hostname}:8600`).replace(/\/$/, "");
  const NATIVE_WS_URL = `ws://${new URL(NATIVE_SERVICE_BASE).host}/ws/dms/native`;
  const NATIVE_RECONNECT_MS = 3000;
  const MQTT_WS_URL = new URLSearchParams(window.location.search).get("mqttWs")
    || window.VISION_SENTINEL_MQTT_WS_URL
    || "wss://192.168.2.13:8084/mqtt";
  const MQTT_WS_TOPIC = new URLSearchParams(window.location.search).get("mqttTopic")
    || window.VISION_SENTINEL_MQTT_TOPIC
    || "vision-sentinel/control";
  const MQTT_WS_RECONNECT_MS = 3000;
  const MQTT_WS_KEEPALIVE_SECONDS = 30;
  const MANUAL_CONTROL_SHORTCUTS = {
    Digit0: { mode: "manual", key: "" },
    Digit1: { mode: "manual", key: "safe" },
    Digit2: { mode: "manual", key: "drowsy" },
    Digit3: { mode: "manual", key: "head_down" },
    Digit4: { mode: "manual", key: "gaze_off" },
    Digit5: { mode: "manual", key: "phone" },
    Digit9: { mode: "auto", key: "" },
  };

  const elements = {
    imageInput: document.querySelector("#inferenceImageInput"),
    cameraButton: document.querySelector("#cameraToggleBtn"),
    cameraVideo: document.querySelector("#cameraVideo"),
    canvas: document.querySelector("#detectionCanvas"),
    placeholder: document.querySelector("#detectionPlaceholder"),
    loading: document.querySelector("#inferenceLoading"),
    runtimeLabel: document.querySelector("#modelRuntimeLabel"),
    speed: document.querySelector("#inferenceSpeed"),
    modelSelect: document.querySelector("#inferenceModelSelect"),
    sourceStatus: document.querySelector("#modelSourceStatus"),
    modelReady: document.querySelector("#modelReadyText"),
    results: document.querySelector("#inferenceResults"),
    detectionCount: document.querySelector("#detectionCount"),
    timeline: document.querySelector("#detectionTimeline"),
    fatigueScore: document.querySelector("#fatigueScore"),
    fatigueLabel: document.querySelector("#fatigueLabel"),
    fatigueReading: document.querySelector("#fatigueScore")?.closest(".monitor-reading"),
    drowsyConfidence: document.querySelector("#drowsyConfidence"),
    attentionScore: document.querySelector("#attentionScore"),
    attentionLabel: document.querySelector("#attentionLabel"),
    attentionReading: document.querySelector("#attentionScore")?.closest(".monitor-reading"),
    safeConfidence: document.querySelector("#safeConfidence"),
    safeConfidenceBar: document.querySelector("#safeConfidenceBar"),
    labelSelect: document.querySelector("#labelSelect"),
    headState: document.querySelector("#headStateValue"),
    eyeState: document.querySelector("#eyeStateValue"),
    facePoseValue: document.querySelector("#facePoseValue"),
    gazeVectorValue: document.querySelector("#gazeVectorValue"),
    eyeVectorValue: document.querySelector("#eyeVectorValue"),
    faceVectorStatus: document.querySelector("#faceVectorStatus"),
    facePreview: document.querySelector("#facePreview"),
    faceIdValue: document.querySelector("#faceIdValue"),
    landmarkCountValue: document.querySelector("#landmarkCountValue"),
    driverActionValue: document.querySelector("#driverActionValue"),
    gazePitchValue: document.querySelector("#gazePitchValue"),
    gazeYawValue: document.querySelector("#gazeYawValue"),
    leftEyeOpenValue: document.querySelector("#leftEyeOpenValue"),
    rightEyeOpenValue: document.querySelector("#rightEyeOpenValue"),
    leftBlinkValue: document.querySelector("#leftBlinkValue"),
    rightBlinkValue: document.querySelector("#rightBlinkValue"),
    leftEyeWave: document.querySelector("#leftEyeWave"),
    rightEyeWave: document.querySelector("#rightEyeWave"),
    headXValue: document.querySelector("#headXValue"),
    headYValue: document.querySelector("#headYValue"),
    headZValue: document.querySelector("#headZValue"),
    headPitchValue: document.querySelector("#headPitchValue"),
    headYawValue: document.querySelector("#headYawValue"),
    headRollValue: document.querySelector("#headRollValue"),
    headPoseDial: document.querySelector("#headPoseDial"),
    leftEyePreview: document.querySelector("#leftEyePreview"),
    rightEyePreview: document.querySelector("#rightEyePreview"),
    leftEyeState: document.querySelector("#leftEyeState"),
    rightEyeState: document.querySelector("#rightEyeState"),
    eyeOpennessBar: document.querySelector("#eyeOpennessBar"),
    topVisionState: document.querySelector("#topVisionState"),
    topFaceCount: document.querySelector("#topFaceCount"),
    topRunTime: document.querySelector("#topRunTime"),
    topLatency: document.querySelector("#topLatency"),
    driverMonitorStatus: document.querySelector("#driverMonitorStatus"),
    driverTargetStatus: document.querySelector("#driverTargetStatus"),
    keyRegionPanel: document.querySelector("#keyRegionPanel"),
    keyRegionList: document.querySelector("#keyRegionList"),
    keyRegionCount: document.querySelector("#keyRegionCount"),
    statusItems: [...document.querySelectorAll(".dms-status-item[data-detection-key]")],
    recordButton: document.querySelector("#landmarkRecordBtn"),
  };

  if (!elements.canvas || !elements.cameraButton) return;

  const state = {
    sessions: { soham: null, chaitanya: null },
    stream: null,
    cameraRunning: false,
    inferencing: false,
    lastSource: null,
    timelineItems: [],
    lastTimelineKey: "",
    lastTimelineAt: 0,
    lastResultSignature: "",
    keyRegionSignature: "",
    keyRegionHoldUntil: 0,
    alertSince: new Map(),
    alertLastSeen: new Map(),
    alertHold: new Map(),
    alertAudio: {
      clips: new Map(),
      lastPlayedAt: new Map(),
      current: null,
      unlocking: null,
      unlocked: false,
      lastManualKey: "",
    },
    manualControl: {
      enabled: false,
      key: "",
      label: "",
      confidence: 0.99,
      source: "manual",
      updatedAt: 0,
      expiresAt: 0,
      lastSignature: "",
      pollTimer: null,
    },
    mqttControl: {
      socket: null,
      reconnectTimer: null,
      pingTimer: null,
      packetId: 1,
      connected: false,
      enabled: true,
    },
    annotationRecorder: {
      recording: false,
      startedAt: 0,
      lastCaptureAt: 0,
      frames: [],
      capture: null,
      lastCaptureFrame: null,
    },
    cameraStartedAt: 0,
    onnxBackend: "wasm",
    phoneRoiLastRunAt: 0,
    phoneRoiLastDetection: null,
    overlayCalibration: readOverlayCalibration(),
    headPose: {
      landmarker: null,
      initPromise: null,
      loadFailed: false,
      mode: "",
      downSince: 0,
      eyeClosedSince: 0,
      gazeOffSince: 0,
      gazeOffHoldUntil: 0,
      overlay: null,
      metrics: null,
      smoothedFaceBox: null,
      faceBoxLastSeenAt: 0,
      tracking: false,
      faceDetections: [],
      objectDetections: [],
      eyeHistory: { left: [], right: [] },
    },
    sixDRepNet: {
      available: null,
      disabled: false,
      pending: null,
      lastAt: 0,
      lastPose: null,
      consecutiveFailures: 0,
      message: "",
    },
    nativeInfer: {
      socket: null,
      connected: false,
      models: {},
      frame: [0, 0],
      inferenceMs: 0,
      reconnectTimer: null,
      img: null,
      streamCanvas: null,
      stream: null,
      drawTimer: null,
    },
  };
  const context = elements.canvas.getContext("2d");
  const preprocessCanvas = document.createElement("canvas");
  preprocessCanvas.width = INPUT_SIZE;
  preprocessCanvas.height = INPUT_SIZE;
  const preprocessContext = preprocessCanvas.getContext("2d", { willReadFrequently: true, alpha: false });
  const frameSampleCanvas = document.createElement("canvas");
  const frameSampleContext = frameSampleCanvas.getContext("2d", { willReadFrequently: true, alpha: false });
  const cameraFrameCanvas = document.createElement("canvas");
  const cameraFrameContext = cameraFrameCanvas.getContext("2d", { willReadFrequently: true, alpha: false });
  const cameraModelCanvas = document.createElement("canvas");
  const cameraModelContext = cameraModelCanvas.getContext("2d", { willReadFrequently: true, alpha: false });
  const modelInputData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const phoneRoiCanvas = document.createElement("canvas");
  phoneRoiCanvas.width = PHONE_ROI_INPUT_SIZE;
  phoneRoiCanvas.height = PHONE_ROI_INPUT_SIZE;
  const phoneRoiContext = phoneRoiCanvas.getContext("2d", { willReadFrequently: true, alpha: false });
  const phoneRoiData = new Float32Array(3 * PHONE_ROI_INPUT_SIZE * PHONE_ROI_INPUT_SIZE);

  function notify(message) {
    if (typeof window.showToast === "function") window.showToast(message);
  }

  function normalizeManualControlPayload(payload = {}) {
    const key = String(payload.key || "").trim();
    const enabled = payload.mode === "manual" || Boolean(payload.active || payload.enabled);
    return {
      enabled,
      key: enabled ? key : "",
      label: enabled && key ? payload.label || CLASS_META[key]?.label || key : "",
      confidence: Number.isFinite(Number(payload.confidence)) ? Number(payload.confidence) : 0.99,
      source: payload.source || "remote",
      updatedAt: Number(payload.updated_at || payload.updatedAt || 0),
      expiresAt: Number(payload.expires_at || payload.expiresAt || 0),
    };
  }

  function manualControlSignature(control = state.manualControl) {
    return `${control.enabled ? 1 : 0}:${control.key}:${Math.round((control.confidence || 0) * 100)}:${control.updatedAt}`;
  }

  function manualControlDetection() {
    const control = state.manualControl;
    if (!control.enabled || !ACTIVE_DETECTION_KEYS.has(control.key)) return null;
    return {
      key: control.key,
      confidence: Math.max(0, Math.min(control.confidence || 0.99, 1)),
      source: "manual-control",
      manualControl: true,
      alertActive: ALARM_DETECTION_KEYS.has(control.key),
    };
  }

  function applyManualControl(detections) {
    const detection = manualControlDetection();
    if (!state.manualControl.enabled) return detections;
    if (!detection) return detections;
    return [detection, ...detections.filter((item) => !item.manualControl)];
  }

  function syncManualControl(payload, announce = false) {
    const next = normalizeManualControlPayload(payload);
    const previousSignature = manualControlSignature();
    Object.assign(state.manualControl, next);
    const signature = manualControlSignature();
    if (signature === previousSignature) return;
    if (!next.enabled) {
      state.alertSince.clear();
      state.alertLastSeen.clear();
      state.alertHold.clear();
      state.alertAudio.lastManualKey = "";
      updateResults([]);
      if (announce) notify("Auto detection restored");
      return;
    }
    setLoading(false);
    elements.modelReady.textContent = "Manual";
    elements.runtimeLabel.textContent = next.key ? `Manual control · ${next.label || next.key}` : "Manual control";
    const detection = manualControlDetection();
    if (!detection) {
      state.alertSince.clear();
      state.alertLastSeen.clear();
      state.alertHold.clear();
      state.alertAudio.lastManualKey = "";
    } else if (!ALARM_DETECTION_KEYS.has(detection.key)) {
      state.alertAudio.lastManualKey = "";
    }
    updateResults(detection ? [detection] : []);
    if (!next.key) {
      return;
    } else if (announce) {
      notify(`Manual control: ${next.label || next.key}`);
    }
  }

  async function fetchManualControl() {
    try {
      const response = await fetch(MANUAL_CONTROL_PATH, { cache: "no-store" });
      if (!response.ok) return;
      syncManualControl(await response.json());
    } catch (error) {
      console.warn("Manual control polling failed", error);
    }
  }

  async function setManualControl(command, source = "keyboard") {
    const payload = typeof command === "object" && command
      ? { mode: command.mode || "manual", key: command.key || "", source }
      : { mode: command ? "manual" : "auto", key: command || "", source };
    try {
      const response = await fetch(MANUAL_CONTROL_PATH, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await response.text());
      syncManualControl(await response.json(), true);
    } catch (error) {
      notify(`Manual control failed: ${error.message}`);
    }
  }

  function mqttString(value) {
    const bytes = new TextEncoder().encode(String(value));
    return [bytes.length >> 8, bytes.length & 255, ...bytes];
  }

  function mqttRemainingLength(value) {
    const bytes = [];
    let next = value;
    do {
      let encoded = next % 128;
      next = Math.floor(next / 128);
      if (next > 0) encoded |= 128;
      bytes.push(encoded);
    } while (next > 0);
    return bytes;
  }

  function mqttPacket(typeAndFlags, body) {
    return new Uint8Array([typeAndFlags, ...mqttRemainingLength(body.length), ...body]);
  }

  function mqttConnectPacket(clientId) {
    const body = [
      ...mqttString("MQTT"),
      4,
      2,
      MQTT_WS_KEEPALIVE_SECONDS >> 8,
      MQTT_WS_KEEPALIVE_SECONDS & 255,
      ...mqttString(clientId),
    ];
    return mqttPacket(0x10, body);
  }

  function mqttSubscribePacket(topic) {
    const id = state.mqttControl.packetId;
    state.mqttControl.packetId = id >= 65535 ? 1 : id + 1;
    return mqttPacket(0x82, [id >> 8, id & 255, ...mqttString(topic), 0]);
  }

  function mqttPingPacket() {
    return new Uint8Array([0xc0, 0]);
  }

  function mqttReadRemainingLength(bytes, offset) {
    let multiplier = 1;
    let value = 0;
    let cursor = offset;
    let encoded = 0;
    do {
      if (cursor >= bytes.length) return null;
      encoded = bytes[cursor++];
      value += (encoded & 127) * multiplier;
      multiplier *= 128;
    } while ((encoded & 128) !== 0);
    return { value, cursor };
  }

  function mqttDecodePublish(bytes, offset, packetEnd, flags) {
    if (offset + 2 > packetEnd) return null;
    const topicLength = (bytes[offset] << 8) | bytes[offset + 1];
    let cursor = offset + 2;
    if (cursor + topicLength > packetEnd) return null;
    const topic = new TextDecoder().decode(bytes.slice(cursor, cursor + topicLength));
    cursor += topicLength;
    const qos = (flags >> 1) & 3;
    if (qos > 0) cursor += 2;
    if (cursor > packetEnd) return null;
    return {
      topic,
      payload: new TextDecoder().decode(bytes.slice(cursor, packetEnd)),
    };
  }

  function parseMqttCommand(text) {
    const rawText = String(text || "").trim();
    const jsonText = rawText
      .replace(/[“”]/g, "\"")
      .replace(/\\_/g, "_");
    let data = {};
    try {
      data = JSON.parse(jsonText);
    } catch {
      data = { command: jsonText };
    }
    const raw = String(data.command || data.key || data.state || data.status || "").trim().toLowerCase();
    const command = raw.replace(/\\/g, "").replace(/[-\s]+/g, "_");
    const aliases = {
      manual_enter: { mode: "manual", key: "" },
      enter: { mode: "manual", key: "" },
      manual: { mode: "manual", key: "" },
      clear: { mode: "manual", key: "" },
      manual_exit: { mode: "auto", key: "" },
      exit: { mode: "auto", key: "" },
      auto: { mode: "auto", key: "" },
      safe: { mode: "manual", key: "safe" },
      normal: { mode: "manual", key: "safe" },
      drowsy: { mode: "manual", key: "drowsy" },
      eye_closure: { mode: "manual", key: "drowsy" },
      head_down: { mode: "manual", key: "head_down" },
      gaze_off: { mode: "manual", key: "gaze_off" },
      phone: { mode: "manual", key: "phone" },
    };
    if (data.mode || ACTIVE_DETECTION_KEYS.has(command)) {
      return {
        mode: data.mode || "manual",
        key: ACTIVE_DETECTION_KEYS.has(command) ? command : String(data.key || ""),
      };
    }
    return aliases[command] || null;
  }

  function handleMqttPacket(bytes) {
    let offset = 0;
    while (offset < bytes.length) {
      const header = bytes[offset++];
      const length = mqttReadRemainingLength(bytes, offset);
      if (!length) return;
      offset = length.cursor;
      const packetEnd = offset + length.value;
      if (packetEnd > bytes.length) return;
      const packetType = header >> 4;
      if (packetType === 2) {
        state.mqttControl.connected = true;
        state.mqttControl.socket?.send(mqttSubscribePacket(MQTT_WS_TOPIC));
        notify("MQTT WebSocket connected");
      } else if (packetType === 3) {
        const message = mqttDecodePublish(bytes, offset, packetEnd, header & 15);
        if (message?.topic === MQTT_WS_TOPIC) {
          const command = parseMqttCommand(message.payload);
          if (command) setManualControl(command, "mqtt-websocket");
          else console.warn("Ignored MQTT command", message.payload);
        }
      }
      offset = packetEnd;
    }
  }

  function stopMqttWebSocketControl() {
    if (state.mqttControl.reconnectTimer) window.clearTimeout(state.mqttControl.reconnectTimer);
    if (state.mqttControl.pingTimer) window.clearInterval(state.mqttControl.pingTimer);
    state.mqttControl.reconnectTimer = null;
    state.mqttControl.pingTimer = null;
    state.mqttControl.connected = false;
    if (state.mqttControl.socket) {
      state.mqttControl.socket.onopen = null;
      state.mqttControl.socket.onmessage = null;
      state.mqttControl.socket.onclose = null;
      state.mqttControl.socket.onerror = null;
      state.mqttControl.socket.close();
      state.mqttControl.socket = null;
    }
  }

  function scheduleMqttReconnect() {
    if (!state.mqttControl.enabled || state.mqttControl.reconnectTimer) return;
    state.mqttControl.reconnectTimer = window.setTimeout(() => {
      state.mqttControl.reconnectTimer = null;
      startMqttWebSocketControl();
    }, MQTT_WS_RECONNECT_MS);
  }

  function startMqttWebSocketControl() {
    if (!MQTT_WS_URL || MQTT_WS_URL === "off" || !("WebSocket" in window)) return;
    stopMqttWebSocketControl();
    let socket = null;
    try {
      socket = new WebSocket(MQTT_WS_URL, "mqtt");
    } catch (error) {
      console.warn(`MQTT WebSocket failed to open: ${MQTT_WS_URL}`, error);
      notify(`MQTT WebSocket failed: ${error.message}`);
      return;
    }
    state.mqttControl.socket = socket;
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      const clientId = `vision-sentinel-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
      socket.send(mqttConnectPacket(clientId));
      state.mqttControl.pingTimer = window.setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) socket.send(mqttPingPacket());
      }, MQTT_WS_KEEPALIVE_SECONDS * 500);
    };
    socket.onmessage = async (event) => {
      let buffer = null;
      if (event.data instanceof ArrayBuffer) {
        buffer = event.data;
      } else if (typeof Blob !== "undefined" && event.data instanceof Blob) {
        buffer = await event.data.arrayBuffer();
      } else if (typeof event.data === "string") {
        buffer = new TextEncoder().encode(event.data).buffer;
      }
      if (buffer) handleMqttPacket(new Uint8Array(buffer));
    };
    socket.onclose = () => {
      if (state.mqttControl.socket === socket) state.mqttControl.socket = null;
      state.mqttControl.connected = false;
      if (state.mqttControl.pingTimer) window.clearInterval(state.mqttControl.pingTimer);
      state.mqttControl.pingTimer = null;
      scheduleMqttReconnect();
    };
    socket.onerror = () => {
      socket.close();
    };
  }

  function installManualControlShortcuts() {
    document.addEventListener("keydown", (event) => {
      if (!event.altKey || event.metaKey || event.ctrlKey || event.shiftKey) return;
      if (!(event.code in MANUAL_CONTROL_SHORTCUTS)) return;
      const target = event.target;
      if (target?.closest?.("input, textarea, select, [contenteditable='true']")) return;
      event.preventDefault();
      setManualControl(MANUAL_CONTROL_SHORTCUTS[event.code], "keyboard");
    });
  }

  function startManualControlPolling() {
    fetchManualControl();
    state.manualControl.pollTimer = window.setInterval(fetchManualControl, MANUAL_CONTROL_POLL_MS);
  }

  function localServerHintUrl() {
    const hash = window.location.hash || "#live-detection";
    const protocol = window.location.protocol === "http:" ? "http" : "https";
    const port = window.location.port || (protocol === "https" ? "8444" : "8000");
    return `${protocol}://127.0.0.1:${port}/index.html${hash}`;
  }

  window.visionSentinelOverlayCalibration = {
    get: () => ({ ...state.overlayCalibration }),
    set: (value = {}) => {
      const calibration = saveOverlayCalibration({ ...state.overlayCalibration, ...value });
      notify(
        `Overlay calibration X:${calibration.offsetX} Y:${calibration.offsetY} SX:${calibration.scaleX} SY:${calibration.scaleY}`,
      );
      return calibration;
    },
    reset: () => {
      const calibration = saveOverlayCalibration(DEFAULT_OVERLAY_CALIBRATION);
      notify("Overlay calibration reset");
      return calibration;
    },
  };

  function setLoading(active, text = "Loading dual models...") {
    elements.loading.hidden = !active;
    const label = elements.loading.querySelector("b");
    if (label) label.textContent = text;
  }

  function normalizeOverlayCalibration(value = {}) {
    const offsetX = Number(value.offsetX);
    const offsetY = Number(value.offsetY);
    const scaleX = Number(value.scaleX);
    const scaleY = Number(value.scaleY);
    return {
      offsetX: Number.isFinite(offsetX) ? offsetX : DEFAULT_OVERLAY_CALIBRATION.offsetX,
      offsetY: Number.isFinite(offsetY) ? offsetY : DEFAULT_OVERLAY_CALIBRATION.offsetY,
      scaleX: Number.isFinite(scaleX) && scaleX > 0 ? scaleX : DEFAULT_OVERLAY_CALIBRATION.scaleX,
      scaleY: Number.isFinite(scaleY) && scaleY > 0 ? scaleY : DEFAULT_OVERLAY_CALIBRATION.scaleY,
    };
  }

  function readOverlayCalibration() {
    try {
      return normalizeOverlayCalibration(JSON.parse(localStorage.getItem(OVERLAY_CALIBRATION_STORAGE_KEY) || "{}"));
    } catch {
      return { ...DEFAULT_OVERLAY_CALIBRATION };
    }
  }

  function saveOverlayCalibration(value) {
    const calibration = normalizeOverlayCalibration(value);
    localStorage.setItem(OVERLAY_CALIBRATION_STORAGE_KEY, JSON.stringify(calibration));
    state.overlayCalibration = calibration;
    return calibration;
  }

  function selectedModels() {
    const value = elements.modelSelect.value;
    const models = value === "ensemble" ? ["soham", "chaitanya"] : [value];
    return models.includes(PHONE_OBJECT_MODEL) ? models : [...models, PHONE_OBJECT_MODEL];
  }

  function confidenceThresholdForKey(key) {
    return key === "phone" ? PHONE_CONFIDENCE_THRESHOLD : CONFIDENCE_THRESHOLD;
  }

  // 疲劳报警以关键点眼部状态为权威:模型来源的 drowsy 在睁眼时一律抑制,
  // 闭眼时也要求更高置信度;关键点路径(完全闭眼持续 1 秒)不受影响。
  function suppressModelDrowsy(detections) {
    const eyeConfidence = state.headPose.metrics?.eye?.confidence;
    const eyesVisiblyOpen = typeof eyeConfidence === "number" && eyeConfidence < EYE_CLOSED_CONFIDENCE_THRESHOLD;
    return detections.filter((detection) => {
      if (detection.key !== "drowsy" || detection.source === "face-vector") return true;
      if (eyesVisiblyOpen) return false;
      return detection.confidence >= DROWSY_MODEL_CONFIDENCE_THRESHOLD;
    });
  }

  // ===================== 原生 NPU 推理模式(浏览器侧) =====================
  // 结果语义与浏览器 parseDetections 输出同构; 阈值过滤/统一 key/phone 归属/
  // NMS/drowsy 抑制/手动控制/报警时序全部沿用浏览器原逻辑, 保证显示一致。

  function nativeObjectDetections() {
    const native = state.nativeInfer;
    const dimensions = sourceDimensions(elements.cameraVideo);
    if (!dimensions.width || !dimensions.height) return [];
    const selected = selectedModels();
    const detections = [];
    for (const modelName of Object.keys(native.models)) {
      if (!selected.includes(modelName)) continue;
      for (const det of native.models[modelName]) {
        const originalClass = det.originalClass;
        const key = UNIFIED_CLASSES[originalClass] || originalClass;
        if (!ACTIVE_DETECTION_KEYS.has(key)) continue;
        // COCO 是手机检测的唯一来源，屏蔽旧模型的高误报 phone 输出
        if (key === "phone" && modelName !== PHONE_OBJECT_MODEL) continue;
        if (det.confidence < confidenceThresholdForKey(key)) continue;
        detections.push({
          key,
          originalClass,
          confidence: det.confidence,
          source: modelName,
          box: [
            det.box[0] * dimensions.width,
            det.box[1] * dimensions.height,
            det.box[2] * dimensions.width,
            det.box[3] * dimensions.height,
          ],
        });
      }
    }
    return detections;
  }

  function startNativeWebSocket() {
    const native = state.nativeInfer;
    if (native.reconnectTimer) window.clearTimeout(native.reconnectTimer);
    native.reconnectTimer = null;
    try {
      native.socket = new WebSocket(NATIVE_WS_URL);
    } catch (error) {
      console.warn("Native WS failed to open", error);
      scheduleNativeReconnect();
      return;
    }
    native.socket.onopen = () => {
      native.connected = true;
      if (!state.manualControl.enabled) {
        elements.modelReady.textContent = "Native NPU";
        elements.runtimeLabel.textContent = `${MODEL_NAMES[elements.modelSelect.value]} ready · NPU native`;
      }
    };
    native.socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type !== "detections") return;
        native.models = payload.models || {};
        native.frame = payload.frame || [0, 0];
        native.inferenceMs = payload.inference_ms || 0;
      } catch (error) {
        console.warn("Native WS payload parse failed", error);
      }
    };
    native.socket.onclose = () => {
      native.connected = false;
      scheduleNativeReconnect();
    };
    native.socket.onerror = () => native.socket?.close();
  }

  function scheduleNativeReconnect() {
    const native = state.nativeInfer;
    if (native.reconnectTimer || !NATIVE_INFER_MODE) return;
    native.reconnectTimer = window.setTimeout(() => {
      native.reconnectTimer = null;
      startNativeWebSocket();
    }, NATIVE_RECONNECT_MS);
  }

  function stopNativeWebSocket() {
    const native = state.nativeInfer;
    if (native.reconnectTimer) window.clearTimeout(native.reconnectTimer);
    native.reconnectTimer = null;
    native.connected = false;
    if (native.socket) {
      native.socket.onopen = null;
      native.socket.onmessage = null;
      native.socket.onclose = null;
      native.socket.onerror = null;
      native.socket.close();
      native.socket = null;
    }
  }

  async function startNativeCameraStream() {
    const native = state.nativeInfer;
    // 隐藏 canvas 接收 MJPEG 帧 -> captureStream 喂给现有 #cameraVideo:
    // 视频显示/object-fit 映射/MediaPipe VIDEO 模式/关键区域裁切全部原样工作
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d", { alpha: false });
    const img = new Image();
    let firstFrame = null;
    let firstFrameReject = null;
    const firstFramePromise = new Promise((resolve, reject) => {
      firstFrame = resolve;
      firstFrameReject = reject;
    });
    window.setTimeout(() => firstFrameReject?.(new Error("MJPEG 超时(>15s): 检查服务与摄像头")), 15000);
    // 注意: <img> 对 multipart MJPEG 只在流开始时触发一次 onload,
    // 后续帧不会重复触发 —— 必须用绘制循环持续把 img 当前解码帧画进 canvas,
    // captureStream 才有连续帧(否则视频冻结在第一帧)。
    img.onload = () => {
      if (img.naturalWidth) firstFrame?.();
    };
    img.onerror = () => firstFrameReject?.(new Error("MJPEG 流中断: 检查原生服务 /video.mjpg"));
    // 跨源(8000 页面 -> 8600 MJPEG)必须声明 anonymous, 服务端已发 ACAO:*;
    // 否则 canvas 非 origin-clean, captureStream 会抛异常
    img.crossOrigin = "anonymous";
    img.src = `${NATIVE_SERVICE_BASE}/video.mjpg`;
    native.img = img;
    native.streamCanvas = canvas;
    const drawTimer = window.setInterval(() => {
      if (!img.naturalWidth) return;
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
      }
      context.drawImage(img, 0, 0, canvas.width, canvas.height);
    }, 33); // ~30fps, 与摄像头帧率对齐
    native.drawTimer = drawTimer;
    await firstFramePromise;
    native.stream = canvas.captureStream(30);
    elements.cameraVideo.srcObject = native.stream;
    await elements.cameraVideo.play();
  }

  function stopNativeCameraStream() {
    const native = state.nativeInfer;
    if (native.drawTimer) {
      window.clearInterval(native.drawTimer);
      native.drawTimer = null;
    }
    if (native.img) {
      native.img.onload = null;
      native.img.onerror = null;
      native.img.src = "";
      native.img = null;
    }
    native.stream?.getTracks().forEach((track) => track.stop());
    native.stream = null;
    native.streamCanvas = null;
  }

  function postNativeAlert(key) {
    fetch(`${NATIVE_SERVICE_BASE}/alert`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    }).catch((error) => console.warn("Native alert post failed", error));
  }

  async function verifyModelFiles() {
    if (NATIVE_INFER_MODE) {
      try {
        const response = await fetch(`${NATIVE_SERVICE_BASE}/health`, { cache: "no-store" });
        const health = response.ok ? await response.json() : null;
        if (!health?.ok) throw new Error("service not ok");
        elements.sourceStatus.textContent = `Native NPU service · ${health.camera ? "camera live" : "camera pending"}`;
        elements.modelReady.textContent = "Native Ready";
      } catch (error) {
        elements.sourceStatus.textContent = "Native NPU service unreachable";
        elements.sourceStatus.classList.add("error");
        elements.modelReady.textContent = "Service Offline";
      }
      return;
    }
    if (window.location.protocol === "file:") {
      elements.sourceStatus.textContent = "Open with local server";
      elements.sourceStatus.classList.add("error");
      elements.modelReady.textContent = "Server Required";
      return;
    }
    try {
      const [responses, sixDStatus] = await Promise.all([
        Promise.all(Object.values(MODEL_PATHS).map((path) => fetch(path, { method: "HEAD" }))),
        fetch(SIXDREPNET_STATUS_PATH, { cache: "no-store" })
          .then((response) => (response.ok ? response.json() : null))
          .catch(() => null),
      ]);
      if (!responses.every((response) => response.ok)) throw new Error("Model files are incomplete");
      state.sixDRepNet.available = Boolean(sixDStatus?.weights_found);
      state.sixDRepNet.message = sixDStatus?.weights_found ? "6DRepNet connected" : "6DRepNet weights pending";
      elements.sourceStatus.textContent = `Local weights found · ${state.sixDRepNet.message}`;
      elements.modelReady.textContent = "Ready to Load";
    } catch (error) {
      elements.sourceStatus.textContent = "Local models not found";
      elements.sourceStatus.classList.add("error");
      elements.modelReady.textContent = "Missing Weights";
    }
  }

  async function ensureOnnxRuntime() {
    if (window.ort) return window.ort;
    if (window.location.protocol === "file:") {
      throw new Error(`当前是 file:// 直接打开，无法加载本地 ONNX Runtime 和模型资源。请在终端运行 npm run start:https，然后打开 ${localServerHintUrl()}`);
    }
    if (!ONNX_RUNTIME.initPromise) {
      ONNX_RUNTIME.initPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = ONNX_RUNTIME.script;
        script.async = true;
        script.onload = () => (window.ort ? resolve(window.ort) : reject(new Error("ONNX runtime did not initialize")));
        script.onerror = () =>
          reject(new Error(`ONNX Runtime 加载失败。请确认通过本地服务打开页面，并检查 ${ONNX_RUNTIME.script} 是否能访问。`));
        document.head.append(script);
      }).catch((error) => {
        ONNX_RUNTIME.initPromise = null;
        throw error;
      });
    }
    return ONNX_RUNTIME.initPromise;
  }

  async function ensureModels() {
    if (NATIVE_INFER_MODE && state.cameraRunning) {
      // native 模式: YOLO 在板端 NPU, 浏览器不加载 ONNX
      if (!state.nativeInfer.socket) startNativeWebSocket();
      elements.modelReady.textContent = state.nativeInfer.connected ? "Native NPU" : "Native connecting";
      return;
    }
    await ensureOnnxRuntime();
    ort.env.wasm.numThreads = window.crossOriginIsolated ? Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 4)) : 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = ONNX_RUNTIME.wasmPath;

    const targets = selectedModels();
    for (let index = 0; index < targets.length; index += 1) {
      const name = targets[index];
      if (state.sessions[name]) continue;
      setLoading(false);
      elements.modelReady.textContent = `Loading ${index + 1}/${targets.length}`;
      elements.runtimeLabel.textContent = `Loading ${MODEL_NAMES[name]} in background`;
      try {
        state.sessions[name] = await ort.InferenceSession.create(MODEL_PATHS[name], {
          executionProviders: ONNX_RUNTIME.executionProviders,
          graphOptimizationLevel: "all",
        });
        if (ONNX_RUNTIME.executionProviders.includes("webgpu") && navigator.gpu) state.onnxBackend = "webgpu";
      } catch (error) {
        console.warn(`WebGPU unavailable for ${name}, falling back to WASM`, error);
        state.onnxBackend = "wasm";
        state.sessions[name] = await ort.InferenceSession.create(MODEL_PATHS[name], {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
      }
      if (state.manualControl.enabled) {
        elements.runtimeLabel.textContent = state.manualControl.key
          ? `Manual control · ${state.manualControl.label || state.manualControl.key} · live detection running`
          : "Manual control · live detection running";
      }
    }
    elements.modelReady.textContent = "Ready";
    elements.runtimeLabel.textContent = state.manualControl.enabled
      ? state.manualControl.key
        ? `Manual control · ${state.manualControl.label || state.manualControl.key} · live detection running`
        : "Manual control · live detection running"
      : `${MODEL_NAMES[elements.modelSelect.value]} ready · ${state.onnxBackend === "webgpu" ? "WebGPU" : "WASM (CPU)"}`;
    setLoading(false);
  }

  function loadImage(file) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      const url = URL.createObjectURL(file);
      image.onload = () => {
        URL.revokeObjectURL(url);
        resolve(image);
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Image read failed"));
      };
      image.src = url;
    });
  }

  function sourceDimensions(source) {
    return {
      width: source.videoWidth || source.naturalWidth || source.width,
      height: source.videoHeight || source.naturalHeight || source.height,
    };
  }

  function objectPositionFactors(value = "50% 50%") {
    const parts = value.trim().split(/\s+/);
    const parsePart = (part, axis) => {
      if (part === "left" || part === "top") return 0;
      if (part === "right" || part === "bottom") return 1;
      if (part === "center") return 0.5;
      if (part?.endsWith("%")) return clamp(Number.parseFloat(part) / 100);
      return axis === "x" ? 0.5 : 0.5;
    };
    return {
      x: parsePart(parts[0], "x"),
      y: parsePart(parts[1] || parts[0], "y"),
    };
  }

  function renderedMediaLayout(source, targetWidth, targetHeight, fit = "contain", objectPosition = "50% 50%") {
    const dimensions = sourceDimensions(source);
    const safeWidth = Math.max(1, dimensions.width);
    const safeHeight = Math.max(1, dimensions.height);
    const normalizedFit = fit === "cover" ? "cover" : "contain";
    const scale =
      normalizedFit === "cover"
        ? Math.max(targetWidth / safeWidth, targetHeight / safeHeight)
        : Math.min(targetWidth / safeWidth, targetHeight / safeHeight);
    const renderedWidth = safeWidth * scale;
    const renderedHeight = safeHeight * scale;
    const position = objectPositionFactors(objectPosition);

    return {
      scale,
      offsetX: (targetWidth - renderedWidth) * position.x,
      offsetY: (targetHeight - renderedHeight) * position.y,
    };
  }

  function cameraObjectFit() {
    const fit = window.getComputedStyle(elements.cameraVideo).objectFit;
    return fit === "cover" ? "cover" : "contain";
  }

  function cameraObjectPosition() {
    return window.getComputedStyle(elements.cameraVideo).objectPosition || "50% 50%";
  }

  function cameraOverlayMirrorX() {
    const transform = window.getComputedStyle(elements.cameraVideo).transform;
    const matrix = transform?.match(/^matrix\(([^)]+)\)$/);
    const matrix3d = transform?.match(/^matrix3d\(([^)]+)\)$/);
    const scaleX = matrix
      ? Number.parseFloat(matrix[1].split(",")[0])
      : matrix3d
        ? Number.parseFloat(matrix3d[1].split(",")[0])
        : 1;
    return CAMERA_OVERLAY_MIRRORED || scaleX < 0;
  }

  function cameraMediaLayout(source, canvasWidth, canvasHeight) {
    const stage = document.querySelector("#detectionStage");
    const stageRect = stage?.getBoundingClientRect();
    const videoRect = elements.cameraVideo.getBoundingClientRect();
    if (!stageRect?.width || !stageRect?.height || !videoRect.width || !videoRect.height) {
      return renderedMediaLayout(source, canvasWidth, canvasHeight, cameraObjectFit(), cameraObjectPosition());
    }
    const cssToCanvasX = canvasWidth / stageRect.width;
    const cssToCanvasY = canvasHeight / stageRect.height;
    const layout = renderedMediaLayout(
      source,
      videoRect.width * cssToCanvasX,
      videoRect.height * cssToCanvasY,
      cameraObjectFit(),
      cameraObjectPosition(),
    );
    return {
      scale: layout.scale,
      offsetX: (videoRect.left - stageRect.left) * cssToCanvasX + layout.offsetX,
      offsetY: (videoRect.top - stageRect.top) * cssToCanvasY + layout.offsetY,
    };
  }

  function ensureCameraCanvasSize() {
    const stage = document.querySelector("#detectionStage");
    const pixelRatio = 1;
    const width = Math.max(1, Math.round(Math.min(stage?.clientWidth || elements.cameraVideo.clientWidth || 1, CAMERA_RENDER_MAX_WIDTH) * pixelRatio));
    const height = Math.max(1, Math.round(Math.min(stage?.clientHeight || elements.cameraVideo.clientHeight || 1, CAMERA_RENDER_MAX_HEIGHT) * pixelRatio));
    if (elements.canvas.width !== width || elements.canvas.height !== height) {
      elements.canvas.width = width;
      elements.canvas.height = height;
    }
    if (cameraFrameCanvas.width !== width || cameraFrameCanvas.height !== height) {
      cameraFrameCanvas.width = width;
      cameraFrameCanvas.height = height;
    }
    if (cameraModelCanvas.width !== width || cameraModelCanvas.height !== height) {
      cameraModelCanvas.width = width;
      cameraModelCanvas.height = height;
    }
    return { width, height };
  }

  function drawCameraFrame(targetContext, source, width, height, offsetX, offsetY, renderedWidth, renderedHeight, mirrorX, filter = "none") {
    targetContext.save();
    targetContext.setTransform(1, 0, 0, 1, 0, 0);
    targetContext.clearRect(0, 0, width, height);
    targetContext.fillStyle = "#050817";
    targetContext.fillRect(0, 0, width, height);
    if (mirrorX) {
      targetContext.translate(width, 0);
      targetContext.scale(-1, 1);
    }
    targetContext.filter = filter;
    targetContext.drawImage(source, offsetX, offsetY, renderedWidth, renderedHeight);
    targetContext.filter = "none";
    targetContext.restore();
  }

  function prepareCameraDisplayFrame(source) {
    const { width, height } = ensureCameraCanvasSize();
    const dimensions = sourceDimensions(source);
    if (!dimensions.width || !dimensions.height) return null;
    const { scale, offsetX, offsetY } = cameraMediaLayout(source, width, height);
    const renderedWidth = dimensions.width * scale;
    const renderedHeight = dimensions.height * scale;
    const mirrorX = cameraOverlayMirrorX();
    drawCameraFrame(cameraModelContext, source, width, height, offsetX, offsetY, renderedWidth, renderedHeight, false);
    drawCameraFrame(
      cameraFrameContext,
      source,
      width,
      height,
      offsetX,
      offsetY,
      renderedWidth,
      renderedHeight,
      mirrorX,
      "grayscale(1) contrast(1.08) brightness(1.02)",
    );
    return cameraFrameCanvas;
  }

  function clamp(value, min = 0, max = 1) {
    return Math.min(max, Math.max(min, value));
  }

  function normalizeSourceBox(box, dimensions) {
    if (!box?.length || !dimensions.width || !dimensions.height) return null;
    const x1 = clamp(Math.min(box[0], box[2]), 0, dimensions.width);
    const y1 = clamp(Math.min(box[1], box[3]), 0, dimensions.height);
    const x2 = clamp(Math.max(box[0], box[2]), 0, dimensions.width);
    const y2 = clamp(Math.max(box[1], box[3]), 0, dimensions.height);
    return x2 > x1 && y2 > y1 ? [x1, y1, x2, y2] : null;
  }

  function faceBoxShiftRatio(previous, next) {
    const previousWidth = Math.max(1, previous[2] - previous[0]);
    const previousHeight = Math.max(1, previous[3] - previous[1]);
    const nextWidth = Math.max(1, next[2] - next[0]);
    const nextHeight = Math.max(1, next[3] - next[1]);
    const previousCenterX = (previous[0] + previous[2]) / 2;
    const previousCenterY = (previous[1] + previous[3]) / 2;
    const nextCenterX = (next[0] + next[2]) / 2;
    const nextCenterY = (next[1] + next[3]) / 2;
    const centerShift = Math.hypot(nextCenterX - previousCenterX, nextCenterY - previousCenterY) / Math.hypot(previousWidth, previousHeight);
    const sizeShift =
      Math.abs(nextWidth - previousWidth) / Math.max(previousWidth, nextWidth) +
      Math.abs(nextHeight - previousHeight) / Math.max(previousHeight, nextHeight);
    return centerShift + sizeShift * 0.35;
  }

  function stabilizeFaceBox(faceBox, dimensions, now = Date.now()) {
    const next = normalizeSourceBox(faceBox, dimensions);
    if (!next) return null;
    const previous = state.headPose.smoothedFaceBox;
    const freshPrevious = previous && now - state.headPose.faceBoxLastSeenAt <= FACE_BOX_HOLD_MS;
    const shouldSmooth = freshPrevious && faceBoxShiftRatio(previous, next) <= FACE_BOX_RESET_SHIFT;
    const stable = shouldSmooth
      ? previous.map((value, index) => value * (1 - FACE_BOX_SMOOTHING_ALPHA) + next[index] * FACE_BOX_SMOOTHING_ALPHA)
      : next;
    state.headPose.smoothedFaceBox = normalizeSourceBox(stable, dimensions);
    state.headPose.faceBoxLastSeenAt = now;
    return state.headPose.smoothedFaceBox;
  }

  function recentStableFaceBox(now = Date.now()) {
    if (!state.headPose.smoothedFaceBox || now - state.headPose.faceBoxLastSeenAt > FACE_BOX_HOLD_MS) return null;
    return [...state.headPose.smoothedFaceBox];
  }

  function cachedPhoneRoiDetections(now = Date.now()) {
    const detection = state.phoneRoiLastDetection;
    if (!detection) return [];
    if (detection.heldUntil <= now) {
      state.phoneRoiLastDetection = null;
      return [];
    }
    return [{ ...detection, retainedPhoneRoi: true }];
  }

  function distance(first, second) {
    return Math.hypot(first.x - second.x, first.y - second.y);
  }

  function averageLandmark(landmarks, indexes) {
    const points = indexes.map((index) => landmarks[index]).filter(Boolean);
    if (!points.length) return null;
    return points.reduce(
      (result, point) => ({
        x: result.x + point.x / points.length,
        y: result.y + point.y / points.length,
        z: result.z + (point.z || 0) / points.length,
      }),
      { x: 0, y: 0, z: 0 },
    );
  }

  function faceBoxFromLandmarks(landmarks, dimensions) {
    const xs = landmarks.map((point) => point.x * dimensions.width);
    const ys = landmarks.map((point) => point.y * dimensions.height);
    const left = Math.min(...xs);
    const top = Math.min(...ys);
    const right = Math.max(...xs);
    const bottom = Math.max(...ys);
    const padding = Math.max(right - left, bottom - top) * 0.12;
    return [
      clamp(left - padding, 0, dimensions.width),
      clamp(top - padding, 0, dimensions.height),
      clamp(right + padding, 0, dimensions.width),
      clamp(bottom + padding, 0, dimensions.height),
    ];
  }

  function cropFaceDataUrl(source, faceBox) {
    const dimensions = sourceDimensions(source);
    if (!faceBox?.length || !dimensions.width || !dimensions.height) return "";
    const [left, top, right, bottom] = faceBox;
    const boxWidth = Math.max(1, right - left);
    const boxHeight = Math.max(1, bottom - top);
    const side = Math.max(boxWidth, boxHeight);
    const centerX = (left + right) / 2;
    const centerY = (top + bottom) / 2;
    const sourceX = clamp(centerX - side / 2, 0, Math.max(0, dimensions.width - side));
    const sourceY = clamp(centerY - side / 2, 0, Math.max(0, dimensions.height - side));
    const cropSize = Math.min(side, dimensions.width - sourceX, dimensions.height - sourceY);
    if (cropSize <= 1) return "";

    const canvas = document.createElement("canvas");
    canvas.width = SIXDREPNET_FACE_CROP_SIZE;
    canvas.height = SIXDREPNET_FACE_CROP_SIZE;
    const cropContext = canvas.getContext("2d", { alpha: false });
    cropContext.drawImage(source, sourceX, sourceY, cropSize, cropSize, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.82);
  }

  async function ensureSixDRepNetAvailable() {
    if (state.sixDRepNet.disabled) return false;
    if (state.sixDRepNet.available !== null) return state.sixDRepNet.available;
    try {
      const response = await fetch(SIXDREPNET_STATUS_PATH, { cache: "no-store" });
      const status = response.ok ? await response.json() : null;
      state.sixDRepNet.available = Boolean(status?.weights_found);
      state.sixDRepNet.message = status?.weights_found ? "6DRepNet connected" : "6DRepNet weights pending";
      if (!state.sixDRepNet.available) state.sixDRepNet.disabled = true;
      return state.sixDRepNet.available;
    } catch {
      state.sixDRepNet.available = false;
      state.sixDRepNet.disabled = true;
      state.sixDRepNet.message = "6DRepNet backend not ready";
      return false;
    }
  }

  async function requestSixDRepNetPose(source, faceBox) {
    if (!(await ensureSixDRepNetAvailable())) return null;
    const now = performance.now();
    if (state.sixDRepNet.lastPose && now - state.sixDRepNet.lastAt < SIXDREPNET_MIN_INTERVAL_MS) {
      return state.sixDRepNet.lastPose;
    }
    if (state.sixDRepNet.pending) return state.sixDRepNet.pending;

    const image = cropFaceDataUrl(source, faceBox);
    if (!image) return null;
    state.sixDRepNet.pending = fetch(SIXDREPNET_INFERENCE_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    })
      .then(async (response) => {
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          throw new Error(detail.error || "6DRepNet inference failed");
        }
        return response.json();
      })
      .then((pose) => {
        const normalized = {
          pitch: Number(pose.pitch) || 0,
          yaw: Number(pose.yaw) || 0,
          roll: Number(pose.roll) || 0,
          source: "6drepnet",
        };
        state.sixDRepNet.available = true;
        state.sixDRepNet.consecutiveFailures = 0;
        state.sixDRepNet.lastAt = performance.now();
        state.sixDRepNet.lastPose = normalized;
        return normalized;
      })
      .catch((error) => {
        state.sixDRepNet.consecutiveFailures += 1;
        state.sixDRepNet.message = error.message;
        if (state.sixDRepNet.consecutiveFailures >= 2) {
          state.sixDRepNet.disabled = true;
          elements.sourceStatus.textContent = "6DRepNet unavailable · fallback to MediaPipe";
        }
        return null;
      })
      .finally(() => {
        state.sixDRepNet.pending = null;
      });
    return state.sixDRepNet.pending;
  }

  function headDownConfidenceFromSixDRepNet(pose) {
    if (!pose) return 0;
    return headDownConfidenceFromPitch(pose.pitch);
  }

  function getBlendshapeScore(result, name, faceIndex = 0) {
    const categories = result.faceBlendshapes?.[faceIndex]?.categories || [];
    return categories.find((item) => item.categoryName === name)?.score || 0;
  }

  function eyeAspectRatio(landmarks, indexes) {
    const [outer, inner, upperA, lowerA, upperB, lowerB] = indexes.map((index) => landmarks[index]);
    if (!outer || !inner || !upperA || !lowerA || !upperB || !lowerB) return null;
    const horizontal = Math.max(0.001, distance(outer, inner));
    const vertical = (distance(upperA, lowerA) + distance(upperB, lowerB)) / 2;
    return vertical / horizontal;
  }

  function facePoseFromLandmarks(landmarks) {
    const forehead = landmarks[10];
    const chin = landmarks[152];
    const nose = landmarks[1];
    const leftEye = averageLandmark(landmarks, [33, 133, 159, 145]);
    const rightEye = averageLandmark(landmarks, [362, 263, 386, 374]);
    const mouth = averageLandmark(landmarks, [13, 14, 61, 291]);
    if (!forehead || !chin || !nose || !leftEye || !rightEye || !mouth) return null;

    const eyeMid = averageLandmark([leftEye, rightEye], [0, 1]);
    const eyeDistance = Math.max(0.001, distance(leftEye, rightEye));
    const eyeMouthY = Math.max(0.001, mouth.y - eyeMid.y);
    const noseDrop = (nose.y - eyeMid.y) / eyeMouthY;
    return {
      pitch: clamp((noseDrop - 0.64) * 92, -38, 42),
      yaw: clamp(((nose.x - eyeMid.x) / eyeDistance) * 112, -45, 45),
      roll: clamp((Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180) / Math.PI, -35, 35),
      nose,
      eyeMid,
      leftEye,
      rightEye,
    };
  }

  function irisCenter(landmarks, indexes) {
    const center = averageLandmark(landmarks, indexes);
    return center && Number.isFinite(center.x) && Number.isFinite(center.y) ? center : null;
  }

  function landmarkPixel(point, dimensions) {
    if (!point || !dimensions.width || !dimensions.height) return null;
    return {
      x: point.x * dimensions.width,
      y: point.y * dimensions.height,
    };
  }

  function captureFrameSample(source, dimensions) {
    if (!source || !dimensions.width || !dimensions.height) return null;
    const sampleScale = Math.min(192 / dimensions.width, 128 / dimensions.height, 1);
    const sampleWidth = Math.max(1, Math.round(dimensions.width * sampleScale));
    const sampleHeight = Math.max(1, Math.round(dimensions.height * sampleScale));
    try {
      if (frameSampleCanvas.width !== sampleWidth || frameSampleCanvas.height !== sampleHeight) {
        frameSampleCanvas.width = sampleWidth;
        frameSampleCanvas.height = sampleHeight;
      }
      frameSampleContext.drawImage(source, 0, 0, sampleWidth, sampleHeight);
      return {
        width: sampleWidth,
        height: sampleHeight,
        data: frameSampleContext.getImageData(0, 0, sampleWidth, sampleHeight).data,
      };
    } catch {
      return null;
    }
  }

  function sampleFramePixel(frame, sourceX, sourceY, dimensions) {
    if (!frame || !dimensions.width || !dimensions.height) return null;
    if (!Number.isFinite(sourceX) || !Number.isFinite(sourceY)) return null;
    const x = Math.round(clamp((sourceX / dimensions.width) * (frame.width - 1), 0, frame.width - 1));
    const y = Math.round(clamp((sourceY / dimensions.height) * (frame.height - 1), 0, frame.height - 1));
    const offset = (y * frame.width + x) * 4;
    return {
      r: frame.data[offset],
      g: frame.data[offset + 1],
      b: frame.data[offset + 2],
    };
  }

  function softRangeScore(value, min, max, margin) {
    if (value >= min && value <= max) return 1;
    const distanceToRange = value < min ? min - value : value - max;
    return clamp(1 - distanceToRange / margin);
  }

  function skinColorScore(pixel) {
    if (!pixel) return 0;
    const { r, g, b } = pixel;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const brightness = (r + g + b) / 3;
    const saturation = max ? (max - min) / max : 0;
    const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    const ycbcr = softRangeScore(cb, 74, 142, 34) * softRangeScore(cr, 128, 186, 34);
    const warm = clamp((r - b + 34) / 88) * clamp((r - g + 46) / 92);
    const notScreenBlue = clamp(1 - (b - r - 12) / 72);
    const usableLight = softRangeScore(brightness, 42, 236, 42);
    const usableSaturation = softRangeScore(saturation, 0.05, 0.64, 0.22);
    return clamp((ycbcr * 0.58 + warm * 0.28 + notScreenBlue * 0.14) * usableLight * usableSaturation);
  }

  function faceImageEvidence(frame, landmarks, faceBox, dimensions) {
    if (!frame || !faceBox?.length) return { available: false, score: 0.55 };
    const [left, top, right, bottom] = faceBox;
    const radius = Math.max(2, Math.min(right - left, bottom - top) * 0.018);
    const offsets = [
      [0, 0],
      [radius, 0],
      [-radius, 0],
      [0, radius],
      [0, -radius],
    ];
    const scores = [];
    FACE_SURFACE_POINTS.forEach((index) => {
      const point = landmarkPixel(landmarks[index], dimensions);
      if (!point) return;
      offsets.forEach(([offsetX, offsetY]) => {
        const pixel = sampleFramePixel(frame, point.x + offsetX, point.y + offsetY, dimensions);
        if (pixel) scores.push(skinColorScore(pixel));
      });
    });
    if (!scores.length) return { available: false, score: 0.55 };
    scores.sort((first, second) => second - first);
    const kept = scores.slice(0, Math.max(1, Math.ceil(scores.length * 0.62)));
    return {
      available: true,
      score: kept.reduce((sum, value) => sum + value, 0) / kept.length,
    };
  }

  function eyeGazeVector(landmarks, eyeIndexes, irisIndexes) {
    const outer = landmarks[eyeIndexes.outer];
    const inner = landmarks[eyeIndexes.inner];
    const upper = landmarks[eyeIndexes.upper];
    const lower = landmarks[eyeIndexes.lower];
    if (!outer || !inner || !upper || !lower) return null;
    const eyeCenter = averageLandmark(landmarks, [eyeIndexes.outer, eyeIndexes.inner, eyeIndexes.upper, eyeIndexes.lower]);
    const iris = irisCenter(landmarks, irisIndexes);
    const eyeWidth = Math.max(0.001, distance(outer, inner));
    const eyeHeight = Math.max(0.001, distance(upper, lower));
    if (!iris) return null;
    const axisX = {
      x: (inner.x - outer.x) / eyeWidth,
      y: (inner.y - outer.y) / eyeWidth,
    };
    const verticalLength = Math.max(0.001, distance(upper, lower));
    const axisY = {
      x: (lower.x - upper.x) / verticalLength,
      y: (lower.y - upper.y) / verticalLength,
    };
    const delta = {
      x: iris.x - eyeCenter.x,
      y: iris.y - eyeCenter.y,
    };
    const localX = clamp((delta.x * axisX.x + delta.y * axisX.y) / (eyeWidth * 0.36), -1, 1);
    const localY = clamp((delta.x * axisY.x + delta.y * axisY.y) / (eyeHeight * 0.68), -1, 1);
    return {
      origin: iris,
      eyeCenter,
      outline: eyeIndexes.outline.map((index) => landmarks[index]).filter(Boolean),
      x: localX,
      y: localY,
      screenDirection: {
        x: axisX.x * localX + axisY.x * localY,
        y: axisX.y * localX + axisY.y * localY,
      },
      confidence: 0.82,
      hasIris: true,
    };
  }

  function analyzeEyeState(result, landmarks, faceIndex = 0) {
    const leftEar = eyeAspectRatio(landmarks, [33, 133, 159, 145, 158, 153]);
    const rightEar = eyeAspectRatio(landmarks, [362, 263, 386, 374, 385, 380]);
    const earValues = [leftEar, rightEar].filter(Number.isFinite);
    const ear = earValues.length ? earValues.reduce((sum, value) => sum + value, 0) / earValues.length : null;
    const leftBlink = getBlendshapeScore(result, "eyeBlinkLeft", faceIndex);
    const rightBlink = getBlendshapeScore(result, "eyeBlinkRight", faceIndex);
    const leftEarClosed = leftEar === null ? 0 : clamp((0.2 - leftEar) / 0.08);
    const rightEarClosed = rightEar === null ? 0 : clamp((0.2 - rightEar) / 0.08);
    const leftConfidence = clamp(Math.max(leftBlink, leftEarClosed));
    const rightConfidence = clamp(Math.max(rightBlink, rightEarClosed));
    const blink = Math.max(leftBlink, rightBlink);
    const earClosed = Math.max(leftEarClosed, rightEarClosed);
    return {
      confidence: clamp(Math.max(blink, earClosed)),
      leftConfidence,
      rightConfidence,
      ear,
      blink,
    };
  }

  function scoreFaceCandidate(landmarks, dimensions, frameSample) {
    if (!landmarks?.length || !dimensions.width || !dimensions.height) return null;
    const faceBox = faceBoxFromLandmarks(landmarks, dimensions);
    const [left, top, right, bottom] = faceBox;
    const width = Math.max(1, right - left);
    const height = Math.max(1, bottom - top);
    const areaRatio = clamp((width * height) / (dimensions.width * dimensions.height), 0, 1);
    const centerX = (left + right) / 2 / dimensions.width;
    const centerY = (top + bottom) / 2 / dimensions.height;
    const centerScore = clamp(1 - Math.hypot(centerX - 0.5, centerY - 0.48) / 0.72);
    const visibleIndexes = [10, 152, 1, 4, 33, 133, 263, 362, 13, 14, 61, 291];
    const visibleCount = visibleIndexes.filter((index) => {
      const point = landmarks[index];
      return point && point.x >= 0.015 && point.x <= 0.985 && point.y >= 0.015 && point.y <= 0.985;
    }).length;
    const visibleScore = visibleCount / visibleIndexes.length;
    const edgeGap = Math.min(left, top, dimensions.width - right, dimensions.height - bottom);
    const edgeScore = clamp(edgeGap / Math.min(dimensions.width, dimensions.height) / 0.055);
    const facePose = facePoseFromLandmarks(landmarks);
    const poseScore = facePose ? clamp(1 - (Math.abs(facePose.yaw) + Math.abs(facePose.roll) * 0.55) / 72) : 0;
    const imageEvidence = faceImageEvidence(frameSample, landmarks, faceBox, dimensions);
    const score =
      areaRatio * 1.55 +
      visibleScore * 0.3 +
      centerScore * 0.2 +
      edgeScore * 0.18 +
      poseScore * 0.14 +
      imageEvidence.score * 1.25;
    return { score, faceBox, facePose, imageEvidence };
  }

  function selectPrimaryFace(result, dimensions, source = null) {
    const frameSample = captureFrameSample(source, dimensions);
    const candidates = (result.faceLandmarks || [])
      .map((landmarks, index) => {
        const candidate = scoreFaceCandidate(landmarks, dimensions, frameSample);
        return candidate ? { landmarks, index, ...candidate } : null;
      })
      .filter(Boolean)
      .filter((candidate) => !candidate.imageEvidence.available || candidate.imageEvidence.score >= MIN_FACE_IMAGE_EVIDENCE)
      .sort((first, second) => second.score - first.score);
    return candidates[0]?.score >= MIN_PRIMARY_FACE_SCORE ? candidates[0] : null;
  }

  function analyzeGazeState(landmarks, pose = null) {
    const left = eyeGazeVector(
      landmarks,
      { outer: 33, inner: 133, upper: 159, lower: 145, outline: [33, 160, 158, 133, 153, 144] },
      [468, 469, 470, 471, 472],
    );
    const right = eyeGazeVector(
      landmarks,
      { outer: 362, inner: 263, upper: 386, lower: 374, outline: [362, 385, 387, 263, 373, 380] },
      [473, 474, 475, 476, 477],
    );
    const vectors = [left, right].filter(Boolean);
    const irisX = vectors.length ? vectors.reduce((sum, item) => sum + item.x, 0) / vectors.length : 0;
    const irisY = vectors.length ? vectors.reduce((sum, item) => sum + item.y, 0) / vectors.length : 0;
    const poseX = pose ? clamp(pose.yaw / 32, -1, 1) : 0;
    const poseY = pose ? clamp(pose.pitch / 34, -1, 1) : 0;
    const x = vectors.length ? irisX * 0.72 + poseX * 0.28 : poseX;
    const y = vectors.length ? irisY * 0.78 + poseY * 0.22 : poseY;
    const irisLateralRisk = vectors.length === 2 ? clamp((Math.abs(irisX) - 0.42) / 0.36) : 0;
    const headYawRisk = pose
      ? clamp((Math.abs(pose.yaw) - GAZE_OFF_HEAD_YAW_START_DEGREES) / (GAZE_OFF_HEAD_YAW_FULL_DEGREES - GAZE_OFF_HEAD_YAW_START_DEGREES))
      : 0;
    const lateralAgreement =
      vectors.length === 2 && pose && Math.sign(irisX) === Math.sign(pose.yaw) && Math.abs(irisX) > 0.42 && Math.abs(pose.yaw) > 14
        ? 0.08
        : 0;
    const lateralRisk = clamp(Math.max(irisLateralRisk, headYawRisk) + lateralAgreement);
    const downwardRisk = Math.max(
      vectors.length === 2 ? clamp((irisY - 0.38) / 0.36) : 0,
      pose
        ? clamp((pose.pitch - GAZE_OFF_HEAD_PITCH_DOWN_START_DEGREES) / (GAZE_OFF_HEAD_PITCH_FULL_DEGREES - GAZE_OFF_HEAD_PITCH_DOWN_START_DEGREES))
        : 0,
    );
    const upwardRisk = Math.max(
      vectors.length === 2 ? clamp((-irisY - 0.42) / 0.36) : 0,
      pose
        ? clamp((-pose.pitch - GAZE_OFF_HEAD_PITCH_UP_START_DEGREES) / (GAZE_OFF_HEAD_PITCH_FULL_DEGREES - GAZE_OFF_HEAD_PITCH_UP_START_DEGREES))
        : 0,
    );
    if (!vectors.length && !pose) return null;
    return {
      left,
      right,
      x,
      y,
      confidence: clamp(Math.max(lateralRisk, downwardRisk, upwardRisk)),
      hasIris: vectors.length === 2,
    };
  }

  function clearEyePreview(canvas) {
    if (!canvas) return;
    canvas.getContext("2d", { alpha: false }).clearRect(0, 0, canvas.width, canvas.height);
    setEyeCanvasFrame(canvas, false);
  }

  function setEyeCanvasFrame(canvas, hasFrame) {
    if (!canvas) return;
    canvas.classList.toggle("has-frame", hasFrame);
    const figure = canvas.closest("figure");
    if (figure) figure.classList.toggle("has-eye-data", Boolean(figure.querySelector("canvas.has-frame")));
  }

  function deriveEyePreviewCrop(landmarks, faceBox, side, dimensions, targetAspect) {
    const config = EYE_PREVIEW_POINTS[side];
    if (!config || !faceBox?.length) return null;

    const outlinePoints = config.outline.map((index) => landmarks[index]).filter(Boolean);
    const iris = irisCenter(landmarks, config.iris);
    if (outlinePoints.length < 8) return null;

    const pixelPoints = outlinePoints.map((point) => landmarkPixel(point, dimensions)).filter(Boolean);
    const irisPixel = landmarkPixel(iris, dimensions);
    const xs = pixelPoints.map((point) => point.x);
    const ys = pixelPoints.map((point) => point.y);
    const eyeLeft = Math.min(...xs);
    const eyeRight = Math.max(...xs);
    const eyeTop = Math.min(...ys);
    const eyeBottom = Math.max(...ys);
    const eyeWidth = Math.max(1, eyeRight - eyeLeft);
    const eyeHeight = Math.max(1, eyeBottom - eyeTop);
    const [faceLeft, faceTop, faceRight, faceBottom] = faceBox;
    const faceWidth = Math.max(1, faceRight - faceLeft);
    const faceHeight = Math.max(1, faceBottom - faceTop);

    if (eyeWidth < faceWidth * 0.045 || eyeWidth > faceWidth * 0.42 || eyeHeight > faceHeight * 0.22) return null;

    const centerFromOutline = {
      x: (eyeLeft + eyeRight) / 2,
      y: (eyeTop + eyeBottom) / 2,
    };
    const center = irisPixel
      ? {
          x: centerFromOutline.x * 0.68 + irisPixel.x * 0.32,
          y: centerFromOutline.y * 0.72 + irisPixel.y * 0.28,
        }
      : centerFromOutline;

    const withinFace =
      center.x >= faceLeft - faceWidth * 0.04 &&
      center.x <= faceRight + faceWidth * 0.04 &&
      center.y >= faceTop - faceHeight * 0.02 &&
      center.y <= faceBottom + faceHeight * 0.03;
    if (!withinFace) return null;

    const minCropWidth = Math.max(64, faceWidth * 0.24);
    const cropWidth = clamp(Math.max(eyeWidth * 2.75, minCropWidth), minCropWidth, Math.min(faceWidth * 0.48, dimensions.width));
    const cropHeight = Math.min(dimensions.height, Math.max(cropWidth / targetAspect, eyeHeight * 4.5, faceHeight * 0.13));
    const eyeLift = cropHeight * 0.52;
    const horizontalBias = side === "left" ? cropWidth * 0.04 : -cropWidth * 0.04;

    const faceBoundPaddingX = faceWidth * 0.03;
    const faceBoundPaddingY = faceHeight * 0.05;
    const minX = clamp(faceLeft - faceBoundPaddingX, 0, dimensions.width);
    const maxX = clamp(faceRight + faceBoundPaddingX - cropWidth, 0, Math.max(0, dimensions.width - cropWidth));
    const minY = clamp(faceTop - faceBoundPaddingY, 0, dimensions.height);
    const maxY = clamp(faceBottom + faceBoundPaddingY - cropHeight, 0, Math.max(0, dimensions.height - cropHeight));

    const sourceX = clamp(center.x - cropWidth / 2 + horizontalBias, Math.min(minX, maxX), Math.max(minX, maxX));
    const sourceY = clamp(center.y - eyeLift, Math.min(minY, maxY), Math.max(minY, maxY));
    const sourceWidth = Math.min(cropWidth, dimensions.width - sourceX);
    const sourceHeight = Math.min(cropHeight, dimensions.height - sourceY);
    if (sourceWidth < 48 || sourceHeight < 24) return null;

    return { sourceX, sourceY, sourceWidth, sourceHeight };
  }

  function drawEyePreview(source, landmarks, faceBox, side, canvas) {
    if (!canvas) return;
    const dimensions = sourceDimensions(source);
    const targetAspect = canvas.width / canvas.height;
    const crop = deriveEyePreviewCrop(landmarks, faceBox, side, dimensions, targetAspect);
    if (!dimensions.width || !dimensions.height || !crop) {
      clearEyePreview(canvas);
      return;
    }
    const cropContext = canvas.getContext("2d", { alpha: false });
    cropContext.clearRect(0, 0, canvas.width, canvas.height);
    cropContext.filter = "contrast(1.04) brightness(1.02)";
    cropContext.drawImage(source, crop.sourceX, crop.sourceY, crop.sourceWidth, crop.sourceHeight, 0, 0, canvas.width, canvas.height);
    cropContext.filter = "none";
    setEyeCanvasFrame(canvas, true);
  }

  function drawFacePreview(source, faceBox, canvas) {
    if (!canvas || !faceBox) return;
    const dimensions = sourceDimensions(source);
    if (!dimensions.width || !dimensions.height) {
      setEyeCanvasFrame(canvas, false);
      return;
    }
    const boxWidth = Math.max(1, faceBox[2] - faceBox[0]);
    const boxHeight = Math.max(1, faceBox[3] - faceBox[1]);
    const paddingX = boxWidth * 0.18;
    const paddingY = boxHeight * 0.16;
    const faceLeft = Math.max(0, faceBox[0] - paddingX);
    const faceTop = Math.max(0, faceBox[1] - paddingY);
    const faceRight = Math.min(dimensions.width, faceBox[2] + paddingX);
    const faceBottom = Math.min(dimensions.height, faceBox[3] + paddingY);
    const faceWidth = Math.max(1, faceRight - faceLeft);
    const faceHeight = Math.max(1, faceBottom - faceTop);
    const targetAspect = canvas.width / canvas.height;
    let sourceWidth = faceWidth;
    let sourceHeight = faceHeight;

    if (sourceWidth / sourceHeight > targetAspect) {
      sourceHeight = sourceWidth / targetAspect;
    } else {
      sourceWidth = sourceHeight * targetAspect;
    }

    sourceWidth = Math.min(sourceWidth, dimensions.width);
    sourceHeight = Math.min(sourceHeight, dimensions.height);
    const centerX = (faceLeft + faceRight) / 2;
    const centerY = (faceTop + faceBottom) / 2;
    const sourceX = clamp(centerX - sourceWidth / 2, 0, Math.max(0, dimensions.width - sourceWidth));
    const sourceY = clamp(centerY - sourceHeight / 2, 0, Math.max(0, dimensions.height - sourceHeight));
    const cropContext = canvas.getContext("2d", { alpha: false });
    cropContext.clearRect(0, 0, canvas.width, canvas.height);
    cropContext.filter = "grayscale(1) contrast(1.18) brightness(1.08)";
    cropContext.drawImage(source, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, canvas.width, canvas.height);
    cropContext.filter = "none";
    setEyeCanvasFrame(canvas, true);
  }

  function drawEyeWave(canvas, values) {
    if (!canvas) return;
    if (!values?.length) {
      clearEyePreview(canvas);
      return;
    }
    const waveContext = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    waveContext.clearRect(0, 0, width, height);
    waveContext.fillStyle = "#08132d";
    waveContext.fillRect(0, 0, width, height);
    waveContext.strokeStyle = "rgba(90, 128, 210, 0.16)";
    waveContext.lineWidth = 1;
    for (let x = 0; x <= width; x += width / 6) {
      waveContext.beginPath();
      waveContext.moveTo(x, 0);
      waveContext.lineTo(x, height);
      waveContext.stroke();
    }
    for (let y = 0; y <= height; y += height / 3) {
      waveContext.beginPath();
      waveContext.moveTo(0, y);
      waveContext.lineTo(width, y);
      waveContext.stroke();
    }
    const series = values;
    const gradient = waveContext.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, "rgba(57, 119, 255, 0.92)");
    gradient.addColorStop(1, "rgba(26, 68, 164, 0.28)");
    waveContext.beginPath();
    waveContext.moveTo(0, height);
    series.forEach((value, index) => {
      const x = series.length > 1 ? (index / (series.length - 1)) * width : width;
      const y = height - clamp(value) * (height - 5) - 2;
      waveContext.lineTo(x, y);
    });
    waveContext.lineTo(width, height);
    waveContext.closePath();
    waveContext.fillStyle = gradient;
    waveContext.fill();
    waveContext.strokeStyle = "#66d8ff";
    waveContext.lineWidth = 2;
    waveContext.beginPath();
    series.forEach((value, index) => {
      const x = series.length > 1 ? (index / (series.length - 1)) * width : width;
      const y = height - clamp(value) * (height - 5) - 2;
      if (index === 0) waveContext.moveTo(x, y);
      else waveContext.lineTo(x, y);
    });
    waveContext.stroke();
    setEyeCanvasFrame(canvas, true);
  }

  function setTelemetryValue(element, value) {
    if (element) element.textContent = value;
  }

  function updateFaceTelemetryUI(metrics, source = null) {
    state.headPose.metrics = metrics || null;
    if (!metrics) {
      [elements.facePoseValue, elements.gazeVectorValue, elements.eyeVectorValue].forEach((element) => setTelemetryValue(element, "--"));
      [elements.headXValue, elements.headYValue, elements.headZValue].forEach((element) => setTelemetryValue(element, "--"));
      [elements.headPitchValue, elements.headYawValue, elements.headRollValue].forEach((element) => setTelemetryValue(element, "--°"));
      setTelemetryValue(elements.faceVectorStatus, "Not Detected");
      setTelemetryValue(elements.faceIdValue, "FACE --");
      setTelemetryValue(elements.landmarkCountValue, "--");
      setTelemetryValue(elements.driverActionValue, "Waiting for Input");
      setTelemetryValue(elements.gazePitchValue, "--°");
      setTelemetryValue(elements.gazeYawValue, "--°");
      [elements.leftEyeOpenValue, elements.rightEyeOpenValue, elements.leftBlinkValue, elements.rightBlinkValue].forEach((element) => setTelemetryValue(element, "--"));
      setTelemetryValue(elements.leftEyeState, "Waiting for Input");
      setTelemetryValue(elements.rightEyeState, "Waiting for Input");
      setTelemetryValue(elements.topFaceCount, "0 people");
      if (elements.eyeOpennessBar) elements.eyeOpennessBar.style.width = "0%";
      if (elements.headPoseDial) elements.headPoseDial.style.transform = "rotate(0deg)";
      clearEyePreview(elements.leftEyePreview);
      clearEyePreview(elements.rightEyePreview);
      clearEyePreview(elements.leftEyeWave);
      clearEyePreview(elements.rightEyeWave);
      clearEyePreview(elements.facePreview);
      state.headPose.eyeHistory.left = [];
      state.headPose.eyeHistory.right = [];
      return;
    }
    const yawText = metrics.pose.yaw > 8 ? `Right ${Math.round(metrics.pose.yaw)}°` : metrics.pose.yaw < -8 ? `Left ${Math.abs(Math.round(metrics.pose.yaw))}°` : "Centered";
    const pitchText = metrics.pose.pitch > 8 ? `Head Down ${Math.round(metrics.pose.pitch)}°` : metrics.pose.pitch < -8 ? `Head Up ${Math.abs(Math.round(metrics.pose.pitch))}°` : "Level";
    const gazeX = metrics.gaze.x > 0.35 ? "Right" : metrics.gaze.x < -0.35 ? "Left" : "Center";
    const gazeY = metrics.gaze.y > 0.32 ? "Down" : metrics.gaze.y < -0.38 ? "Up" : "Level";
    const openness = Math.round((1 - metrics.eye.confidence) * 100);
    const leftOpenness = Math.round((1 - metrics.eye.leftConfidence) * 100);
    const rightOpenness = Math.round((1 - metrics.eye.rightConfidence) * 100);
    const gazePitch = metrics.gaze.y * 24;
    const gazeYaw = metrics.gaze.x * 32;
    const poseEngine = metrics.poseSource === "6drepnet" ? "6DRepNet Head Pose" : "MediaPipe Head Pose";
    const action = metrics.eye.confidence >= EYE_CLOSED_CONFIDENCE_THRESHOLD
      ? "Eye Closed"
      : metrics.pose.pitch > 8
        ? "Head Down"
        : metrics.gaze.confidence >= GAZE_OFF_CONFIDENCE_THRESHOLD
          ? "Gaze Offset"
          : "Normal Gaze";
    setTelemetryValue(elements.facePoseValue, `${pitchText} / ${yawText}`);
    setTelemetryValue(elements.gazeVectorValue, `${gazeX}${gazeY} · ${Math.round(metrics.gaze.confidence * 100)}%`);
    setTelemetryValue(elements.eyeVectorValue, metrics.eye.confidence >= EYE_CLOSED_CONFIDENCE_THRESHOLD ? `Eyes Closed ${Math.round(metrics.eye.confidence * 100)}%` : `Eyes Open ${openness}%`);
    setTelemetryValue(elements.faceVectorStatus, `${poseEngine} · ${metrics.gaze.hasIris ? "Dual iris direction lines" : "Single iris direction line"}`);
    setTelemetryValue(elements.faceIdValue, "FACE 01");
    setTelemetryValue(elements.landmarkCountValue, `${metrics.landmarkCount} pts`);
    setTelemetryValue(elements.driverActionValue, action);
    setTelemetryValue(elements.gazePitchValue, `${gazePitch >= 0 ? "+" : ""}${gazePitch.toFixed(1)}°`);
    setTelemetryValue(elements.gazeYawValue, `${gazeYaw >= 0 ? "+" : ""}${gazeYaw.toFixed(1)}°`);
    setTelemetryValue(elements.leftEyeOpenValue, `${leftOpenness}%`);
    setTelemetryValue(elements.rightEyeOpenValue, `${rightOpenness}%`);
    setTelemetryValue(elements.leftBlinkValue, `${Math.round(metrics.eye.leftConfidence * 100)}%`);
    setTelemetryValue(elements.rightBlinkValue, `${Math.round(metrics.eye.rightConfidence * 100)}%`);
    setTelemetryValue(elements.headXValue, `${metrics.position.x >= 0 ? "+" : ""}${metrics.position.x.toFixed(1)}%`);
    setTelemetryValue(elements.headYValue, `${metrics.position.y >= 0 ? "+" : ""}${metrics.position.y.toFixed(1)}%`);
    setTelemetryValue(elements.headZValue, metrics.position.z.toFixed(2));
    setTelemetryValue(elements.headPitchValue, `${metrics.pose.pitch.toFixed(1)}°`);
    setTelemetryValue(elements.headYawValue, `${metrics.pose.yaw.toFixed(1)}°`);
    setTelemetryValue(elements.headRollValue, `${metrics.pose.roll.toFixed(1)}°`);
    setTelemetryValue(elements.leftEyeState, metrics.eye.leftConfidence >= EYE_CLOSED_CONFIDENCE_THRESHOLD ? "Closed" : "Open");
    setTelemetryValue(elements.rightEyeState, metrics.eye.rightConfidence >= EYE_CLOSED_CONFIDENCE_THRESHOLD ? "Closed" : "Open");
    setTelemetryValue(elements.topFaceCount, `${metrics.faceCount || 1} person${(metrics.faceCount || 1) > 1 ? "s" : ""}`);
    if (elements.eyeOpennessBar) elements.eyeOpennessBar.style.width = `${openness}%`;
    if (elements.headPoseDial) elements.headPoseDial.style.transform = `rotate(${metrics.pose.roll.toFixed(1)}deg)`;
    if (source && metrics.landmarks?.length) {
      drawFacePreview(source, metrics.faceBox, elements.facePreview);
      drawEyePreview(source, metrics.landmarks, metrics.faceBox, "left", elements.leftEyePreview);
      drawEyePreview(source, metrics.landmarks, metrics.faceBox, "right", elements.rightEyePreview);
      drawEyeWave(elements.leftEyeWave, state.headPose.eyeHistory.left);
      drawEyeWave(elements.rightEyeWave, state.headPose.eyeHistory.right);
    }
  }

  function pitchConfidenceFromMatrix(matrix) {
    const data = matrix?.data || matrix;
    if (!data || data.length < 16) return 0;
    const candidates = [
      Math.atan2(-data[9], Math.hypot(data[8], data[10])),
      Math.atan2(data[6], Math.hypot(data[2], data[10])),
    ]
      .map((value) => (value * 180) / Math.PI)
      .filter(Number.isFinite);
    if (!candidates.length) return 0;
    const pitch = Math.max(0, ...candidates);
    return headDownConfidenceFromPitch(pitch);
  }

  function headDownConfidenceFromPitch(pitch) {
    return clamp((pitch - SIXDREPNET_HEAD_DOWN_PITCH_DEGREES) / SIXDREPNET_HEAD_DOWN_RANGE_DEGREES);
  }

  function landmarkConfidence(landmarks) {
    const forehead = landmarks[10];
    const chin = landmarks[152];
    const nose = landmarks[1];
    const leftEye = averageLandmark(landmarks, [33, 133, 159, 145]);
    const rightEye = averageLandmark(landmarks, [362, 263, 386, 374]);
    const mouth = averageLandmark(landmarks, [13, 14, 61, 291]);
    if (!forehead || !chin || !nose || !leftEye || !rightEye || !mouth) return 0;

    const eyeMid = averageLandmark([leftEye, rightEye], [0, 1]);
    const faceHeight = Math.max(0.001, distance(forehead, chin));
    const eyeMouthY = Math.max(0.001, mouth.y - eyeMid.y);
    const noseDrop = (nose.y - eyeMid.y) / eyeMouthY;
    const lowerFace = distance(mouth, chin) / faceHeight;
    const upperFace = distance(forehead, eyeMid) / faceHeight;
    const noseScore = clamp((noseDrop - 0.58) / 0.24);
    const tuckScore = clamp((0.26 - lowerFace) / 0.13);
    const compressionScore = clamp((0.28 - upperFace) / 0.12);
    return clamp(noseScore * 0.55 + tuckScore * 0.3 + compressionScore * 0.15);
  }

  async function ensureHeadPoseModel() {
    if (state.headPose.landmarker || state.headPose.loadFailed) return state.headPose.landmarker;
    if (!state.headPose.initPromise) {
      state.headPose.initPromise = (async () => {
        const { FaceLandmarker, FilesetResolver } = await import(HEAD_POSE_MODEL.bundle);
        const vision = await FilesetResolver.forVisionTasks(HEAD_POSE_MODEL.wasmPath);
        const createOptions = (delegate) => ({
          baseOptions: {
            modelAssetPath: HEAD_POSE_MODEL.modelPath,
            delegate,
          },
          outputFacialTransformationMatrixes: true,
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: MAX_FACE_CANDIDATES,
          minFaceDetectionConfidence: 0.55,
          minFacePresenceConfidence: 0.55,
          minTrackingConfidence: 0.58,
        });
        try {
          state.headPose.landmarker = await FaceLandmarker.createFromOptions(vision, createOptions("GPU"));
        } catch (error) {
          console.warn("Face landmarker GPU delegate unavailable, falling back to CPU", error);
          state.headPose.landmarker = await FaceLandmarker.createFromOptions(vision, createOptions("CPU"));
        }
        return state.headPose.landmarker;
      })().catch((error) => {
        state.headPose.loadFailed = true;
        console.warn("Head pose model failed to load", error);
        if (elements.sourceStatus && !elements.sourceStatus.classList.contains("error")) {
          elements.sourceStatus.textContent = "Local weights found · head pose model not loaded";
        }
        return null;
      });
    }
    return state.headPose.initPromise;
  }

  async function analyzeHeadPose(source) {
    const landmarker = await ensureHeadPoseModel();
    if (!landmarker) return [];
    const mode = source instanceof HTMLVideoElement || source === cameraFrameCanvas ? "VIDEO" : "IMAGE";
    if (state.headPose.mode !== mode) {
      await landmarker.setOptions({ runningMode: mode });
      state.headPose.mode = mode;
    }
    const result = mode === "VIDEO" ? landmarker.detectForVideo(source, performance.now()) : landmarker.detect(source);
    const dimensions = sourceDimensions(source);
    const now = Date.now();
    const primaryFace = selectPrimaryFace(result, dimensions, source);
    if (!primaryFace) {
      state.headPose.downSince = 0;
      state.headPose.eyeClosedSince = 0;
      state.headPose.gazeOffSince = 0;
      state.headPose.gazeOffHoldUntil = 0;
      if (now - state.headPose.faceBoxLastSeenAt > FACE_BOX_HOLD_MS) {
        state.headPose.smoothedFaceBox = null;
      }
      state.headPose.overlay = null;
      updateFaceTelemetryUI(null);
      return [];
    }

    const detections = [];
    const { landmarks, index: faceIndex, faceBox: rawFaceBox, facePose: landmarkPose } = primaryFace;
    const faceBox = stabilizeFaceBox(rawFaceBox, dimensions, now) || rawFaceBox;
    const eye = analyzeEyeState(result, landmarks, faceIndex);
    const sixDRepNetPose = await requestSixDRepNetPose(source, faceBox);
    const pose = sixDRepNetPose || landmarkPose;
    const gaze = landmarkPose ? analyzeGazeState(landmarks, pose) : null;

    if (pose && gaze) {
      const faceWidth = Math.max(1, faceBox[2] - faceBox[0]);
      const faceCenterX = (faceBox[0] + faceBox[2]) / 2;
      const faceCenterY = (faceBox[1] + faceBox[3]) / 2;
      const metrics = {
        pose,
        poseSource: sixDRepNetPose ? "6drepnet" : "mediapipe",
        eye,
        gaze,
        faceBox,
        landmarks,
        landmarkCount: landmarks.length,
        faceCount: result.faceLandmarks?.length || 0,
        imageEvidence: primaryFace.imageEvidence,
        position: {
          x: ((faceCenterX / dimensions.width) - 0.5) * 100,
          y: ((faceCenterY / dimensions.height) - 0.5) * 100,
          z: dimensions.width / faceWidth,
        },
      };
      state.headPose.eyeHistory.left.push(1 - eye.leftConfidence);
      state.headPose.eyeHistory.right.push(1 - eye.rightConfidence);
      state.headPose.eyeHistory.left = state.headPose.eyeHistory.left.slice(-48);
      state.headPose.eyeHistory.right = state.headPose.eyeHistory.right.slice(-48);
      state.headPose.overlay = metrics;
      updateFaceTelemetryUI(metrics, source);
    } else {
      state.headPose.overlay = null;
      updateFaceTelemetryUI(null);
    }

    const transformScore = pitchConfidenceFromMatrix(result.facialTransformationMatrixes?.[faceIndex]);
    const pointScore = landmarkConfidence(landmarks);
    const posePitchScore = headDownConfidenceFromPitch(pose?.pitch || 0);
    const sixDScore = headDownConfidenceFromSixDRepNet(sixDRepNetPose);
    const confidence = sixDRepNetPose
      ? clamp(Math.max(sixDScore, pointScore * 0.25))
      : clamp(Math.max(posePitchScore, pointScore >= 0.22 ? transformScore * 0.9 : 0, pointScore * 0.72));
    if (confidence < HEAD_DOWN_CONFIDENCE_THRESHOLD) {
      state.headPose.downSince = 0;
    } else {
      if (!state.headPose.downSince) state.headPose.downSince = now;
      detections.push({
        key: "head_down",
        originalClass: "HeadDown",
        confidence,
        source: sixDRepNetPose ? "6drepnet" : "face-vector",
        box: faceBox,
      });
    }

    // 闭眼报警:只有"完全闭眼"连续保持超过 1 秒才触发;
    // 眯眼/小眼睛(置信度处于 0.52~0.75 灰区)视为睁眼,计时器随时归零,不会报警。
    if (eye.confidence < EYE_FULLY_CLOSED_CONFIDENCE) {
      state.headPose.eyeClosedSince = 0;
    } else {
      if (!state.headPose.eyeClosedSince) state.headPose.eyeClosedSince = now;
      if (now - state.headPose.eyeClosedSince >= EYE_CLOSED_CONFIRM_MS) {
        detections.push({
          key: "drowsy",
          originalClass: "EyeClosed",
          confidence: eye.confidence,
          source: "face-vector",
          box: faceBox,
        });
      }
    }

    const gazeEyesReliable =
      gaze?.hasIris &&
      eye.leftConfidence < GAZE_EYE_CLOSED_SUPPRESS_THRESHOLD &&
      eye.rightConfidence < GAZE_EYE_CLOSED_SUPPRESS_THRESHOLD;
    const gazeOffActive = gazeEyesReliable && gaze.confidence >= GAZE_OFF_CONFIDENCE_THRESHOLD;
    const gazeOffCooling = gazeEyesReliable && gaze.confidence >= GAZE_OFF_RELEASE_THRESHOLD && state.headPose.gazeOffHoldUntil > now;
    if (!gazeEyesReliable || (!gazeOffActive && !gazeOffCooling)) {
      state.headPose.gazeOffSince = 0;
      state.headPose.gazeOffHoldUntil = 0;
    } else {
      if (gazeOffActive && !state.headPose.gazeOffSince) state.headPose.gazeOffSince = now;
      if (gazeOffActive && now - state.headPose.gazeOffSince >= GAZE_OFF_CONFIRM_MS) {
        state.headPose.gazeOffHoldUntil = now + GAZE_OFF_HOLD_MS;
      }
      if (state.headPose.gazeOffHoldUntil > now) {
        detections.push({
          key: "gaze_off",
          originalClass: "GazeOffRoad",
          confidence: Math.max(gaze.confidence, GAZE_OFF_CONFIDENCE_THRESHOLD),
          source: "face-vector",
          box: faceBox,
        });
      }
    }

    return detections;
  }

  function preprocess(source) {
    const dimensions = sourceDimensions(source);
    preprocessContext.fillStyle = "#808080";
    preprocessContext.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    const scale = Math.min(INPUT_SIZE / dimensions.width, INPUT_SIZE / dimensions.height);
    const width = dimensions.width * scale;
    const height = dimensions.height * scale;
    const offsetX = (INPUT_SIZE - width) / 2;
    const offsetY = (INPUT_SIZE - height) / 2;
    preprocessContext.drawImage(source, offsetX, offsetY, width, height);

    const pixels = preprocessContext.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const channelSize = INPUT_SIZE * INPUT_SIZE;
    for (let index = 0; index < channelSize; index += 1) {
      modelInputData[index] = pixels[index * 4] / 255;
      modelInputData[channelSize + index] = pixels[index * 4 + 1] / 255;
      modelInputData[channelSize * 2 + index] = pixels[index * 4 + 2] / 255;
    }
    return {
      tensor: new ort.Tensor("float32", modelInputData, [1, 3, INPUT_SIZE, INPUT_SIZE]),
      scale,
      offsetX,
      offsetY,
      ...dimensions,
    };
  }

  function parseDetections(outputs, prep, modelName) {
    const output = outputs[Object.keys(outputs)[0]];
    const dimensions = output.dims;
    if (dimensions.length !== 3 || dimensions[0] !== 1) return [];
    const classes = MODEL_CLASSES[modelName];
    const attrMajor = dimensions[1] < dimensions[2];
    const attrCount = attrMajor ? dimensions[1] : dimensions[2];
    const detectionCount = attrMajor ? dimensions[2] : dimensions[1];
    const hasObjectness = attrCount >= classes.length + 5;
    const classStart = hasObjectness ? 5 : 4;
    const classCount = Math.min(classes.length, Math.max(0, attrCount - classStart));
    const detections = [];
    if (classCount <= 0) return detections;

    const readAttr = (detectionIndex, attrIndex) =>
      attrMajor ? output.data[attrIndex * detectionCount + detectionIndex] : output.data[detectionIndex * attrCount + attrIndex];

    for (let detectionIndex = 0; detectionIndex < detectionCount; detectionIndex += 1) {
      let confidence = 0;
      let classIndex = 0;
      for (let index = 0; index < classCount; index += 1) {
        const score = readAttr(detectionIndex, classStart + index);
        if (score > confidence) {
          confidence = score;
          classIndex = index;
        }
      }
      const objectness = hasObjectness ? readAttr(detectionIndex, 4) : 1;
      confidence *= objectness;
      let x = readAttr(detectionIndex, 0);
      let y = readAttr(detectionIndex, 1);
      let width = readAttr(detectionIndex, 2);
      let height = readAttr(detectionIndex, 3);
      if (Math.max(x, y, width, height) <= 2) {
        x *= INPUT_SIZE;
        y *= INPUT_SIZE;
        width *= INPUT_SIZE;
        height *= INPUT_SIZE;
      }
      if (width <= 0 || height <= 0) continue;
      const originalClass = classes[classIndex];
      const key = UNIFIED_CLASSES[originalClass] || originalClass;
      if (!ACTIVE_DETECTION_KEYS.has(key)) continue;
      // COCO 是手机检测的唯一来源，屏蔽旧模型的高误报 phone 输出
      if (key === "phone" && modelName !== PHONE_OBJECT_MODEL) continue;
      if (confidence < confidenceThresholdForKey(key)) continue;
      detections.push({
        key,
        originalClass,
        confidence,
        source: modelName,
        box: [
          Math.max(0, (x - width / 2 - prep.offsetX) / prep.scale),
          Math.max(0, (y - height / 2 - prep.offsetY) / prep.scale),
          Math.min(prep.width, (x + width / 2 - prep.offsetX) / prep.scale),
          Math.min(prep.height, (y + height / 2 - prep.offsetY) / prep.scale),
        ],
      });
    }
    return detections;
  }

  // 打电话时手机贴耳、有倾斜角且目标小:以人脸框为基准裁 ROI 放大,
  // 再做 ±22° 旋转 TTA,任何一次命中都按最高置信度保留。
  function phoneRoiRegion(faceBox, dims) {
    const faceWidth = Math.max(1, faceBox[2] - faceBox[0]);
    const faceHeight = Math.max(1, faceBox[3] - faceBox[1]);
    const size = Math.max(faceWidth, faceHeight) * PHONE_ROI_EXPAND;
    const centerX = (faceBox[0] + faceBox[2]) / 2;
    const centerY = (faceBox[1] + faceBox[3]) / 2;
    const left = clamp(centerX - size / 2, 0, Math.max(0, dims.width - size));
    const top = clamp(centerY - size / 2, 0, Math.max(0, dims.height - size));
    return {
      left,
      top,
      width: Math.min(size, dims.width - left),
      height: Math.min(size, dims.height - top),
    };
  }

  function runPhoneRoiInference(inferenceSource, region, dims, angleDegrees) {
    const session = state.sessions[PHONE_OBJECT_MODEL];
    if (!session) return [];
    const size = PHONE_ROI_INPUT_SIZE;
    const half = size / 2;
    const angle = (angleDegrees * Math.PI) / 180;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const boundWidth = region.width * Math.abs(cos) + region.height * Math.abs(sin);
    const boundHeight = region.width * Math.abs(sin) + region.height * Math.abs(cos);
    const scale = size / Math.max(boundWidth, boundHeight);

    phoneRoiContext.setTransform(1, 0, 0, 1, 0, 0);
    phoneRoiContext.fillStyle = "#808080";
    phoneRoiContext.fillRect(0, 0, size, size);
    phoneRoiContext.translate(half, half);
    phoneRoiContext.rotate(angle);
    phoneRoiContext.scale(scale, scale);
    phoneRoiContext.translate(-(region.left + region.width / 2), -(region.top + region.height / 2));
    phoneRoiContext.drawImage(inferenceSource, 0, 0, dims.width, dims.height);
    phoneRoiContext.setTransform(1, 0, 0, 1, 0, 0);

    const pixels = phoneRoiContext.getImageData(0, 0, size, size).data;
    const channelSize = size * size;
    for (let index = 0; index < channelSize; index += 1) {
      phoneRoiData[index] = pixels[index * 4] / 255;
      phoneRoiData[channelSize + index] = pixels[index * 4 + 1] / 255;
      phoneRoiData[channelSize * 2 + index] = pixels[index * 4 + 2] / 255;
    }

    return session
      .run({ [session.inputNames[0]]: new ort.Tensor("float32", phoneRoiData, [1, 3, size, size]) })
      .then((outputs) => {
        const output = outputs[Object.keys(outputs)[0]];
        const outDims = output.dims;
        if (outDims.length !== 3 || outDims[0] !== 1) return [];
        const classes = MODEL_CLASSES[PHONE_OBJECT_MODEL];
        const attrMajor = outDims[1] < outDims[2];
        const attrCount = attrMajor ? outDims[1] : outDims[2];
        const detectionCount = attrMajor ? outDims[2] : outDims[1];
        const classStart = 4;
        const classCount = Math.min(classes.length, Math.max(0, attrCount - classStart));
        const phoneClassIndex = classes.indexOf("cell phone");
        if (classCount <= 0 || phoneClassIndex < 0) return [];
        const readAttr = (detectionIndex, attrIndex) =>
          attrMajor
            ? output.data[attrIndex * detectionCount + detectionIndex]
            : output.data[detectionIndex * attrCount + attrIndex];
        const results = [];
        for (let detectionIndex = 0; detectionIndex < detectionCount; detectionIndex += 1) {
          const confidence = readAttr(detectionIndex, classStart + phoneClassIndex);
          if (confidence < PHONE_CONFIDENCE_THRESHOLD) continue;
          const centerX = readAttr(detectionIndex, 0);
          const centerY = readAttr(detectionIndex, 1);
          const boxWidth = readAttr(detectionIndex, 2);
          const boxHeight = readAttr(detectionIndex, 3);
          if (boxWidth <= 0 || boxHeight <= 0) continue;
          results.push({
            confidence,
            corners: [
              [centerX - boxWidth / 2, centerY - boxHeight / 2],
              [centerX + boxWidth / 2, centerY - boxHeight / 2],
              [centerX - boxWidth / 2, centerY + boxHeight / 2],
              [centerX + boxWidth / 2, centerY + boxHeight / 2],
            ],
          });
        }
        return results;
      });
  }

  function roiPointToSource(px, py, region, scale, angleRadians, dims) {
    const half = PHONE_ROI_INPUT_SIZE / 2;
    const cos = Math.cos(-angleRadians);
    const sin = Math.sin(-angleRadians);
    const dx = (px - half) / scale;
    const dy = (py - half) / scale;
    return {
      x: clamp(dx * cos - dy * sin + region.left + region.width / 2, 0, dims.width),
      y: clamp(dx * sin + dy * cos + region.top + region.height / 2, 0, dims.height),
    };
  }

  async function detectPhoneAroundFace(inferenceSource, existingDetections) {
    const nowMs = Date.now();
    if (state.sessions[PHONE_OBJECT_MODEL] === undefined) return [];
    if (
      existingDetections.some(
        (item) => item.key === "phone" && item.confidence >= PHONE_CONFIDENCE_THRESHOLD,
      )
    ) {
      state.phoneRoiLastDetection = null;
      return [];
    }
    const faceBox = state.headPose.overlay?.faceBox || recentStableFaceBox(nowMs);
    if (!faceBox) return cachedPhoneRoiDetections(nowMs);
    if (nowMs - state.phoneRoiLastRunAt < PHONE_ROI_MIN_INTERVAL_MS) return cachedPhoneRoiDetections(nowMs);
    state.phoneRoiLastRunAt = nowMs;
    const dims = sourceDimensions(inferenceSource);
    if (!dims.width || !dims.height) return [];
    const region = phoneRoiRegion(faceBox, dims);
    const attempts = [0, ...PHONE_ROI_ROTATIONS_DEGREES];
    let best = null;
    for (const angleDegrees of attempts) {
      const angleRadians = (angleDegrees * Math.PI) / 180;
      const boundWidth = region.width * Math.abs(Math.cos(angleRadians)) + region.height * Math.abs(Math.sin(angleRadians));
      const boundHeight = region.width * Math.abs(Math.sin(angleRadians)) + region.height * Math.abs(Math.cos(angleRadians));
      const scale = PHONE_ROI_INPUT_SIZE / Math.max(boundWidth, boundHeight);
      const candidates = await runPhoneRoiInference(inferenceSource, region, dims, angleDegrees);
      for (const candidate of candidates) {
        const mapped = candidate.corners.map(([px, py]) => roiPointToSource(px, py, region, scale, angleRadians, dims));
        const box = [
          Math.min(...mapped.map((point) => point.x)),
          Math.min(...mapped.map((point) => point.y)),
          Math.max(...mapped.map((point) => point.x)),
          Math.max(...mapped.map((point) => point.y)),
        ];
        if (!best || candidate.confidence > best.confidence) {
          best = {
            key: "phone",
            originalClass: "cell phone",
            confidence: candidate.confidence,
            source: PHONE_OBJECT_MODEL,
            box,
          };
        }
      }
      if (best && best.confidence >= PHONE_CONFIDENCE_THRESHOLD) break;
    }
    if (best) {
      state.phoneRoiLastDetection = { ...best, heldUntil: nowMs + PHONE_ROI_RESULT_HOLD_MS };
      return [best];
    }
    return cachedPhoneRoiDetections(nowMs);
  }

  function intersectionOverUnion(first, second) {
    const left = Math.max(first[0], second[0]);
    const top = Math.max(first[1], second[1]);
    const right = Math.min(first[2], second[2]);
    const bottom = Math.min(first[3], second[3]);
    const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
    const firstArea = Math.max(0, first[2] - first[0]) * Math.max(0, first[3] - first[1]);
    const secondArea = Math.max(0, second[2] - second[0]) * Math.max(0, second[3] - second[1]);
    const union = firstArea + secondArea - intersection;
    return union > 0 ? intersection / union : 0;
  }

  function suppressOverlaps(detections) {
    const grouped = Object.groupBy
      ? Object.groupBy(detections, (item) => item.key)
      : detections.reduce((result, item) => {
          (result[item.key] ||= []).push(item);
          return result;
        }, {});
    return Object.values(grouped).flatMap((items) => {
      const candidates = [...items].sort((first, second) => second.confidence - first.confidence);
      const kept = [];
      while (candidates.length) {
        const best = candidates.shift();
        kept.push(best);
        for (let index = candidates.length - 1; index >= 0; index -= 1) {
          if (intersectionOverUnion(best.box, candidates[index].box) > IOU_THRESHOLD) candidates.splice(index, 1);
        }
      }
      return kept;
    });
  }

  function drawVectorArrow(ctx, startX, startY, endX, endY, color, width) {
    const angle = Math.atan2(endY - startY, endX - startX);
    const head = Math.max(8, width * 4.5);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(endX - head * Math.cos(angle - Math.PI / 6), endY - head * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(endX - head * Math.cos(angle + Math.PI / 6), endY - head * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawEyeOutline(ctx, points, mapPoint, lineWidth) {
    if (!points?.length) return;
    ctx.save();
    ctx.strokeStyle = "rgba(96, 165, 250, 0.86)";
    ctx.lineWidth = Math.max(1.2, lineWidth * 0.52);
    ctx.shadowColor = "rgba(59, 130, 246, 0.72)";
    ctx.shadowBlur = 6;
    ctx.beginPath();
    points.forEach((point, index) => {
      const mapped = mapPoint(point);
      if (index === 0) ctx.moveTo(mapped.x, mapped.y);
      else ctx.lineTo(mapped.x, mapped.y);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }

  function drawGazeRay(ctx, origin, direction, faceSpan, lineWidth, mirrorX = false) {
    const rawX = Number.isFinite(direction?.x) ? direction.x : 0;
    const rawY = Number.isFinite(direction?.y) ? direction.y : 0;
    const vector = {
      x: mirrorX ? -rawX : rawX,
      y: rawY,
    };
    const strength = Math.min(1.4, Math.hypot(vector.x, vector.y));
    const originRadius = Math.max(4.5, lineWidth * 1.55);
    const rayColor = "#f4f07a";
    const coreColor = "#fffbd1";

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.fillStyle = coreColor;
    ctx.shadowColor = rayColor;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(origin.x, origin.y, originRadius, 0, Math.PI * 2);
    ctx.fill();

    if (strength < 0.08) {
      ctx.strokeStyle = "rgba(244, 240, 122, 0.84)";
      ctx.lineWidth = Math.max(2.2, lineWidth * 0.62);
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, originRadius * 1.85, 0, Math.PI * 2);
      ctx.stroke();
      if (state.annotationRecorder.capture) {
        state.annotationRecorder.capture.gaze.push({
          o: [captureNormX(ctx, origin.x), captureNormY(ctx, origin.y)],
          e: null,
        });
      }
      ctx.restore();
      return;
    }

    const unitX = vector.x / strength;
    const unitY = vector.y / strength;
    const length = Math.max(44, Math.min(192, faceSpan * 0.68)) * (0.55 + Math.min(1, strength) * 0.65);
    const end = {
      x: origin.x + unitX * length,
      y: origin.y + unitY * length,
    };
    if (state.annotationRecorder.capture) {
      state.annotationRecorder.capture.gaze.push({
        o: [captureNormX(ctx, origin.x), captureNormY(ctx, origin.y)],
        e: [captureNormX(ctx, end.x), captureNormY(ctx, end.y)],
      });
    }
    const arrowSize = Math.max(11, lineWidth * 3.6);
    const angle = Math.atan2(unitY, unitX);

    ctx.strokeStyle = "rgba(244, 240, 122, 0.44)";
    ctx.lineWidth = Math.max(9, lineWidth * 2.7);
    ctx.shadowBlur = 24;
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();

    ctx.strokeStyle = coreColor;
    ctx.lineWidth = Math.max(3.2, lineWidth * 0.92);
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();

    ctx.fillStyle = rayColor;
    ctx.beginPath();
    ctx.moveTo(end.x, end.y);
    ctx.lineTo(end.x - arrowSize * Math.cos(angle - Math.PI / 6), end.y - arrowSize * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(end.x - arrowSize * Math.cos(angle + Math.PI / 6), end.y - arrowSize * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawHeadCoordinateSystem(ctx, pose, mapPoint, axisLength, lineWidth, mirrorX = false) {
    const origin = mapPoint(pose.nose);
    const mirrorSign = mirrorX ? -1 : 1;
    const roll = (pose.roll * Math.PI) / 180;
    const yaw = (pose.yaw * Math.PI) / 180;
    const pitch = (pose.pitch * Math.PI) / 180;
    const xAxis = {
      x: Math.cos(roll) * axisLength * Math.cos(yaw) * mirrorSign,
      y: Math.sin(roll) * axisLength * Math.cos(yaw),
    };
    const yAxis = {
      x: Math.sin(roll) * axisLength * Math.cos(pitch) * mirrorSign,
      y: -Math.cos(roll) * axisLength * Math.cos(pitch),
    };
    const zAxis = {
      x: Math.sin(yaw) * axisLength * 1.18 * mirrorSign,
      y: Math.sin(pitch) * axisLength * 1.18,
    };

    const drawCenteredAxis = (vector, color, label) => {
      const startX = origin.x - vector.x * 0.62;
      const startY = origin.y - vector.y * 0.62;
      const endX = origin.x + vector.x;
      const endY = origin.y + vector.y;
      drawVectorArrow(ctx, startX, startY, endX, endY, color, Math.max(1.5, lineWidth * 0.64));
      ctx.save();
      ctx.fillStyle = color;
      ctx.shadowColor = "rgba(0, 0, 0, 0.9)";
      ctx.shadowBlur = 4;
      ctx.font = `700 ${Math.max(10, ctx.canvas.width / 90)}px system-ui, sans-serif`;
      ctx.fillText(label, endX + 5, endY - 5);
      ctx.restore();
    };

    drawCenteredAxis(xAxis, "#ff3b4a", "X");
    drawCenteredAxis(yAxis, "#41f37a", "Y");

    const zMagnitude = Math.hypot(zAxis.x, zAxis.y);
    if (zMagnitude > axisLength * 0.13) {
      drawVectorArrow(ctx, origin.x, origin.y, origin.x + zAxis.x, origin.y + zAxis.y, "#f4f07a", Math.max(1.8, lineWidth * 0.72));
      ctx.save();
      ctx.fillStyle = "#f4f07a";
      ctx.font = `700 ${Math.max(10, ctx.canvas.width / 90)}px system-ui, sans-serif`;
      ctx.fillText("Z", origin.x + zAxis.x + 5, origin.y + zAxis.y - 5);
      ctx.restore();
    } else {
      ctx.save();
      ctx.strokeStyle = "#f4f07a";
      ctx.fillStyle = "#f4f07a";
      ctx.lineWidth = Math.max(1.5, lineWidth * 0.62);
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, Math.max(6, lineWidth * 2.2), 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, Math.max(2.2, lineWidth * 0.72), 0, Math.PI * 2);
      ctx.fill();
      ctx.font = `700 ${Math.max(10, ctx.canvas.width / 90)}px system-ui, sans-serif`;
      ctx.fillText("Z", origin.x + 9, origin.y + 13);
      ctx.restore();
    }
  }

  function canvasXFromSource(sourceX, dimensions, scale, offsetX, canvasWidth, mirrorX) {
    const mappedX = sourceX * scale + offsetX;
    return mirrorX ? canvasWidth - mappedX : mappedX;
  }

  function calibrateOverlayPoint(point, canvasWidth, canvasHeight) {
    const calibration = state.overlayCalibration || DEFAULT_OVERLAY_CALIBRATION;
    return {
      x: (point.x - canvasWidth / 2) * calibration.scaleX + canvasWidth / 2 + calibration.offsetX,
      y: (point.y - canvasHeight / 2) * calibration.scaleY + canvasHeight / 2 + calibration.offsetY,
    };
  }

  function mapSourcePoint(sourceX, sourceY, dimensions, scale, offsetX, offsetY, canvasWidth, canvasHeight, mirrorX) {
    return calibrateOverlayPoint(
      {
        x: canvasXFromSource(sourceX, dimensions, scale, offsetX, canvasWidth, mirrorX),
        y: sourceY * scale + offsetY,
      },
      canvasWidth,
      canvasHeight,
    );
  }

  function mapSourceBox(box, dimensions, scale, offsetX, offsetY, canvasWidth, canvasHeight, mirrorX) {
    const [sourceX1, sourceY1, sourceX2, sourceY2] = box;
    const first = mapSourcePoint(sourceX1, sourceY1, dimensions, scale, offsetX, offsetY, canvasWidth, canvasHeight, mirrorX);
    const second = mapSourcePoint(sourceX2, sourceY2, dimensions, scale, offsetX, offsetY, canvasWidth, canvasHeight, mirrorX);
    return {
      x1: Math.min(first.x, second.x),
      y1: Math.min(first.y, second.y),
      x2: Math.max(first.x, second.x),
      y2: Math.max(first.y, second.y),
    };
  }

  function drawFaceDetectionFrame(ctx, bounds, lineWidth) {
    if (!bounds) return;
    const { x1, y1, x2, y2 } = bounds;
    const width = Math.max(1, x2 - x1);
    const height = Math.max(1, y2 - y1);
    const corner = Math.max(18, Math.min(width, height) * 0.16);
    const strokeWidth = Math.max(2.4, lineWidth * 0.78);

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.shadowColor = "rgba(96, 165, 250, 0.75)";
    ctx.shadowBlur = 14;
    ctx.strokeStyle = "rgba(96, 165, 250, 0.32)";
    ctx.lineWidth = Math.max(1.2, strokeWidth * 0.48);
    ctx.strokeRect(x1, y1, width, height);

    ctx.strokeStyle = "#70d7ff";
    ctx.lineWidth = strokeWidth;
    ctx.beginPath();
    ctx.moveTo(x1, y1 + corner);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1 + corner, y1);
    ctx.moveTo(x2 - corner, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + corner);
    ctx.moveTo(x2, y2 - corner);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2 - corner, y2);
    ctx.moveTo(x1 + corner, y2);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1, y2 - corner);
    ctx.stroke();

    const label = "FACE 01";
    if (state.annotationRecorder.capture) {
      state.annotationRecorder.capture.faceBox = {
        x1: captureNormX(ctx, x1),
        y1: captureNormY(ctx, y1),
        x2: captureNormX(ctx, x2),
        y2: captureNormY(ctx, y2),
        label,
      };
    }
    ctx.shadowBlur = 0;
    ctx.font = `800 ${Math.max(11, ctx.canvas.width / 82)}px system-ui, sans-serif`;
    const metrics = ctx.measureText(label);
    const labelHeight = Math.max(20, ctx.canvas.width / 60);
    const labelY = Math.max(labelHeight + 6, y1 - 6);
    ctx.fillStyle = "rgba(8, 16, 48, 0.76)";
    ctx.fillRect(x1, labelY - labelHeight, metrics.width + 16, labelHeight);
    ctx.fillStyle = "#dbeafe";
    ctx.fillText(label, x1 + 8, labelY - 6);
    ctx.restore();
  }

  function drawFaceVectors(ctx, overlay, dimensions, scale, offsetX, offsetY, mirrorX = false) {
    if (!overlay?.landmarks?.length) return;
    const mapPoint = (point) =>
      mapSourcePoint(
        point.x * dimensions.width,
        point.y * dimensions.height,
        dimensions,
        scale,
        offsetX,
        offsetY,
        ctx.canvas.width,
        ctx.canvas.height,
        mirrorX,
      );
    const { x1, y1, x2, y2 } = mapSourceBox(
      overlay.faceBox || [0, 0, 0, 0],
      dimensions,
      scale,
      offsetX,
      offsetY,
      ctx.canvas.width,
      ctx.canvas.height,
      mirrorX,
    );
    const lineWidth = Math.max(2, ctx.canvas.width / 360);
    const faceWidth = Math.max(1, x2 - x1);
    const faceHeight = Math.max(1, y2 - y1);
    const faceSpan = Math.min(faceWidth, faceHeight);
    const axisLength = Math.max(34, Math.min(92, Math.min(faceWidth, faceHeight) * 0.24));

    ctx.save();
    drawFaceDetectionFrame(ctx, { x1, y1, x2, y2 }, lineWidth);
    drawFacialFeaturePoints(ctx, overlay.landmarks, dimensions, scale, offsetX, offsetY, lineWidth, mirrorX);
    if (overlay.pose) drawHeadCoordinateSystem(ctx, overlay.pose, mapPoint, axisLength, lineWidth, mirrorX);

    [overlay.gaze?.left, overlay.gaze?.right].filter(Boolean).forEach((eyeRay) => {
      drawEyeOutline(ctx, eyeRay.outline, mapPoint, lineWidth);
      const origin = mapPoint(eyeRay.origin);
      const fallbackDirection = { x: eyeRay.x, y: eyeRay.y };
      drawGazeRay(ctx, origin, eyeRay.screenDirection || fallbackDirection, faceSpan, lineWidth, mirrorX);
    });

    if (overlay.pose) {
      const label = `P${Math.round(overlay.pose.pitch)}°  Y${Math.round(overlay.pose.yaw)}°  R${Math.round(overlay.pose.roll)}°`;
      if (state.annotationRecorder.capture) {
        state.annotationRecorder.capture.poseLabel = label;
      }
      ctx.shadowBlur = 0;
      ctx.font = `700 ${Math.max(12, ctx.canvas.width / 72)}px system-ui, sans-serif`;
      const metrics = ctx.measureText(label);
      const labelHeight = Math.max(22, ctx.canvas.width / 52);
      const labelX = Math.max(8, Math.min(ctx.canvas.width - metrics.width - 24, (x1 + x2 - metrics.width) / 2));
      const labelY = Math.min(ctx.canvas.height - 6, Math.max(labelHeight + 8, y2 + labelHeight + 2));
      ctx.fillStyle = "rgba(8, 22, 39, 0.78)";
      ctx.fillRect(labelX, labelY - labelHeight, metrics.width + 16, labelHeight);
      ctx.fillStyle = "#dbeafe";
      ctx.fillText(label, labelX + 8, labelY - 7);
    }
    ctx.restore();
  }

  function drawFacialFeaturePoints(ctx, landmarks, dimensions, scale, offsetX, offsetY, lineWidth, mirrorX = false) {
    if (!landmarks?.length) return;
    const mapLandmark = (point) =>
      mapSourcePoint(
        point.x * dimensions.width,
        point.y * dimensions.height,
        dimensions,
        scale,
        offsetX,
        offsetY,
        ctx.canvas.width,
        ctx.canvas.height,
        mirrorX,
      );
    const drawFeaturePath = (indexes, closed = false) => {
      const points = indexes.map((index) => landmarks[index]).filter(Boolean).map(mapLandmark);
      if (points.length < 2) return;
      if (state.annotationRecorder.capture) {
        state.annotationRecorder.capture.paths.push(points.map((point) => [captureNormX(ctx, point.x), captureNormY(ctx, point.y)]));
      }
      ctx.beginPath();
      points.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      if (closed) ctx.closePath();
      ctx.stroke();
    };
    const pointRadius = Math.max(2.2, Math.min(4.8, lineWidth * 1.12));
    const centerRadius = pointRadius * 1.42;
    const keyPoints = new Set([1, 4, 13, 14, 33, 133, 152, 263, 362, 468, 473]);
    ctx.save();
    ctx.strokeStyle = "rgba(96, 165, 250, 0.7)";
    ctx.lineWidth = Math.max(1.2, lineWidth * 0.38);
    ctx.shadowColor = "rgba(37, 99, 235, 0.65)";
    ctx.shadowBlur = 6;
    drawFeaturePath([10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109], true);
    drawFeaturePath([33, 160, 158, 133, 153, 144], true);
    drawFeaturePath([362, 385, 387, 263, 373, 380], true);
    drawFeaturePath([168, 6, 197, 195, 5, 4]);
    drawFeaturePath([98, 97, 2, 326, 327]);
    drawFeaturePath([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146], true);
    ctx.shadowColor = "rgba(37, 99, 235, 0.82)";
    ctx.shadowBlur = 7;
    FACIAL_FEATURE_POINTS.forEach((index) => {
      const point = landmarks[index];
      if (!point) return;
      const { x, y } = mapLandmark(point);
      if (state.annotationRecorder.capture) {
        state.annotationRecorder.capture.dots.push([captureNormX(ctx, x), captureNormY(ctx, y), keyPoints.has(index) ? 1 : 0]);
      }
      const radius = keyPoints.has(index) ? centerRadius : pointRadius;
      ctx.beginPath();
      ctx.fillStyle = keyPoints.has(index) ? "#dbeafe" : "#3b82f6";
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.strokeStyle = "rgba(248, 250, 252, 0.95)";
      ctx.lineWidth = Math.max(0.9, lineWidth * 0.22);
      ctx.arc(x, y, radius + 0.9, 0, Math.PI * 2);
      ctx.stroke();
    });
    ctx.restore();
  }

  function drawYoloDetectionBoxes(ctx, detections, dimensions, scale, offsetX, offsetY, mirrorX = false) {
    const yoloDetections = detections.filter(
      (detection) => detection.key === "phone" && detection.box?.length === 4 && detection.source !== "face-vector" && detection.source !== "6drepnet",
    );
    if (!yoloDetections.length) return;

    const lineWidth = Math.max(2.4, ctx.canvas.width / 420);
    yoloDetections.forEach((detection, index) => {
      const { x1, y1, x2, y2 } = mapSourceBox(
        detection.box,
        dimensions,
        scale,
        offsetX,
        offsetY,
        ctx.canvas.width,
        ctx.canvas.height,
        mirrorX,
      );
      const width = Math.max(1, x2 - x1);
      const height = Math.max(1, y2 - y1);
      const label = `Phone Call ${Math.round(detection.confidence * 100)}%`;
      const labelSize = Math.max(13, ctx.canvas.width / 86);
      const labelHeight = Math.max(22, labelSize * 1.65);
      if (state.annotationRecorder.capture) {
        state.annotationRecorder.capture.redBoxes.push({
          x1: captureNormX(ctx, x1),
          y1: captureNormY(ctx, y1),
          x2: captureNormX(ctx, x2),
          y2: captureNormY(ctx, y2),
          label,
        });
      }

      ctx.save();
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = "#ff4651";
      ctx.fillStyle = "rgba(255, 70, 81, 0.14)";
      ctx.lineWidth = lineWidth;
      ctx.shadowColor = "rgba(255, 70, 81, 0.82)";
      ctx.shadowBlur = 16;
      ctx.strokeRect(x1, y1, width, height);
      ctx.shadowBlur = 0;
      ctx.fillRect(x1, y1, width, height);

      ctx.font = `800 ${labelSize}px system-ui, sans-serif`;
      const textMetrics = ctx.measureText(label);
      const labelWidth = textMetrics.width + 16;
      const labelX = clamp(x1, 4, Math.max(4, ctx.canvas.width - labelWidth - 4));
      const labelY = y1 - labelHeight > 2 ? y1 - labelHeight : y1 + 2;
      ctx.fillStyle = "rgba(104, 21, 39, 0.88)";
      ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
      ctx.fillStyle = "#ffe4e6";
      ctx.fillText(label, labelX + 8, labelY + labelHeight - 7);

      const corner = Math.max(12, Math.min(width, height) * 0.16);
      ctx.strokeStyle = "#ffe4e6";
      ctx.lineWidth = Math.max(1.4, lineWidth * 0.48);
      ctx.beginPath();
      ctx.moveTo(x1, y1 + corner);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + corner, y1);
      ctx.moveTo(x2 - corner, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + corner);
      ctx.moveTo(x2, y2 - corner);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2 - corner, y2);
      ctx.moveTo(x1 + corner, y2);
      ctx.lineTo(x1, y2);
      ctx.lineTo(x1, y2 - corner);
      ctx.stroke();

      if (index === 0) {
        ctx.fillStyle = "#ff4651";
        ctx.beginPath();
        ctx.arc(x2, y1, Math.max(4, lineWidth * 1.8), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    });
  }

  function drawResults(source, detections, cameraMode = false) {
    document.querySelectorAll(".sample-preview-media.active").forEach((media) => media.classList.remove("active"));
    document.querySelector("#annotationCanvas")?.classList.remove("active", "drawing-enabled");
    const dimensions = sourceDimensions(source);
    let scale;
    let offsetX = 0;
    let offsetY = 0;
    if (cameraMode) {
      ensureCameraCanvasSize();
      if (source === cameraFrameCanvas) {
        scale = 1;
      } else {
        ({ scale, offsetX, offsetY } = cameraMediaLayout(source, elements.canvas.width, elements.canvas.height));
      }
    } else {
      const maxWidth = 1280;
      scale = Math.min(1, maxWidth / dimensions.width);
      elements.canvas.width = Math.round(dimensions.width * scale);
      elements.canvas.height = Math.round(dimensions.height * scale);
    }
    context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    if (!cameraMode || source === cameraFrameCanvas) {
      context.drawImage(source, 0, 0, elements.canvas.width, elements.canvas.height);
    }
    context.lineWidth = Math.max(2, elements.canvas.width / 420);
    context.font = `700 ${Math.max(13, elements.canvas.width / 55)}px system-ui, sans-serif`;

    // Model input stays unmirrored; the mirrored display needs the overlay flipped back.
    const mirrorOverlay = cameraMode && cameraOverlayMirrorX();

    const annotationRecorder = state.annotationRecorder;
    const captureFrame = annotationRecorder.recording && cameraMode && Date.now() - annotationRecorder.lastCaptureAt >= ANNOTATION_CAPTURE_INTERVAL_MS
      ? { t: Date.now(), aw: context.canvas.width, ah: context.canvas.height, faceBox: null, poseLabel: null, paths: [], dots: [], gaze: [], redBoxes: [] }
      : null;
    annotationRecorder.capture = captureFrame;
    if (captureFrame) annotationRecorder.lastCaptureAt = captureFrame.t;

    drawYoloDetectionBoxes(context, detections, dimensions, scale, offsetX, offsetY, mirrorOverlay);
    drawFaceVectors(context, state.headPose.overlay, dimensions, scale, offsetX, offsetY, mirrorOverlay);

    if (annotationRecorder.capture) {
      annotationRecorder.frames.push(annotationRecorder.capture);
      annotationRecorder.capture = null;
    }
    annotationRecorder.lastCaptureFrame = captureFrame;
    elements.canvas.classList.add("active");
    elements.canvas.classList.toggle("camera-overlay", cameraMode);
    elements.placeholder.hidden = true;
  }

  function clearKeyRegions() {
    if (!elements.keyRegionPanel || !elements.keyRegionList) return;
    elements.keyRegionPanel.hidden = true;
    elements.keyRegionList.replaceChildren();
    if (elements.keyRegionCount) elements.keyRegionCount.textContent = "0 regions";
    state.keyRegionSignature = "";
    state.keyRegionHoldUntil = 0;
  }

  function keyRegionCandidates(detections) {
    const seen = new Set();
    return [...detections]
      .filter((item) => item.box?.length === 4 && item.box[2] > item.box[0] && item.box[3] > item.box[1])
      .sort((first, second) => {
        const riskOrder = Number(isAlarmDetection(second)) - Number(isAlarmDetection(first));
        return riskOrder || second.confidence - first.confidence;
      })
      .filter((item) => {
        if (seen.has(item.key)) return false;
        seen.add(item.key);
        return true;
      })
      .slice(0, 3);
  }

  function drawKeyRegion(canvas, source, box) {
    const dimensions = sourceDimensions(source);
    if (!dimensions.width || !dimensions.height) return;
    const boxWidth = Math.max(1, box[2] - box[0]);
    const boxHeight = Math.max(1, box[3] - box[1]);
    const padding = Math.max(boxWidth, boxHeight) * 0.12;
    let sourceX = Math.max(0, box[0] - padding);
    let sourceY = Math.max(0, box[1] - padding);
    let sourceWidth = Math.min(dimensions.width, box[2] + padding) - sourceX;
    let sourceHeight = Math.min(dimensions.height, box[3] + padding) - sourceY;
    const targetAspect = canvas.width / canvas.height;
    const sourceAspect = sourceWidth / sourceHeight;
    if (sourceAspect > targetAspect) {
      const fittedWidth = sourceHeight * targetAspect;
      sourceX += (sourceWidth - fittedWidth) / 2;
      sourceWidth = fittedWidth;
    } else {
      const fittedHeight = sourceWidth / targetAspect;
      sourceY += (sourceHeight - fittedHeight) / 2;
      sourceHeight = fittedHeight;
    }
    const cropContext = canvas.getContext("2d", { alpha: false });
    cropContext.imageSmoothingEnabled = true;
    cropContext.imageSmoothingQuality = "high";
    cropContext.clearRect(0, 0, canvas.width, canvas.height);
    cropContext.drawImage(source, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, canvas.width, canvas.height);
  }

  function updateKeyRegions(source, detections) {
    if (!elements.keyRegionPanel || !elements.keyRegionList) return;
    const candidates = source ? keyRegionCandidates(detections) : [];
    const now = Date.now();
    if (!candidates.length) {
      if (state.keyRegionHoldUntil > now) return;
      clearKeyRegions();
      return;
    }
    state.keyRegionHoldUntil = now + ALERT_HOLD_MS;
    const signature = candidates.map((item) => `${item.key}:${item.source}`).join("|");
    if (signature !== state.keyRegionSignature) {
      const slots = candidates.map(() => {
        const slot = document.createElement("article");
        slot.className = "key-region-item";
        const canvas = document.createElement("canvas");
        canvas.width = 240;
        canvas.height = 132;
        const meta = document.createElement("div");
        const label = document.createElement("span");
        const title = document.createElement("b");
        const note = document.createElement("small");
        const confidence = document.createElement("strong");
        note.textContent = "Realtime crop from source frame";
        label.append(title, note);
        meta.append(label, confidence);
        slot.append(canvas, meta);
        return slot;
      });
      elements.keyRegionList.replaceChildren(...slots);
      state.keyRegionSignature = signature;
    }
    [...elements.keyRegionList.children].forEach((slot, index) => {
      const detection = candidates[index];
      const meta = CLASS_META[detection.key] || { label: detection.key, risk: false };
      slot.classList.toggle("alert", isAlarmDetection(detection));
      slot.querySelector("b").textContent = meta.label;
      slot.querySelector("strong").textContent = `${Math.round(detection.confidence * 100)}%`;
      drawKeyRegion(slot.querySelector("canvas"), source, detection.box);
    });
    elements.keyRegionPanel.hidden = false;
    if (elements.keyRegionCount) elements.keyRegionCount.textContent = `${candidates.length} region${candidates.length > 1 ? "s" : ""}`;
  }

  function maxConfidence(detections, key) {
    return Math.max(0, ...detections.filter((item) => item.key === key).map((item) => item.confidence));
  }

  function toggleAlarm(element, active) {
    element?.classList.toggle("alarm", Boolean(active));
  }

  function sourceLabelFor(detection) {
    if (detection.source === "manual-control") return "Manual/Remote Control";
    if (detection.source === "6drepnet") return "6DRepNet Head Pose";
    if (detection.source === "head-pose" || detection.source === "face-vector") return "Gaze/Face Detection";
    return MODEL_NAMES[detection.source] || "Action Object Model";
  }

  function isAlarmDetection(detection) {
    if (!detection || detection.confidence < confidenceThresholdForKey(detection.key)) return false;
    if ((detection.source === "face-vector" || detection.source === "6drepnet") && !FACE_ALARM_KEYS.has(detection.key)) {
      return false;
    }
    return ALARM_DETECTION_KEYS.has(detection.key);
  }

  function alertConfirmMsFor(detectionOrKey) {
    const key = typeof detectionOrKey === "string" ? detectionOrKey : detectionOrKey?.key;
    if (key === "drowsy") return DROWSY_ALERT_CONFIRM_MS;
    return key === "gaze_off" ? GAZE_OFF_ALERT_CONFIRM_MS : ALERT_CONFIRM_MS;
  }

  function alertMissGraceMsFor(detectionOrKey) {
    const key = typeof detectionOrKey === "string" ? detectionOrKey : detectionOrKey?.key;
    return key === "phone" ? PHONE_MISS_GRACE_MS : ALERT_MISS_GRACE_MS;
  }

  function isConfirmedAlert(detection) {
    return Boolean(detection?.alertActive || detection?.retainedAlert);
  }

  function alertAudioForKey(key) {
    if (!ALERT_AUDIO_SOURCES[key]) return null;
    if (!state.alertAudio.clips.has(key)) {
      const audio = new Audio(ALERT_AUDIO_SOURCES[key]);
      audio.preload = "auto";
      state.alertAudio.clips.set(key, audio);
    }
    return state.alertAudio.clips.get(key);
  }

  function unlockAlertAudio() {
    if (state.alertAudio.unlocked) return Promise.resolve(true);
    if (state.alertAudio.unlocking) return state.alertAudio.unlocking;

    const clips = Object.keys(ALERT_AUDIO_SOURCES)
      .map((key) => alertAudioForKey(key))
      .filter(Boolean);
    clips.forEach((audio) => audio.load());

    const primer = clips[0];
    if (!primer) return Promise.resolve(false);
    const previousVolume = primer.volume;
    const previousMuted = primer.muted;
    primer.muted = false;
    primer.volume = 0.01;
    primer.currentTime = 0;

    state.alertAudio.unlocking = primer
      .play()
      .then(
        () =>
          new Promise((resolve) => {
            window.setTimeout(resolve, 90);
          }),
      )
      .then(() => {
        primer.pause();
        primer.currentTime = 0;
        primer.volume = previousVolume;
        primer.muted = previousMuted;
        state.alertAudio.unlocked = true;
        return true;
      })
      .catch((error) => {
        primer.volume = previousVolume;
        primer.muted = previousMuted;
        state.alertAudio.unlocked = false;
        console.warn("Alert audio unlock failed", error);
        return false;
      })
      .finally(() => {
        state.alertAudio.unlocking = null;
      });

    return state.alertAudio.unlocking;
  }

  function installAlertAudioGestureUnlock() {
    const unlock = () => {
      unlockAlertAudio();
    };
    document.addEventListener("pointerdown", unlock, { passive: true });
    document.addEventListener("keydown", unlock);
  }

  function preloadAlertAudio() {
    Object.keys(ALERT_AUDIO_SOURCES).forEach((key) => {
      const audio = alertAudioForKey(key);
      if (!audio) return;
      audio.load();
    });
  }

  function playConfirmedAlertAudio(alerts, now = Date.now()) {
    const candidates = alerts
      .filter((item) => ALERT_AUDIO_SOURCES[item.key])
      .sort((first, second) => {
        const priority = (ALERT_AUDIO_PRIORITY[second.key] || 0) - (ALERT_AUDIO_PRIORITY[first.key] || 0);
        return priority || second.confidence - first.confidence;
      });
    const selected = candidates[0];
    if (!selected) return;
    if (selected.manualControl) {
      if (state.alertAudio.lastManualKey === selected.key) return;
      state.alertAudio.lastManualKey = selected.key;
    }

    const lastForKey = state.alertAudio.lastPlayedAt.get(selected.key) || 0;
    const lastGlobal = state.alertAudio.lastPlayedAt.get("__global") || 0;
    if (now - lastForKey < ALERT_AUDIO_COOLDOWN_MS || now - lastGlobal < 900) return;

    if (NATIVE_INFER_MODE) {
      // 报警音频由板端 ffplay 直接播放(音量 100), 浏览器只上报 key;
      // 触发节奏(冷却/优先级/手动控制)沿用浏览器原逻辑, 保证行为一致
      state.alertAudio.lastPlayedAt.set(selected.key, now);
      state.alertAudio.lastPlayedAt.set("__global", now);
      postNativeAlert(selected.key);
      return;
    }

    const audio = alertAudioForKey(selected.key);
    if (!audio) return;
    if (state.alertAudio.current && state.alertAudio.current !== audio) {
      state.alertAudio.current.pause();
      state.alertAudio.current.currentTime = 0;
    }
    state.alertAudio.current = audio;
    state.alertAudio.lastPlayedAt.set(selected.key, now);
    state.alertAudio.lastPlayedAt.set("__global", now);
    audio.currentTime = 0;
    audio.play().catch((error) => {
      console.warn("Alert audio playback failed", error);
      state.alertAudio.unlocked = false;
      notify("Audio alert needs one more click on the page to enable sound.");
    });
  }

  const ANNOTATION_TRACK_STORAGE_KEY = "dms_annotation_track";
  const ANNOTATION_CAPTURE_INTERVAL_MS = 200;

  function captureNormX(ctx, value) {
    return Math.round((value / ctx.canvas.width) * 100000) / 100000;
  }

  function captureNormY(ctx, value) {
    return Math.round((value / ctx.canvas.height) * 100000) / 100000;
  }

  function updateRecordButton() {
    const button = elements.recordButton;
    if (!button) return;
    const recording = state.annotationRecorder.recording;
    button.classList.toggle("recording", recording);
    button.querySelector("span:last-child").textContent = recording ? "Stop Rec" : "Record";
    button.title = recording
      ? `Stop recording (${state.annotationRecorder.frames.length} annotation frames captured) and auto-save to storage/annotation-tracks/`
      : "Record blue/red/yellow annotations; auto-saves to storage/annotation-tracks/ on stop";
  }

  function startAnnotationRecording() {
    const recorder = state.annotationRecorder;
    if (recorder.recording) return;
    if (!state.cameraRunning) {
      notify("Start the camera before recording annotations");
      return;
    }
    recorder.recording = true;
    recorder.startedAt = Date.now();
    recorder.lastCaptureAt = 0;
    recorder.frames = [];
    recorder.capture = null;
    updateRecordButton();
    notify("Annotation recording started (blue/red/yellow overlays)");
  }

  async function stopAnnotationRecording() {
    const recorder = state.annotationRecorder;
    if (!recorder.recording) return;
    recorder.recording = false;
    recorder.capture = null;
    recorder.lastCaptureFrame = null;
    updateRecordButton();
    const frames = recorder.frames;
    recorder.frames = [];
    if (!frames.length) {
      notify("No annotation frames captured");
      return;
    }
    const payload = {
      schema: "vision-sentinel-annotation-track/1",
      recordedAt: new Date().toISOString(),
      durationMs: frames[frames.length - 1].t - frames[0].t,
      frames,
    };
    // 本地兜底:回放页同浏览器立即可用
    try {
      localStorage.setItem(ANNOTATION_TRACK_STORAGE_KEY, JSON.stringify(payload));
    } catch (error) {
      console.warn("Annotation track localStorage save failed", error);
    }
    // 落盘:POST 给后端写入 storage/annotation-tracks/ 下的独立 JSON 文件
    let fileName = null;
    try {
      const response = await fetch("/api/annotation-tracks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        const data = await response.json();
        fileName = data.name;
      } else {
        console.warn("Annotation track upload failed", await response.text());
      }
    } catch (error) {
      console.warn("Annotation track upload failed", error);
    }
    const seconds = Math.round(payload.durationMs / 1000);
    notify(fileName
      ? `Annotation recording saved (${frames.length} frames, ${seconds}s) → storage/annotation-tracks/${fileName}`
      : `Annotation recording kept in browser only (${frames.length} frames, ${seconds}s); backend unavailable`);
  }

  function updateResults(detections) {
    const sorted = detections.filter((item) => ACTIVE_DETECTION_KEYS.has(item.key)).sort((first, second) => second.confidence - first.confidence);
    const now = Date.now();
    const alertCandidates = sorted.filter((item) => isAlarmDetection(item) && (!state.manualControl.enabled || item.manualControl));
    const candidateKeys = new Set(alertCandidates.map((item) => item.key));
    alertCandidates.forEach((item) => {
      const startedAt = item.manualControl ? now - alertConfirmMsFor(item) : now;
      if (!state.alertSince.has(item.key)) state.alertSince.set(item.key, startedAt);
      state.alertLastSeen.set(item.key, now);
    });
    state.alertSince.forEach((startedAt, key) => {
      const lastSeen = state.alertLastSeen.get(key) || 0;
      if (!candidateKeys.has(key) && (state.manualControl.enabled || now - lastSeen > alertMissGraceMsFor(key))) {
        state.alertSince.delete(key);
        state.alertLastSeen.delete(key);
      }
    });
    const pendingAlerts = alertCandidates.filter((item) => now - (state.alertSince.get(item.key) || now) < alertConfirmMsFor(item));
    const liveAlerts = alertCandidates
      .filter((item) => now - (state.alertSince.get(item.key) || now) >= alertConfirmMsFor(item))
      .map((item) => ({ ...item, alertActive: true }));
    const liveAlertKeys = new Set(liveAlerts.map((item) => item.key));
    liveAlerts.forEach((item) => {
      state.alertHold.set(item.key, { ...item, heldUntil: now + ALERT_HOLD_MS });
    });
    const heldAlerts = [];
    state.alertHold.forEach((item, key) => {
      if (state.manualControl.enabled && !item.manualControl) {
        state.alertHold.delete(key);
      } else if (item.heldUntil <= now) {
        state.alertHold.delete(key);
      } else if (!liveAlertKeys.has(key)) {
        heldAlerts.push({ ...item, retainedAlert: true });
      }
    });
    const displayBase = sorted.map((item) => (liveAlertKeys.has(item.key) ? { ...item, alertActive: true } : item));
    const alarmDetections = [...liveAlerts, ...heldAlerts];
    playConfirmedAlertAudio(liveAlerts, now);
    const displayed = [...displayBase, ...heldAlerts].sort((first, second) => second.confidence - first.confidence);
    const pendingAlert = pendingAlerts[0];
    const pendingRemainingSeconds = pendingAlert
      ? Math.max(1, Math.ceil((alertConfirmMsFor(pendingAlert) - (now - (state.alertSince.get(pendingAlert.key) || now))) / 1000))
      : 0;
    const resultSignature = displayed
      .map((item) => `${item.key}:${Math.round(item.confidence * 20)}:${item.retainedAlert ? "hold" : item.alertActive ? "alert" : "live"}`)
      .join("|");
    if (resultSignature !== state.lastResultSignature) {
      state.lastResultSignature = resultSignature;
      elements.detectionCount.textContent = `${displayed.length} item${displayed.length > 1 ? "s" : ""}`;
      elements.results.innerHTML = displayed.length
        ? displayed
            .slice(0, 8)
            .map((detection) => {
              const meta = CLASS_META[detection.key] || { label: detection.key, color: "#0ea5e9" };
              const sourceLabel = sourceLabelFor(detection);
              const holdLabel = detection.retainedAlert ? " · alert hold" : "";
              return `<div class="inference-result-row${isConfirmedAlert(detection) ? " alert" : ""}"><i style="background:${meta.color}"></i><span><b>${meta.label}</b><small>${sourceLabel}${holdLabel}</small></span><strong>${Math.round(detection.confidence * 100)}%</strong></div>`;
            })
            .join("")
        : `<div class="empty-inference">No target above ${Math.round(CONFIDENCE_THRESHOLD * 100)}% confidence detected</div>`;
    }

    const drowsy = maxConfidence(alarmDetections, "drowsy");
    const safe = maxConfidence(sorted, "safe");
    const headDown = maxConfidence(alarmDetections, "head_down");
    const danger = Math.max(
      drowsy,
      maxConfidence(alarmDetections, "head_down"),
      maxConfidence(alarmDetections, "gaze_off"),
      maxConfidence(alarmDetections, "phone"),
    );
    const fatigue = Math.round(drowsy * 100);
    const attention = sorted.length ? Math.round(Math.max(safe, 1 - danger) * 100) : null;
    const hasDetections = sorted.length > 0;
    const monitorAlarm = alarmDetections.length > 0;
    const fatigueAlarm = fatigue >= 35;
    const attentionAlarm = monitorAlarm || (attention !== null && attention < 75);
    elements.fatigueScore.textContent = String(fatigue);
    elements.fatigueLabel.textContent = sorted.length
      ? fatigue >= 70
        ? "High Fatigue Risk"
        : fatigue >= 35
          ? "Mild Fatigue Risk"
          : "No Obvious Fatigue"
      : "";
    elements.drowsyConfidence.textContent = `${fatigue}%`;
    elements.attentionScore.textContent = attention === null ? "--" : String(attention);
    elements.attentionLabel.textContent = attention === null ? "" : attention >= 75 ? "Focused" : attention >= 50 ? "Needs Attention" : "High Risk";
    elements.safeConfidence.textContent = `${Math.round(safe * 100)}%`;
    elements.safeConfidenceBar.style.width = `${Math.round(safe * 100)}%`;
    toggleAlarm(elements.fatigueReading, fatigueAlarm);
    toggleAlarm(elements.fatigueScore, fatigueAlarm);
    toggleAlarm(elements.fatigueLabel, fatigueAlarm);
    toggleAlarm(elements.drowsyConfidence, fatigueAlarm);
    toggleAlarm(elements.attentionReading, attentionAlarm);
    toggleAlarm(elements.attentionScore, attentionAlarm);
    toggleAlarm(elements.attentionLabel, attentionAlarm);
    toggleAlarm(elements.safeConfidence, attentionAlarm);
    toggleAlarm(elements.safeConfidence?.closest("label"), attentionAlarm);
    if (elements.driverTargetStatus) {
      elements.driverTargetStatus.textContent = state.headPose.metrics ? "Face Locked" : hasDetections ? "Target Detected" : "Not Detected";
    }
    if (elements.driverMonitorStatus) {
      elements.driverMonitorStatus.textContent = monitorAlarm ? "Alert" : pendingAlert ? `Delay ${pendingRemainingSeconds}s` : state.cameraRunning ? "Monitoring" : hasDetections ? "Analyzed" : "Standby";
      toggleAlarm(elements.driverMonitorStatus, monitorAlarm);
    }
    if (state.manualControl.enabled) {
      if (elements.driverTargetStatus) elements.driverTargetStatus.textContent = "Manual Control";
      if (elements.driverMonitorStatus) {
        elements.driverMonitorStatus.textContent = monitorAlarm ? "Manual Alert" : "Manual";
        toggleAlarm(elements.driverMonitorStatus, monitorAlarm);
      }
    }
    if (elements.headState) {
      elements.headState.textContent = headDown ? `${Math.round(headDown * 100)}% risk` : sorted.length ? "Pose Normal" : "";
      elements.headState.classList.toggle("risk", headDown >= HEAD_DOWN_CONFIDENCE_THRESHOLD);
    }
    if (elements.eyeState) {
      elements.eyeState.textContent = drowsy ? `${Math.round(drowsy * 100)}% risk` : sorted.length ? "Eyes Normal" : "";
      elements.eyeState.classList.toggle("risk", drowsy >= 0.35);
    }
    if (state.manualControl.enabled && elements.driverActionValue) {
      setTelemetryValue(elements.driverActionValue, state.manualControl.key ? `Manual: ${state.manualControl.label || state.manualControl.key}` : "--");
    }
    const statusDetections = [...displayed];
    elements.statusItems.forEach((item) => {
      const key = item.dataset.detectionKey;
      const statusDetection = statusDetections.find((detection) => detection.key === key);
      const pendingDetection = pendingAlerts.find((detection) => detection.key === key);
      const confidence = statusDetection?.confidence || 0;
      item.classList.toggle("active", confidence >= confidenceThresholdForKey(key) || Boolean(pendingDetection));
      item.classList.toggle("alert", Boolean(statusDetection && isConfirmedAlert(statusDetection)));
      const detail = item.querySelector("small");
      if (detail) {
        const pendingSeconds = pendingDetection
          ? Math.max(1, Math.ceil((alertConfirmMsFor(key) - (now - (state.alertSince.get(key) || now))) / 1000))
          : 0;
        detail.textContent = statusDetection?.retainedAlert
          ? "Alert Hold"
          : statusDetection?.alertActive
            ? `Alert ${Math.round(confidence * 100)}%`
            : pendingDetection
              ? `Delay ${pendingSeconds}s`
              : confidence
                ? `Confidence ${Math.round(confidence * 100)}%`
                : item.dataset.defaultLabel || detail.textContent.replace(/^Confidence.*$/, "Not Detected");
      }
    });

    if (state.annotationRecorder.recording && state.annotationRecorder.lastCaptureFrame) {
      state.annotationRecorder.lastCaptureFrame.alarm = [...new Set(alarmDetections.map((item) => item.key))];
    }

    const primary = alarmDetections[0] || sorted.find((item) => !SECONDARY_STATUS_KEYS.has(item.key)) || sorted[0];
    if (primary && elements.labelSelect && ANNOTATION_LABELS[primary.key]) {
      elements.labelSelect.value = ANNOTATION_LABELS[primary.key];
    }
    updateTimeline(primary);
  }

  function updateTimeline(primary) {
    if (!primary) return;
    const meta = CLASS_META[primary.key] || { label: primary.key, risk: false };
    const alarm = isConfirmedAlert(primary);
    const timestamp = new Date().toLocaleTimeString("en-US", { hour12: false });
    const now = Date.now();
    const signature = primary.key;
    if (state.lastTimelineKey === signature && now - state.lastTimelineAt < 5000) return;
    state.lastTimelineKey = signature;
    state.lastTimelineAt = now;
    state.timelineItems.unshift({
      signature,
      tone: alarm ? (primary.confidence >= 0.7 ? "danger" : "warning") : "success",
      title: meta.label,
      description: `${sourceLabelFor(primary)} confidence ${Math.round(primary.confidence * 100)}%.`,
      timestamp,
    });
    state.timelineItems = state.timelineItems.slice(0, 4);
    elements.timeline.innerHTML = state.timelineItems
      .map((item) => `<div class="event ${item.tone}"><b>${item.title}</b><p>${item.description}</p><time>${item.timestamp}</time></div>`)
      .join("");
  }

  async function infer(source) {
    if (state.inferencing) return;
    state.inferencing = true;
    const startedAt = performance.now();
    try {
      const cameraMode = state.cameraRunning && source === elements.cameraVideo;
      const displaySource = cameraMode ? prepareCameraDisplayFrame(source) || source : source;
      const inferenceSource = cameraMode && displaySource === cameraFrameCanvas ? cameraModelCanvas : displaySource;
      const headPosePromise = cameraMode
        ? Promise.resolve(state.headPose.faceDetections)
        : analyzeHeadPose(inferenceSource).catch((error) => {
            console.warn("Head pose inference failed", error);
            return [];
      });
      await ensureModels();
      let objectDetections;
      if (cameraMode && NATIVE_INFER_MODE) {
        // 原生模式: 检测来自板端 NPU(WebSocket 缓存), 坐标已映射到当前帧尺寸;
        // 后续融合链路(suppressOverlaps/报警/UI)与浏览器推理完全一致
        objectDetections = nativeObjectDetections();
      } else {
        const prep = preprocess(inferenceSource);
        objectDetections = [];
        for (const modelName of selectedModels()) {
          const session = state.sessions[modelName];
          const outputs = await session.run({ [session.inputNames[0]]: prep.tensor });
          objectDetections.push(...parseDetections(outputs, prep, modelName));
        }
      }
      const faceDetections = await headPosePromise;
      if (cameraMode) {
        state.headPose.objectDetections = objectDetections;
        if (!NATIVE_INFER_MODE) {
          // ROI 复检依赖浏览器端 COCO 会话, native 模式由板端全帧 COCO 承担手机检测
          objectDetections.push(...(await detectPhoneAroundFace(inferenceSource, objectDetections)));
        }
      }
      const merged = applyManualControl(suppressOverlaps([...suppressModelDrowsy(objectDetections), ...faceDetections]));
      drawResults(displaySource, merged, cameraMode);
      updateKeyRegions(cameraMode ? cameraModelCanvas : displaySource, merged);
      updateResults(merged);
      const latency = Math.round(performance.now() - startedAt);
      elements.speed.textContent = cameraMode && NATIVE_INFER_MODE
        ? `NPU ${Math.round(state.nativeInfer.inferenceMs)} ms · native${state.nativeInfer.connected ? "" : " (reconnecting)"}`
        : `Inference ${latency} ms · ${state.onnxBackend === "webgpu" ? "WebGPU" : "WASM (CPU)"}${cameraMode ? " · smooth video" : ""}`;
      setTelemetryValue(elements.topLatency, `${latency} ms`);
      if (state.cameraStartedAt) {
        const elapsedSeconds = Math.floor((Date.now() - state.cameraStartedAt) / 1000);
        const minutes = String(Math.floor(elapsedSeconds / 60)).padStart(2, "0");
        const seconds = String(elapsedSeconds % 60).padStart(2, "0");
        setTelemetryValue(elements.topRunTime, `${minutes}:${seconds}`);
      }
    } finally {
      state.inferencing = false;
      setLoading(false);
    }
  }

  async function cameraFaceLoop() {
    if (!state.cameraRunning) return;
    if (!state.headPose.tracking && elements.cameraVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      state.headPose.tracking = true;
      try {
        const displaySource = prepareCameraDisplayFrame(elements.cameraVideo) || elements.cameraVideo;
        const analysisSource = displaySource === cameraFrameCanvas ? cameraModelCanvas : displaySource;
        const faceDetections = await analyzeHeadPose(analysisSource);
        if (!state.cameraRunning) return;
        state.headPose.faceDetections = faceDetections;
        const merged = applyManualControl(suppressOverlaps([...suppressModelDrowsy(state.headPose.objectDetections), ...state.headPose.faceDetections]));
        drawResults(displaySource, merged, true);
        updateKeyRegions(analysisSource, merged);
        updateResults(merged);
        if (state.manualControl.enabled) {
          elements.speed.textContent = "Manual alert control · live detection running";
          setTelemetryValue(elements.topLatency, "-- ms");
        }
      } catch (error) {
        console.warn("Face vector tracking failed", error);
      } finally {
        state.headPose.tracking = false;
      }
    }
    window.setTimeout(cameraFaceLoop, FACE_TRACK_INTERVAL_MS);
  }

  function cameraCaptureUnavailableMessage() {
    if (NATIVE_INFER_MODE) {
      if (window.location.protocol === "https:") {
        return `native 模式当前页面是 https, MJPEG/WS 会被浏览器拦截(mixed content)。请改用 http 打开本页(或给原生服务配置 https)。`;
      }
      return "";
    }
    if (window.location.protocol === "file:") {
      return `当前是 file:// 直接打开，摄像头识别依赖本地服务加载 ONNX/WASM/模型文件。请在终端运行 npm run start:https，然后打开 ${localServerHintUrl()}`;
    }
    if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function") return "";
    const hash = window.location.hash || "#live-detection";
    const currentPort = window.location.port || "8000";
    const hostname = window.location.hostname;
    const localhostUrl = `http://127.0.0.1:${currentPort}/index.html${hash}`;
    if (!window.isSecureContext) {
      const lanHint = hostname && hostname !== "127.0.0.1" && hostname !== "localhost" ? `https://${hostname}:<https-port>/index.html${hash}` : "";
      return lanHint
        ? `Camera requires HTTPS or localhost. Current LAN HTTP page is not secure. Start HTTPS with npm run start:https, then open the HTTPS LAN URL shown in the terminal, for example ${lanHint}. On this Mac, ${localhostUrl} can also use the camera.`
        : `Camera requires HTTPS or localhost. Open ${localhostUrl} on this Mac, or start HTTPS with npm run start:https for LAN camera access.`;
    }
    return "Camera capture is not available in this browser. Check browser support and camera permissions.";
  }

  function stopCamera() {
    state.cameraRunning = false;
    state.cameraStartedAt = 0;
    if (NATIVE_INFER_MODE) stopNativeInfer();
    if (state.alertAudio.current) {
      state.alertAudio.current.pause();
      state.alertAudio.current.currentTime = 0;
      state.alertAudio.current = null;
    }
    state.stream?.getTracks().forEach((track) => track.stop());
    state.stream = null;
    elements.cameraVideo.srcObject = null;
    elements.cameraVideo.classList.remove("active");
    elements.canvas.classList.remove("active", "camera-overlay");
    context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    clearKeyRegions();
    state.phoneRoiLastRunAt = 0;
    state.phoneRoiLastDetection = null;
    state.headPose.overlay = null;
    state.headPose.metrics = null;
    state.headPose.smoothedFaceBox = null;
    state.headPose.faceBoxLastSeenAt = 0;
    state.headPose.downSince = 0;
    state.headPose.eyeClosedSince = 0;
    state.headPose.gazeOffSince = 0;
    state.headPose.gazeOffHoldUntil = 0;
    state.headPose.tracking = false;
    state.headPose.faceDetections = [];
    state.headPose.objectDetections = [];
    updateFaceTelemetryUI(null);
    updateResults([]);
    setTelemetryValue(elements.topVisionState, "Standby");
    setTelemetryValue(elements.topRunTime, "00:00");
    setTelemetryValue(elements.topLatency, "-- ms");
    elements.placeholder.hidden = false;
    elements.cameraButton.querySelector("span:last-child").textContent = "Start Camera";
    elements.cameraButton.classList.remove("danger-action");
  }

  async function cameraLoop() {
    if (!state.cameraRunning) return;
    if (elements.cameraVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      try {
        await infer(elements.cameraVideo);
      } catch (error) {
        stopCamera();
        elements.runtimeLabel.textContent = "Inference Failed";
        notify(error.message);
        return;
      }
    }
    window.setTimeout(cameraLoop, CAMERA_INTERVAL_MS);
  }

  async function toggleCamera() {
    if (state.cameraRunning) {
      stopCamera();
      notify("Camera detection stopped");
      return;
    }
    const cameraIssue = cameraCaptureUnavailableMessage();
    if (cameraIssue) {
      setLoading(false);
      elements.runtimeLabel.textContent = "Camera Unavailable";
      notify(cameraIssue);
      return;
    }
    try {
      setLoading(true, NATIVE_INFER_MODE ? "Connecting native NPU service..." : "Requesting camera permission...");
      if (NATIVE_INFER_MODE) {
        await startNativeCameraStream();
        startNativeWebSocket();
      } else {
        unlockAlertAudio();
        state.stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: CAMERA_CAPTURE_WIDTH, max: CAMERA_CAPTURE_MAX_WIDTH },
            height: { ideal: CAMERA_CAPTURE_HEIGHT, max: CAMERA_CAPTURE_MAX_HEIGHT },
            frameRate: { ideal: CAMERA_CAPTURE_FPS, max: CAMERA_CAPTURE_FPS },
          },
          audio: false,
        });
        elements.cameraVideo.srcObject = state.stream;
        await elements.cameraVideo.play();
      }
      state.cameraRunning = true;
      state.cameraStartedAt = Date.now();
      elements.canvas.classList.remove("active", "camera-overlay");
      context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
      elements.cameraVideo.classList.add("active");
      elements.cameraButton.querySelector("span:last-child").textContent = "Stop Camera";
      elements.cameraButton.classList.add("danger-action");
      elements.placeholder.hidden = true;
      if (elements.driverMonitorStatus) {
        elements.driverMonitorStatus.textContent = "Monitoring";
        elements.driverMonitorStatus.classList.remove("alarm");
      }
      if (elements.driverTargetStatus) elements.driverTargetStatus.textContent = "Detecting";
      setTelemetryValue(elements.topVisionState, "Tracking");
      elements.runtimeLabel.textContent = `${MODEL_NAMES[elements.modelSelect.value]} · Live`;
      setLoading(false);
      cameraFaceLoop();
      window.setTimeout(cameraLoop, CAMERA_MODEL_START_DELAY_MS);
    } catch (error) {
      stopCamera();
      setLoading(false);
      notify(error.name === "NotAllowedError" ? "Camera permission was not granted" : `Camera startup failed: ${error.message}`);
    }
  }

  function stopNativeInfer() {
    stopNativeWebSocket();
    stopNativeCameraStream();
    state.nativeInfer.models = {};
    state.nativeInfer.inferenceMs = 0;
  }

  if (elements.imageInput) {
    elements.imageInput.addEventListener("change", async (event) => {
      const file = event.target.files?.[0];
      if (!file) return;
      stopCamera();
      try {
        unlockAlertAudio();
        setLoading(true, "Reading image and loading models...");
        state.lastSource = await loadImage(file);
        await infer(state.lastSource);
        notify("Image detection complete");
      } catch (error) {
        setLoading(false);
        elements.runtimeLabel.textContent = "Inference Failed";
        notify(error.message);
      } finally {
        event.target.value = "";
      }
    });
  }

  elements.cameraButton.addEventListener("click", toggleCamera);
  if (elements.recordButton) {
    elements.recordButton.addEventListener("click", () => {
      if (state.annotationRecorder.recording) stopAnnotationRecording();
      else startAnnotationRecording();
    });
  }

  // 点击 Standby 状态徽章:导入 JSON 并跳转回放页完整放映
  const REPLAY_PAGE_PATH = "/annotation-replay.html";
  function setupReplayImport() {
    const modal = document.querySelector("#replayImportModal");
    if (!modal) return;
    const drop = modal.querySelector("#replayImportDrop");
    const fileInput = modal.querySelector("#replayImportFile");
    const info = modal.querySelector("#replayImportInfo");
    const playBtn = modal.querySelector("#replayImportPlay");
    const closeBtn = modal.querySelector("#replayImportClose");
    let payload = null;

    const openModal = () => {
      modal.hidden = false;
      payload = null;
      info.hidden = true;
      info.classList.remove("error");
      playBtn.hidden = true;
      fileInput.value = "";
    };
    const closeModal = () => { modal.hidden = true; };
    const showInfo = (message, isError) => {
      info.hidden = false;
      info.classList.toggle("error", Boolean(isError));
      info.textContent = message;
    };
    const acceptPayload = (data, sourceName) => {
      if (!data || !Array.isArray(data.frames) || !data.frames.length) {
        payload = null;
        playBtn.hidden = true;
        showInfo("文件格式不对:需要包含非空 frames 数组的标注录制 JSON", true);
        return;
      }
      payload = data;
      const seconds = Math.round((data.durationMs || (data.frames[data.frames.length - 1].t - data.frames[0].t)) / 1000);
      showInfo(`✓ ${sourceName}:${data.frames.length} 帧 · 时长约 ${seconds} 秒`, false);
      playBtn.hidden = false;
    };
    const readFile = (file) => {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          acceptPayload(JSON.parse(String(reader.result)), file.name);
        } catch (error) {
          payload = null;
          playBtn.hidden = true;
          showInfo(`读取失败:${error.message}`, true);
        }
      };
      reader.onerror = () => showInfo("文件读取失败", true);
      reader.readAsText(file);
    };
    const startPlayback = () => {
      if (!payload) return;
      try {
        localStorage.setItem(ANNOTATION_TRACK_STORAGE_KEY, JSON.stringify(payload));
      } catch (error) {
        notify(`播放数据过大,无法写入浏览器存储:${error.message}`);
        return;
      }
      window.location.href = `${REPLAY_PAGE_PATH}?source=local`;
    };

    if (elements.driverMonitorStatus) {
      elements.driverMonitorStatus.addEventListener("click", openModal);
    }
    closeBtn.addEventListener("click", closeModal);
    modal.addEventListener("click", (event) => { if (event.target === modal) closeModal(); });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !modal.hidden) closeModal(); });
    fileInput.addEventListener("change", () => readFile(fileInput.files?.[0]));
    ["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach((type) => drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.remove("dragover");
    }));
    drop.addEventListener("drop", (event) => {
      const file = event.dataTransfer?.files?.[0];
      readFile(file);
    });
    playBtn.addEventListener("click", startPlayback);
  }
  setupReplayImport();
  elements.modelSelect.addEventListener("change", async () => {
    elements.runtimeLabel.textContent = `${MODEL_NAMES[elements.modelSelect.value]} pending`;
    if (!state.cameraRunning && state.lastSource) {
      try {
        setLoading(true, "Switching inference model...");
        await infer(state.lastSource);
      } catch (error) {
        notify(error.message);
      }
    }
  });
  installManualControlShortcuts();
  startManualControlPolling();
  startMqttWebSocketControl();
  window.addEventListener("beforeunload", () => {
    if (state.manualControl.pollTimer) window.clearInterval(state.manualControl.pollTimer);
    stopMqttWebSocketControl();
    stopCamera();
  });
  preloadAlertAudio();
  installAlertAudioGestureUnlock();
  updateFaceTelemetryUI(null);
  verifyModelFiles();
})();
