/* ==========================================
   main.js — EngageAI Dashboard Frontend Logic
   ========================================== */

"use strict";

// ─────────────────────────────────────────────
// DOM references
// ─────────────────────────────────────────────
const dropZone       = document.getElementById("dropZone");
const videoInput     = document.getElementById("videoInput");
const dropIcon       = document.getElementById("dropIcon");
const dropTitle      = document.getElementById("dropTitle");
const dropHint       = document.getElementById("dropHint");
const analyseBtn     = document.getElementById("analyseBtn");
const btnText        = document.getElementById("btnText");
const btnIcon        = document.getElementById("btnIcon");
const uploadSection  = document.getElementById("upload-section");
const processingCard = document.getElementById("processingCard");
const resultsSection = document.getElementById("resultsSection");
const stageLabel     = document.getElementById("stageLabel");
const progressBar    = document.getElementById("progressBar");
const progressPct    = document.getElementById("progressPct");
const progressStage  = document.getElementById("progressStage");
const studentsGrid   = document.getElementById("studentsGrid");
const chartsGrid     = document.getElementById("chartsGrid");
const downloadsList  = document.getElementById("downloadsList");
const resultVideo    = document.getElementById("resultVideo");
const errorBanner    = document.getElementById("errorBanner");
const errorText      = document.getElementById("errorText");
const errorClose     = document.getElementById("errorClose");
const newAnalysisBtn = document.getElementById("newAnalysisBtn");

// Demo Controls
const demoBtn        = document.getElementById("demoBtn");
const demoBanner     = document.getElementById("demoBanner");
const demoSwitchBtn  = document.getElementById("demoSwitchBtn");

// Metrics Hub DOM Elements
const execRateVal       = document.getElementById("execRateVal");
const execStatusPill    = document.getElementById("execStatusPill");
const sessionMetaPill   = document.getElementById("sessionMetaPill");
const distCountsSummary = document.getElementById("distCountsSummary");
const segEngaged        = document.getElementById("segEngaged");
const segPartial        = document.getElementById("segPartial");
const segDisengaged     = document.getElementById("segDisengaged");
const legEngVal         = document.getElementById("legEngVal");
const legPartVal        = document.getElementById("legPartVal");
const legDisVal         = document.getElementById("legDisVal");

const countEngaged      = document.getElementById("countEngaged");
const pctEngaged        = document.getElementById("pctEngaged");
const countPartial      = document.getElementById("countPartial");
const pctPartial        = document.getElementById("pctPartial");
const countDisengaged   = document.getElementById("countDisengaged");
const pctDisengaged     = document.getElementById("pctDisengaged");

const htFirstHalfVal    = document.getElementById("htFirstHalfVal");
const htSecondHalfVal   = document.getElementById("htSecondHalfVal");
const htArrow           = document.getElementById("htArrow");
const htDeltaVal        = document.getElementById("htDeltaVal");
const htTrendTag        = document.getElementById("htTrendTag");
const htInsightText     = document.getElementById("htInsightText");
const halftimeBadge     = document.getElementById("halftimeBadge");

const telemStudents     = document.getElementById("telemStudents");
const telemDuration     = document.getElementById("telemDuration");
const telemFrames       = document.getElementById("telemFrames");
const telemFps          = document.getElementById("telemFps");

// Students Hub DOM Elements
const studentsTableWrap = document.getElementById("studentsTableWrap");
const studentsTableBody = document.getElementById("studentsTableBody");
const viewTableBtn      = document.getElementById("viewTableBtn");
const viewCardsBtn      = document.getElementById("viewCardsBtn");
const studentSearch     = document.getElementById("studentSearch");
const studentSort       = document.getElementById("studentSort");
const filterAll         = document.getElementById("filterAll");
const filterEng         = document.getElementById("filterEng");
const filterPart        = document.getElementById("filterPart");
const filterDis         = document.getElementById("filterDis");
const fCountAll         = document.getElementById("fCountAll");
const fCountEng         = document.getElementById("fCountEng");
const fCountPart        = document.getElementById("fCountPart");
const fCountDis         = document.getElementById("fCountDis");

// Webcam DOM Elements
const tabUploadMode     = document.getElementById("tabUploadMode");
const tabWebcamMode     = document.getElementById("tabWebcamMode");
const webcamStudio      = document.getElementById("webcamStudio");
const webcamVideo       = document.getElementById("webcamVideo");
const recordedPlayback  = document.getElementById("recordedPlayback");
const webcamStatus      = document.getElementById("webcamStatus");
const camStatusDot      = document.getElementById("camStatusDot");
const camStatusText     = document.getElementById("camStatusText");
const webcamTimer       = document.getElementById("webcamTimer");
const webcamStartPrompt = document.getElementById("webcamStartPrompt");
const startCamBtn       = document.getElementById("startCamBtn");
const cameraSelect      = document.getElementById("cameraSelect");
const recordBtn         = document.getElementById("recordBtn");
const recordBtnText     = document.getElementById("recordBtnText");
const retakeBtn         = document.getElementById("retakeBtn");
const useRecordedBtn    = document.getElementById("useRecordedBtn");

// Settings
const confSlider  = document.getElementById("confThresh");
const engSlider   = document.getElementById("engThresh");
const seqSlider   = document.getElementById("seqBuf");
const minSlider   = document.getElementById("minFrames");
const confVal     = document.getElementById("confVal");
const engVal      = document.getElementById("engVal");
const seqVal      = document.getElementById("seqVal");
const minVal      = document.getElementById("minVal");

// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
let selectedFile = null;
let pollInterval = null;
let particleAnim = null;
let allStudentsData = [];
let currentFilter = "all";
let currentSearch = "";
let currentSort = "id_asc";
let currentView = "table"; // 'table' or 'cards'
let isDemoMode = false;

// Webcam State
let webcamStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let recordStartTime = 0;
let recordTimerInterval = null;
let recordedBlob = null;
let recordedVideoFile = null;
let currentInputMode = "upload"; // 'upload' or 'webcam'

// ─────────────────────────────────────────────
// Slider value display
// ─────────────────────────────────────────────
confSlider.addEventListener("input", () => { confVal.textContent = confSlider.value; });
engSlider.addEventListener("input",  () => { engVal.textContent  = engSlider.value;  });
seqSlider.addEventListener("input",  () => { seqVal.textContent  = seqSlider.value;  });
minSlider.addEventListener("input",  () => { minVal.textContent  = minSlider.value;  });

// ─────────────────────────────────────────────
// Drop zone interaction
// ─────────────────────────────────────────────
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelect(file);
});
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") videoInput.click();
});
videoInput.addEventListener("change", () => {
  if (videoInput.files[0]) handleFileSelect(videoInput.files[0]);
});

function handleFileSelect(file) {
  const allowed = ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime",
                   "video/x-matroska", "video/webm", "video/x-ms-wmv", "video/mov"];
  const ext = file.name.split(".").pop().toLowerCase();
  const validExts = ["mp4","avi","mov","mkv","webm"];

  if (!validExts.includes(ext)) {
    showError("Unsupported file type. Please upload MP4, AVI, MOV, MKV, or WEBM.");
    return;
  }
  selectedFile = file;
  const sizeMB = (file.size / 1024 / 1024).toFixed(1);
  dropIcon.textContent   = "✅";
  dropTitle.textContent  = file.name;
  dropHint.textContent   = `${sizeMB} MB — Ready to analyse`;
  dropZone.classList.add("has-file");
  analyseBtn.disabled    = false;
  btnText.textContent    = "⚡ Run Engagement Analysis";
}

// ─────────────────────────────────────────────
// Analysis Submit
// ─────────────────────────────────────────────
analyseBtn.addEventListener("click", startAnalysis);

async function startAnalysis() {
  if (!selectedFile) return;
  analyseBtn.disabled = true;
  hideError();

  // Stop camera feed so webcam hardware is released while running model
  stopWebcamStream();

  const formData = new FormData();
  formData.append("video", selectedFile);
  formData.append("conf_thresh",   confSlider.value);
  formData.append("eng_thresh",    engSlider.value);
  formData.append("seq_buf",       seqSlider.value);
  formData.append("min_frames",    minSlider.value);

  // Switch views
  uploadSection.classList.add("hidden");
  processingCard.classList.remove("hidden");
  resultsSection.classList.add("hidden");

  // Reset pipeline stages visual state
  maxStageIndex = -1;
  activeStageName = null;
  PIPELINE_ORDER.forEach(n => {
    const el = document.getElementById(PIPELINE_STAGES[n].el);
    el?.classList.remove("active", "done");
    const s = el?.querySelector(".ps-status");
    if (s) s.textContent = "waiting";
    if (PIPELINE_STAGES[n].conn) {
      const c = document.getElementById(PIPELINE_STAGES[n].conn);
      if (c) c.classList.remove("active", "done");
    }
  });

  // Start particles
  startParticles();

  try {
    const resp = await fetch("/upload", { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || "Upload failed");
    }
    const { job_id } = await resp.json();
    startPolling(job_id);
  } catch (err) {
    showError(err.message);
    resetToUpload();
  }
}

// ─────────────────────────────────────────────
// Live Webcam Studio Handlers
// ─────────────────────────────────────────────
if (tabUploadMode && tabWebcamMode) {
  tabUploadMode.addEventListener("click", () => switchInputMode("upload"));
  tabWebcamMode.addEventListener("click", () => switchInputMode("webcam"));
}

function switchInputMode(mode) {
  currentInputMode = mode;
  if (mode === "upload") {
    tabUploadMode.classList.add("active");
    tabUploadMode.setAttribute("aria-selected", "true");
    tabWebcamMode.classList.remove("active");
    tabWebcamMode.setAttribute("aria-selected", "false");
    dropZone.classList.remove("hidden");
    webcamStudio.classList.add("hidden");
    stopWebcamStream();
    if (selectedFile && !recordedVideoFile) {
      analyseBtn.disabled = false;
      btnText.textContent = "⚡ Run Engagement Analysis";
    } else {
      analyseBtn.disabled = !selectedFile;
      btnText.textContent = selectedFile ? "⚡ Run Engagement Analysis" : "Select a video to begin";
    }
  } else {
    tabWebcamMode.classList.add("active");
    tabWebcamMode.setAttribute("aria-selected", "true");
    tabUploadMode.classList.remove("active");
    tabUploadMode.setAttribute("aria-selected", "false");
    dropZone.classList.add("hidden");
    webcamStudio.classList.remove("hidden");
    populateCameraDevices();
    if (recordedVideoFile) {
      analyseBtn.disabled = false;
      btnText.textContent = "⚡ Run Engagement Analysis (Recorded Video)";
    }
  }
}

async function populateCameraDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === "videoinput");
    if (videoDevices.length > 0 && cameraSelect) {
      const cur = cameraSelect.value;
      cameraSelect.innerHTML = "";
      videoDevices.forEach((dev, idx) => {
        const opt = document.createElement("option");
        opt.value = dev.deviceId;
        opt.textContent = dev.label || `Camera ${idx + 1}`;
        if (dev.deviceId === cur) opt.selected = true;
        cameraSelect.appendChild(opt);
      });
    }
  } catch (e) {
    // Ignore enumeration errors until permission is granted
  }
}

async function startWebcamStream() {
  hideError();
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Webcam recording is not supported in this browser. Please use Chrome, Edge, Safari, or Firefox.");
    return;
  }

  const deviceId = cameraSelect?.value ? { exact: cameraSelect.value } : undefined;
  const constraints = {
    video: deviceId ? { deviceId, width: { ideal: 1280 }, height: { ideal: 720 } } : { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  };

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
    webcamVideo.srcObject = webcamStream;
    await webcamVideo.play();

    webcamStartPrompt.classList.add("hidden");
    webcamVideo.classList.remove("hidden");
    recordedPlayback.classList.add("hidden");

    camStatusDot.className = "status-dot dot-live";
    camStatusText.textContent = "🟢 Live Camera Active";
    recordBtn.disabled = false;
    recordBtnText.textContent = "Start Recording";
    recordBtn.classList.remove("is-recording", "hidden");
    retakeBtn.classList.add("hidden");
    useRecordedBtn.classList.add("hidden");

    // Populate camera labels now that permission is granted
    await populateCameraDevices();
  } catch (err) {
    showError("Camera access denied or unavailable: " + (err.message || err.name));
    camStatusDot.className = "status-dot dot-idle";
    camStatusText.textContent = "Camera Error / Denied";
  }
}

function stopWebcamStream() {
  if (recordTimerInterval) {
    clearInterval(recordTimerInterval);
    recordTimerInterval = null;
  }
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  if (webcamVideo) {
    webcamVideo.srcObject = null;
  }
  isRecording = false;
  if (camStatusDot) camStatusDot.className = "status-dot dot-idle";
  if (camStatusText) camStatusText.textContent = "Camera Inactive";
  if (webcamStartPrompt && !recordedVideoFile) webcamStartPrompt.classList.remove("hidden");
  if (recordBtn) {
    recordBtn.disabled = true;
    recordBtn.classList.remove("is-recording");
    recordBtnText.textContent = "Start Recording";
  }
  if (webcamTimer) webcamTimer.classList.add("hidden");
}

if (startCamBtn) {
  startCamBtn.addEventListener("click", startWebcamStream);
}

if (cameraSelect) {
  cameraSelect.addEventListener("change", () => {
    if (webcamStream) {
      stopWebcamStream();
      startWebcamStream();
    }
  });
}

if (recordBtn) {
  recordBtn.addEventListener("click", toggleRecording);
}

function toggleRecording() {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
}

function startRecording() {
  if (!webcamStream) return;
  recordedChunks = [];
  recordedBlob = null;
  recordedVideoFile = null;

  let mimeType = "video/webm;codecs=vp8";
  if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported) {
    if (MediaRecorder.isTypeSupported("video/mp4;codecs=avc1")) {
      mimeType = "video/mp4;codecs=avc1";
    } else if (MediaRecorder.isTypeSupported("video/mp4")) {
      mimeType = "video/mp4";
    } else if (MediaRecorder.isTypeSupported("video/webm;codecs=vp8")) {
      mimeType = "video/webm;codecs=vp8";
    } else if (MediaRecorder.isTypeSupported("video/webm")) {
      mimeType = "video/webm";
    }
  }

  try {
    mediaRecorder = new MediaRecorder(webcamStream, { mimeType });
  } catch (e) {
    mediaRecorder = new MediaRecorder(webcamStream);
  }

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      recordedChunks.push(e.data);
    }
  };

  mediaRecorder.onstop = handleRecordingStopped;

  mediaRecorder.start(100);
  isRecording = true;
  recordStartTime = Date.now();

  camStatusDot.className = "status-dot dot-rec";
  camStatusText.textContent = "● RECORDING LIVE";
  recordBtn.classList.add("is-recording");
  recordBtnText.textContent = "⏹ Stop Recording";
  webcamTimer.classList.remove("hidden");
  webcamTimer.textContent = "00:00";

  recordTimerInterval = setInterval(() => {
    const elapsedSec = Math.floor((Date.now() - recordStartTime) / 1000);
    const m = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
    const s = String(elapsedSec % 60).padStart(2, "0");
    webcamTimer.textContent = `${m}:${s}`;
  }, 500);
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;
  mediaRecorder.stop();
  if (recordTimerInterval) {
    clearInterval(recordTimerInterval);
    recordTimerInterval = null;
  }
  isRecording = false;
  camStatusDot.className = "status-dot dot-idle";
  camStatusText.textContent = "Processing Recording…";
}

function handleRecordingStopped() {
  const mimeType = mediaRecorder?.mimeType || "video/webm";
  recordedBlob = new Blob(recordedChunks, { type: mimeType });
  const ext = mimeType.includes("mp4") ? "mp4" : "webm";
  recordedVideoFile = new File([recordedBlob], `webcam_recording_${Date.now()}.${ext}`, { type: mimeType });
  selectedFile = recordedVideoFile;

  // Show recorded video in review player
  recordedPlayback.src = URL.createObjectURL(recordedBlob);
  recordedPlayback.classList.remove("hidden");
  webcamVideo.classList.add("hidden");

  camStatusDot.className = "status-dot dot-live";
  const durationSec = Math.max(1, Math.round((Date.now() - recordStartTime) / 1000));
  camStatusText.textContent = `Recorded ${durationSec}s · Ready to Analyse`;

  recordBtn.classList.add("hidden");
  retakeBtn.classList.remove("hidden");
  useRecordedBtn.classList.remove("hidden");

  analyseBtn.disabled = false;
  btnText.textContent = "⚡ Run Engagement Analysis (Recorded Video)";
}

if (retakeBtn) {
  retakeBtn.addEventListener("click", () => {
    recordedChunks = [];
    recordedBlob = null;
    recordedVideoFile = null;
    selectedFile = null;
    recordedPlayback.pause();
    recordedPlayback.src = "";
    recordedPlayback.classList.add("hidden");
    webcamVideo.classList.remove("hidden");

    recordBtn.classList.remove("hidden", "is-recording");
    recordBtn.disabled = false;
    recordBtnText.textContent = "Start Recording";
    retakeBtn.classList.add("hidden");
    useRecordedBtn.classList.add("hidden");
    webcamTimer.classList.add("hidden");

    camStatusDot.className = "status-dot dot-live";
    camStatusText.textContent = "🟢 Live Camera Active";

    analyseBtn.disabled = true;
    btnText.textContent = "Record or select a video to begin";
  });
}

if (useRecordedBtn) {
  useRecordedBtn.addEventListener("click", () => {
    if (!recordedVideoFile) return;
    selectedFile = recordedVideoFile;
    startAnalysis();
  });
}

// ─────────────────────────────────────────────
// Polling & Pipeline Visualization
// ─────────────────────────────────────────────
const PIPELINE_ORDER = ["yolo", "track", "effnet", "openface", "transformer"];
const PIPELINE_STAGES = {
  "yolo":        { el: "ps-yolo",        conn: null,   keywords: ["yolo", "detection"] },
  "track":       { el: "ps-track",       conn: "pc-1", keywords: ["bytetrack", "tracking", "track"] },
  "effnet":      { el: "ps-effnet",      conn: "pc-2", keywords: ["efficientnet", "visual", "effnet"] },
  "openface":    { el: "ps-openface",    conn: "pc-3", keywords: ["openface", "dynamics", "pose", "facial"] },
  "transformer": { el: "ps-transformer", conn: "pc-4", keywords: ["transformer", "predict", "summary", "chart", "report", "encode", "classif"] },
};

let activeStageName = null;
let maxStageIndex = -1;

function stageFromLabel(label) {
  const lower = label.toLowerCase();
  for (const name of PIPELINE_ORDER) {
    const cfg = PIPELINE_STAGES[name];
    if (cfg.keywords.some(k => lower.includes(k))) return name;
  }
  return null;
}

function updatePipelineVisuals(stageName) {
  const incomingIdx = PIPELINE_ORDER.indexOf(stageName);
  if (incomingIdx === -1) return;

  // Monotonic advancement: NEVER move backwards in the pipeline visual
  if (incomingIdx > maxStageIndex) {
    maxStageIndex = incomingIdx;
  }
  const currentIdx = maxStageIndex;

  PIPELINE_ORDER.forEach((n, i) => {
    const cfg = PIPELINE_STAGES[n];
    const el  = document.getElementById(cfg.el);
    const statusEl = el?.querySelector(".ps-status");

    if (i < currentIdx) {
      // Completed stages
      el.classList.remove("active");
      el.classList.add("done");
      if (statusEl) statusEl.textContent = "✓ done";
      if (cfg.conn) {
        const c = document.getElementById(cfg.conn);
        if (c) { c.classList.remove("active"); c.classList.add("done"); }
      }
    } else if (i === currentIdx) {
      // Currently executing stage
      el.classList.remove("done");
      el.classList.add("active");
      if (statusEl) statusEl.textContent = "⚙ running…";
      // Ensure connector leading to this stage is active
      const prevName = PIPELINE_ORDER[i - 1];
      if (prevName) {
        const prevConn = PIPELINE_STAGES[prevName].conn;
        if (prevConn) {
          const c = document.getElementById(prevConn);
          if (c) { c.classList.remove("done"); c.classList.add("active"); }
        }
      }
    } else {
      // Pending stages
      el.classList.remove("active", "done");
      if (statusEl) statusEl.textContent = "waiting";
      if (cfg.conn) {
        const c = document.getElementById(cfg.conn);
        if (c) { c.classList.remove("active", "done"); }
      }
    }
  });
}

function startPolling(jobId) {
  const stageEl = document.getElementById("stageLabel");
  maxStageIndex = -1;
  activeStageName = null;

  pollInterval = setInterval(async () => {
    try {
      const resp = await fetch(`/status/${jobId}`);
      if (!resp.ok) {
        // Network hiccup or container restart — keep progress intact
        return;
      }
      const job  = await resp.json();

      // Update progress bar
      const pct = job.progress || 0;
      progressBar.style.width = pct + "%";
      progressBar.setAttribute("aria-valuenow", pct);
      progressPct.textContent = pct + "%";
      progressStage.textContent = job.stage || "";
      stageLabel.textContent = job.stage || "";

      // Prioritize explicit stage_key sent from backend, fallback to label mapping
      const stageKey = job.stage_key || stageFromLabel(job.stage || "");
      if (stageKey && stageKey !== activeStageName) {
        activeStageName = stageKey;
        updatePipelineVisuals(stageKey);
      }

      if (job.status === "done") {
        clearInterval(pollInterval);
        // Mark all done
        PIPELINE_ORDER.forEach(n => {
          const cfg = PIPELINE_STAGES[n];
          const el  = document.getElementById(cfg.el);
          el.classList.remove("active"); el.classList.add("done");
          const s = el?.querySelector(".ps-status");
          if (s) s.textContent = "✓ done";
          if (cfg.conn) {
            const c = document.getElementById(cfg.conn);
            if (c) { c.classList.remove("active"); c.classList.add("done"); }
          }
        });
        progressBar.style.width = "100%";
        progressPct.textContent = "100%";
        stageLabel.textContent = "✅ Analysis complete!";
        stopParticles();
        setTimeout(() => renderResults(job.result), 600);
      }

      if (job.status === "error") {
        clearInterval(pollInterval);
        stopParticles();
        showError(job.error || "Processing failed. Check server logs.");
        resetToUpload();
      }
    } catch (err) {
      // network hiccup — keep polling
    }
  }, 800);
}

// ─────────────────────────────────────────────
// Demo Video Mode Handler
// ─────────────────────────────────────────────
if (demoBtn) {
  demoBtn.addEventListener("click", loadDemoSession);
}
if (demoSwitchBtn) {
  demoSwitchBtn.addEventListener("click", resetToUpload);
}

async function loadDemoSession() {
  hideError();
  isDemoMode = true;
  uploadSection.classList.add("hidden");
  processingCard.classList.remove("hidden");
  resultsSection.classList.add("hidden");

  stageLabel.textContent = "Loading pre-analyzed classroom demo session…";
  progressBar.style.width = "40%";
  progressPct.textContent = "40%";
  progressStage.textContent = "Loading cached model features & temporal tracks…";
  startParticles();

  // Monotonic pipeline animation for instant preview
  let step = 0;
  const stages = ["yolo", "track", "effnet", "openface", "transformer"];
  const iv = setInterval(() => {
    if (step < stages.length) {
      updatePipelineVisuals(stages[step]);
      step++;
      const p = Math.min(40 + step * 12, 95);
      progressBar.style.width = p + "%";
      progressPct.textContent = p + "%";
    }
  }, 120);

  try {
    const resp = await fetch("/demo");
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || "Could not load demo session");
    }
    const demoData = await resp.json();
    clearInterval(iv);

    PIPELINE_ORDER.forEach(n => {
      const cfg = PIPELINE_STAGES[n];
      const el  = document.getElementById(cfg.el);
      if (el) { el.classList.remove("active"); el.classList.add("done"); }
      const s = el?.querySelector(".ps-status");
      if (s) s.textContent = "✓ done";
      if (cfg.conn) {
        const c = document.getElementById(cfg.conn);
        if (c) { c.classList.remove("active"); c.classList.add("done"); }
      }
    });

    progressBar.style.width = "100%";
    progressPct.textContent = "100%";
    progressStage.textContent = "Demo analysis ready!";
    stopParticles();

    setTimeout(() => {
      renderResults(demoData, true);
    }, 400);
  } catch (err) {
    clearInterval(iv);
    stopParticles();
    showError("Demo load error: " + err.message);
    resetToUpload();
  }
}

// ─────────────────────────────────────────────
// Results Rendering
// ─────────────────────────────────────────────
function renderResults(result, isDemo = false) {
  processingCard.classList.add("hidden");
  resultsSection.classList.remove("hidden");

  // Show/hide demo banner
  if (isDemo || result.is_demo) {
    demoBanner.classList.remove("hidden");
  } else {
    demoBanner.classList.add("hidden");
  }

  renderKPI(result.kpi, result.halftime_analysis);
  renderVideo(result.video_url);
  setupStudentHub(result.student_summary || []);
  renderCharts(result.charts || {});
  renderDownloads(result);

  // Switch to video tab by default
  switchTab("video");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Organized Metrics Hub (Executive + Segmented Bar + 3 Panels) ─────────────
function renderKPI(kpi, halftime) {
  if (!kpi) return;

  const overall = kpi.overall_engagement ?? 0;
  execRateVal.textContent = overall.toFixed(1) + "%";

  // Executive Status Pill
  if (overall >= 65) {
    execStatusPill.className = "status-pill status-high";
    execStatusPill.textContent = "🟢 High Classroom Attention";
  } else if (overall >= 45) {
    execStatusPill.className = "status-pill status-moderate";
    execStatusPill.textContent = "🟡 Moderate Focus / Partial Fatigue";
  } else {
    execStatusPill.className = "status-pill status-low";
    execStatusPill.textContent = "🔴 Low Engagement / Intervention Needed";
  }

  const durationStr = (kpi.duration_sec ?? (kpi.processed_frames / (kpi.fps || 25)).toFixed(1)) + "s";
  sessionMetaPill.textContent = `${kpi.students_tracked ?? 0} Students · ${durationStr} · ${kpi.processed_frames ?? 0} Frames`;

  // Segmented 3-Way Distribution Bar
  const engPct  = kpi.engaged_pct ?? (kpi.students_tracked ? (kpi.engaged_students / kpi.students_tracked * 100) : 0);
  const partPct = kpi.partially_engaged_pct ?? (kpi.students_tracked ? (kpi.partially_engaged_students / kpi.students_tracked * 100) : 0);
  const disPct  = kpi.disengaged_pct ?? (kpi.students_tracked ? (kpi.disengaged_students / kpi.students_tracked * 100) : 0);

  segEngaged.style.width    = engPct.toFixed(1) + "%";
  segPartial.style.width    = partPct.toFixed(1) + "%";
  segDisengaged.style.width = disPct.toFixed(1) + "%";

  distCountsSummary.textContent = `${kpi.engaged_students ?? 0} Engaged (${engPct.toFixed(0)}%) · ${kpi.partially_engaged_students ?? 0} Partial (${partPct.toFixed(0)}%) · ${kpi.disengaged_students ?? 0} Disengaged (${disPct.toFixed(0)}%)`;

  legEngVal.textContent  = `${engPct.toFixed(1)}% (${kpi.engaged_students ?? 0})`;
  legPartVal.textContent = `${partPct.toFixed(1)}% (${kpi.partially_engaged_students ?? 0})`;
  legDisVal.textContent  = `${disPct.toFixed(1)}% (${kpi.disengaged_students ?? 0})`;

  // Panel 1: Breakdown
  countEngaged.textContent    = kpi.engaged_students ?? 0;
  pctEngaged.textContent      = engPct.toFixed(1) + "%";
  countPartial.textContent    = kpi.partially_engaged_students ?? 0;
  pctPartial.textContent      = partPct.toFixed(1) + "%";
  countDisengaged.textContent = kpi.disengaged_students ?? 0;
  pctDisengaged.textContent   = disPct.toFixed(1) + "%";

  // Panel 2: Half-Time Temporal Analysis (Req #2)
  const h1 = halftime?.first_half_engagement ?? kpi.first_half_engagement ?? overall;
  const h2 = halftime?.second_half_engagement ?? kpi.second_half_engagement ?? overall;
  const delta = halftime?.delta ?? kpi.halftime_delta ?? (h2 - h1);

  htFirstHalfVal.textContent  = h1.toFixed(1) + "%";
  htSecondHalfVal.textContent = h2.toFixed(1) + "%";

  const deltaSign = delta >= 0 ? "+" : "";
  htDeltaVal.textContent = `Δ ${deltaSign}${delta.toFixed(1)}%`;
  htDeltaVal.className = `ht-delta ${delta >= 0 ? "trend-up" : "trend-down"}`;
  htArrow.textContent = delta >= 0 ? "↗" : "↘";
  htArrow.className = `ht-arrow ${delta >= 0 ? "arrow-up" : "arrow-down"}`;

  htTrendTag.textContent = halftime?.trend_label || (delta >= 0 ? "📈 Attention Growth" : "📉 Attention Fatigue");
  htInsightText.textContent = halftime?.insight || "Attentive engagement was monitored across both halves of the session.";
  halftimeBadge.textContent = `Midpoint: ${halftime?.midpoint_sec ?? (kpi.duration_sec / 2).toFixed(1)}s`;

  // Panel 3: Session Telemetry
  telemStudents.textContent = kpi.students_tracked ?? 0;
  telemDuration.textContent = durationStr;
  telemFrames.textContent   = kpi.processed_frames ?? 0;
  telemFps.textContent      = (kpi.fps ?? 25.0).toFixed(1);
}

// Video
function renderVideo(url) {
  if (!url) return;
  resultVideo.querySelector("source").src = url;
  resultVideo.load();
}

// ── Per-Student Analysis Hub (Req #3) ────────────────────────────────────────
function setupStudentHub(students) {
  allStudentsData = students || [];

  // Update count badges on filter pills
  const nAll  = allStudentsData.length;
  const nEng  = allStudentsData.filter(s => s.overall_status === "Engaged").length;
  const nPart = allStudentsData.filter(s => s.overall_status === "Partially Engaged").length;
  const nDis  = allStudentsData.filter(s => s.overall_status === "Disengaged").length;

  fCountAll.textContent  = nAll;
  fCountEng.textContent  = nEng;
  fCountPart.textContent = nPart;
  fCountDis.textContent  = nDis;

  // Filter click handlers
  document.querySelectorAll(".filter-pill").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll(".filter-pill").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentFilter = btn.dataset.filter;
      applyStudentFiltersAndSort();
    };
  });

  // Search input handler
  studentSearch.oninput = () => {
    currentSearch = studentSearch.value.trim().toLowerCase();
    applyStudentFiltersAndSort();
  };

  // Sort dropdown handler
  studentSort.onchange = () => {
    currentSort = studentSort.value;
    applyStudentFiltersAndSort();
  };

  // View toggle handlers (Table vs Cards)
  viewTableBtn.onclick = () => {
    currentView = "table";
    viewTableBtn.classList.add("active");
    viewCardsBtn.classList.remove("active");
    studentsTableWrap.classList.remove("hidden");
    studentsGrid.classList.add("hidden");
  };

  viewCardsBtn.onclick = () => {
    currentView = "cards";
    viewCardsBtn.classList.add("active");
    viewTableBtn.classList.remove("active");
    studentsTableWrap.classList.add("hidden");
    studentsGrid.classList.remove("hidden");
  };

  applyStudentFiltersAndSort();
}

function applyStudentFiltersAndSort() {
  let list = [...allStudentsData];

  // 1. Status Filter
  if (currentFilter !== "all") {
    list = list.filter(s => s.overall_status === currentFilter);
  }

  // 2. Search Query
  if (currentSearch) {
    list = list.filter(s => {
      const label = (s.label || "").toLowerCase();
      const idStr = String(s.track_id);
      return label.includes(currentSearch) || idStr.includes(currentSearch);
    });
  }

  // 3. Sorting
  list.sort((a, b) => {
    switch (currentSort) {
      case "id_asc":
        return a.track_id - b.track_id;
      case "eng_desc":
        return (b.avg_engaged_prob || 0) - (a.avg_engaged_prob || 0);
      case "dis_desc":
        return (b.avg_disengaged_prob || 0) - (a.avg_disengaged_prob || 0);
      case "frames_desc":
        return (b.tracked_frames || 0) - (a.tracked_frames || 0);
      case "trend_desc":
        return (b.student_trend_delta || 0) - (a.student_trend_delta || 0);
      default:
        return a.track_id - b.track_id;
    }
  });

  renderStudentTable(list);
  renderStudentCards(list);
}

function renderStudentTable(list) {
  studentsTableBody.innerHTML = "";
  if (!list || list.length === 0) {
    studentsTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:32px;color:var(--text-muted)">No matching student records found.</td></tr>`;
    return;
  }

  list.forEach(s => {
    const isEngage = s.overall_status === "Engaged";
    const isPartial = s.overall_status === "Partially Engaged";
    const engPct   = ((s.avg_engaged_prob || 0) * 100).toFixed(1);
    const disPct   = ((s.avg_disengaged_prob || 0) * 100).toFixed(1);

    const badgeCls = isPartial ? "status-partial" : (isEngage ? "status-engaged" : "status-disengaged");
    const badgeTxt = isPartial ? "⚠️ Partially Engaged" : (isEngage ? "✅ Engaged" : "❌ Disengaged");

    // Temporal trend pill
    const h1Pct = s.first_half_eng_pct != null ? s.first_half_eng_pct.toFixed(0) + "%" : "--";
    const h2Pct = s.second_half_eng_pct != null ? s.second_half_eng_pct.toFixed(0) + "%" : "--";
    const delta = s.student_trend_delta != null ? s.student_trend_delta : 0;
    const sign  = delta >= 0 ? "+" : "";
    let trendPillCls = "trend-pill-steady";
    let trendIcon = "➡️";
    if (s.student_trend === "improving") {
      trendPillCls = "trend-pill-up";
      trendIcon = "📈";
    } else if (s.student_trend === "dropping") {
      trendPillCls = "trend-pill-down";
      trendIcon = "📉";
    }

    const tr = document.createElement("tr");
    tr.className = `student-row ${isPartial ? "row-partial" : (isEngage ? "row-engaged" : "row-disengaged")}`;
    tr.innerHTML = `
      <td>
        <div class="student-cell-profile">
          <div class="student-avatar ${isPartial ? "av-part" : (isEngage ? "av-eng" : "av-dis")}">#${s.track_id}</div>
          <div>
            <div class="student-name">${s.label}</div>
            <div class="student-sub">Track ID: ${s.track_id}</div>
          </div>
        </div>
      </td>
      <td>
        <span class="student-status-badge ${badgeCls}">${badgeTxt}</span>
      </td>
      <td>
        <div class="prob-matrix-pill">
          <span class="matrix-e">E: ${engPct}%</span>
          <span class="matrix-divider">|</span>
          <span class="matrix-d">D: ${disPct}%</span>
        </div>
      </td>
      <td>
        <div class="table-prob-meter">
          <div class="tpm-bar">
            <div class="tpm-fill fill-eng" style="width:${engPct}%"></div>
            <div class="tpm-fill fill-dis" style="width:${disPct}%"></div>
          </div>
          <div class="tpm-labels">
            <span style="color:var(--accent-green)">${engPct}% Engaged</span>
            <span style="color:var(--accent-red)">${disPct}% Disengaged</span>
          </div>
        </div>
      </td>
      <td>
        <div class="temporal-trend-cell">
          <span class="trend-steps">${h1Pct} → ${h2Pct}</span>
          <span class="trend-badge ${trendPillCls}">${trendIcon} ${sign}${delta.toFixed(0)}%</span>
        </div>
      </td>
      <td>
        <div class="frames-cell">
          <strong>${s.tracked_frames} frames</strong>
          <span class="frames-time">~${(s.tracked_frames / 25).toFixed(1)}s</span>
        </div>
      </td>
    `;
    studentsTableBody.appendChild(tr);
  });
}

function renderStudentCards(list) {
  studentsGrid.innerHTML = "";
  if (!list || list.length === 0) {
    studentsGrid.innerHTML = `<p style="color:var(--text-muted);grid-column:1/-1;text-align:center;padding:30px">No matching student records found.</p>`;
    return;
  }

  list.forEach(s => {
    const isEngage = s.overall_status === "Engaged";
    const isPartial = s.overall_status === "Partially Engaged";
    const engPct   = ((s.avg_engaged_prob || 0) * 100).toFixed(1);
    const disPct   = ((s.avg_disengaged_prob || 0) * 100).toFixed(1);

    const statusCls  = isPartial ? "status-partial" : (isEngage ? "status-engaged" : "status-disengaged");
    const statusText = isPartial ? "⚠️ Partially Engaged" : (isEngage ? "✅ Engaged" : "❌ Disengaged");
    const cardCls    = isPartial ? "partial-card" : (isEngage ? "engaged-card" : "disengaged-card");

    const h1 = s.first_half_eng_pct != null ? s.first_half_eng_pct.toFixed(0) + "%" : "--";
    const h2 = s.second_half_eng_pct != null ? s.second_half_eng_pct.toFixed(0) + "%" : "--";

    const card = document.createElement("div");
    card.className = `student-card ${cardCls}`;
    card.innerHTML = `
      <div class="student-header">
        <div class="student-id-wrap">
          <span class="student-id">${s.label}</span>
          <span class="student-meta-mini">ID #${s.track_id} · ${s.tracked_frames} frames</span>
        </div>
        <span class="student-status-badge ${statusCls}">${statusText}</span>
      </div>

      <!-- 2×1 Probability Matrix -->
      <div class="prob-matrix">
        <div class="prob-cell cell-engaged">
          <div class="cell-val">${engPct}%</div>
          <div class="cell-name">Engaged Prob</div>
        </div>
        <div class="prob-cell cell-disengaged">
          <div class="cell-val">${disPct}%</div>
          <div class="cell-name">Disengaged Prob</div>
        </div>
      </div>

      <!-- Dual probability bars -->
      <div style="margin-top:14px">
        <div class="prob-row">
          <div class="prob-label">
            <span style="color:var(--accent-green)">Engaged</span>
            <span class="pct">${engPct}%</span>
          </div>
          <div class="prob-bar-wrap">
            <div class="prob-bar-fill bar-engaged" style="width:${engPct}%"></div>
          </div>
        </div>
        <div class="prob-row">
          <div class="prob-label">
            <span style="color:var(--accent-red)">Disengaged</span>
            <span class="pct">${disPct}%</span>
          </div>
          <div class="prob-bar-wrap">
            <div class="prob-bar-fill bar-disengaged" style="width:${disPct}%"></div>
          </div>
        </div>
      </div>

      <!-- Temporal 1st vs 2nd Half Strip -->
      <div class="card-halftime-strip">
        <span class="chs-label">1st → 2nd Half:</span>
        <span class="chs-val">${h1} → ${h2}</span>
        <span class="chs-trend">${s.student_trend === "improving" ? "📈 Improving" : (s.student_trend === "dropping" ? "📉 Dropping" : "➡️ Steady")}</span>
      </div>
    `;
    studentsGrid.appendChild(card);
  });
}

// Charts
function renderCharts(charts) {
  chartsGrid.innerHTML = "";
  const titles = {
    halftime_chart: "Classroom Half-Time Temporal Progression (1st Half vs 2nd Half)",
    timeline_chart: "Classroom Engagement Timeline (Midpoint Half 1 / Half 2 Split)",
    student_chart:  "Per-Student 2×1 Probability Matrix Breakdown",
    pie_chart:      "3-Way Engagement Class Distribution (Engaged / Partially / Disengaged)",
  };

  // Ordered list of chart keys
  const order = ["halftime_chart", "timeline_chart", "student_chart", "pie_chart"];
  order.forEach(key => {
    const url = charts[key];
    if (!url) return;
    const div = document.createElement("div");
    div.className = "chart-card";
    div.innerHTML = `
      <div class="chart-title">
        <span>📊 ${titles[key] || key}</span>
        <a class="chart-dl-link" href="${url}" download title="Download PNG Chart">Download PNG ↓</a>
      </div>
      <img class="chart-img" src="${url}" alt="${titles[key] || key}" loading="lazy"/>
    `;
    chartsGrid.appendChild(div);
  });

  // Any remaining charts
  Object.entries(charts).forEach(([key, url]) => {
    if (!order.includes(key)) {
      const div = document.createElement("div");
      div.className = "chart-card";
      div.innerHTML = `
        <div class="chart-title"><span>📊 ${titles[key] || key}</span></div>
        <img class="chart-img" src="${url}" alt="${titles[key] || key}" loading="lazy"/>`;
      chartsGrid.appendChild(div);
    }
  });
}

// Downloads
function renderDownloads(result) {
  downloadsList.innerHTML = "";
  const items = [
    {
      icon: "🎬", name: "Annotated Classroom Video",
      desc: "H.264 MP4 with YOLOv8 face bounding boxes, track IDs, and E:XX% D:XX% probability HUD",
      url: result.video_url, label: "Download Video (.mp4)"
    },
    {
      icon: "📊", name: "Detailed Engagement Log CSV",
      desc: "Frame-by-frame records with student IDs, engaged/disengaged probabilities, and bounding boxes",
      url: result.csv_url, label: "Download CSV (.csv)"
    },
    {
      icon: "📄", name: "Classroom Analysis Report",
      desc: "Complete pedagogical and statistical report with classroom summary and student diagnostics",
      url: result.report_url, label: "Download Report (.txt)"
    },
  ];

  if (result.charts?.halftime_chart) {
    items.push({ icon: "⏱️", name: "Half-Time Comparison Chart", desc: "1st Half vs 2nd Half Classroom Analysis Chart", url: result.charts.halftime_chart, label: "Download PNG" });
  }
  if (result.charts?.timeline_chart) {
    items.push({ icon: "📈", name: "Engagement Timeline Chart", desc: "Continuous classroom engagement with midpoint split", url: result.charts.timeline_chart, label: "Download PNG" });
  }
  if (result.charts?.student_chart) {
    items.push({ icon: "👥", name: "Per-Student Probability Chart", desc: "Dual probability matrix bar chart for all students", url: result.charts.student_chart, label: "Download PNG" });
  }
  if (result.charts?.pie_chart) {
    items.push({ icon: "🥧", name: "3-Class Distribution Donut Chart", desc: "Proportional split of Engaged, Partially Engaged, and Disengaged", url: result.charts.pie_chart, label: "Download PNG" });
  }

  items.forEach(item => {
    const div = document.createElement("div");
    div.className = "dl-item";
    div.innerHTML = `
      <div class="dl-info">
        <div class="dl-icon">${item.icon}</div>
        <div>
          <div class="dl-name">${item.name}</div>
          <div class="dl-desc">${item.desc}</div>
        </div>
      </div>
      <a class="dl-btn" href="${item.url}" download>${item.label} ↓</a>`;
    downloadsList.appendChild(div);
  });
}

// ─────────────────────────────────────────────
// Tab Switching
// ─────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.tab === name);
    b.setAttribute("aria-selected", b.dataset.tab === name);
  });
  document.querySelectorAll(".tab-content").forEach(c => {
    c.classList.toggle("active", c.id === `tab-${name}-content`);
  });
}

// ─────────────────────────────────────────────
// New analysis
// ─────────────────────────────────────────────
newAnalysisBtn.addEventListener("click", resetToUpload);
function resetToUpload() {
  selectedFile = null;
  isDemoMode = false;

  // Reset webcam state
  stopWebcamStream();
  recordedChunks = [];
  recordedBlob = null;
  recordedVideoFile = null;
  if (recordedPlayback) {
    recordedPlayback.pause();
    recordedPlayback.src = "";
    recordedPlayback.classList.add("hidden");
  }
  if (webcamVideo) webcamVideo.classList.remove("hidden");
  if (recordBtn) {
    recordBtn.classList.remove("hidden", "is-recording");
    recordBtn.disabled = true;
    recordBtnText.textContent = "Start Recording";
  }
  if (retakeBtn) retakeBtn.classList.add("hidden");
  if (useRecordedBtn) useRecordedBtn.classList.add("hidden");
  if (webcamTimer) webcamTimer.classList.add("hidden");

  demoBanner.classList.add("hidden");
  dropIcon.textContent  = "🎬";
  dropTitle.textContent = "Drop your classroom video here";
  dropHint.textContent  = "or click to browse files";
  dropZone.classList.remove("has-file", "drag-over");
  analyseBtn.disabled   = true;
  btnText.textContent   = "Select a video to begin";
  videoInput.value      = "";
  activeStageName       = null;

  // Reset pipeline stages
  Object.keys(PIPELINE_STAGES).forEach(n => {
    const cfg = PIPELINE_STAGES[n];
    const el  = document.getElementById(cfg.el);
    if (el) el.classList.remove("active","done");
    const s = el?.querySelector(".ps-status");
    if (s) s.textContent = "waiting";
    if (cfg.conn) {
      const c = document.getElementById(cfg.conn);
      if (c) { c.classList.remove("active","done"); }
    }
  });
  progressBar.style.width = "0%";
  progressPct.textContent = "0%";
  progressStage.textContent = "Queued…";

  uploadSection.classList.remove("hidden");
  processingCard.classList.add("hidden");
  resultsSection.classList.add("hidden");
  uploadSection.scrollIntoView({ behavior: "smooth" });
}

// ─────────────────────────────────────────────
// Error Banner
// ─────────────────────────────────────────────
function showError(msg) {
  errorText.textContent = msg;
  errorBanner.classList.remove("hidden");
}
function hideError() { errorBanner.classList.add("hidden"); }
errorClose.addEventListener("click", hideError);

// ─────────────────────────────────────────────
// Particle System (canvas animation during processing)
// ─────────────────────────────────────────────
function startParticles() {
  const canvas = document.getElementById("particleCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const resize = () => {
    canvas.width  = processingCard.offsetWidth;
    canvas.height = processingCard.offsetHeight;
  };
  resize();
  window.addEventListener("resize", resize);

  const particles = Array.from({ length: 55 }, () => createParticle(canvas.width, canvas.height));

  function createParticle(w, h) {
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.6,
      r: Math.random() * 2.5 + 0.5,
      alpha: Math.random() * 0.6 + 0.15,
      color: ["#4f8ef7","#a78bfa","#22d3ee","#2ecc71"][Math.floor(Math.random() * 4)],
    };
  }

  function draw() {
    if (!particleAnim) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.alpha;
      ctx.fill();
    });

    // Draw connecting lines
    ctx.globalAlpha = 0.07;
    ctx.strokeStyle = "#4f8ef7";
    ctx.lineWidth = 0.7;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 90) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;
    particleAnim = requestAnimationFrame(draw);
  }

  particleAnim = requestAnimationFrame(draw);
}

function stopParticles() {
  if (particleAnim) {
    cancelAnimationFrame(particleAnim);
    particleAnim = null;
  }
  const canvas = document.getElementById("particleCanvas");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

// ─────────────────────────────────────────────
// Hero pipeline step hover glow (subtle)
// ─────────────────────────────────────────────
document.querySelectorAll(".pipe-step").forEach((el, i) => {
  el.style.animationDelay = `${i * 0.1}s`;
});
