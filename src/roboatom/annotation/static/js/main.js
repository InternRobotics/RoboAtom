import {
  getMeta,
  getCollabConfig,
  selectAnnotator,
  getVideo,
  switchInput,
  updateAnnotation,
  updateDiscard,
  saveAll,
  sendAuditEvent,
} from "./api.js";
import {
  state,
  cloneDeep,
  snapshotHash,
  setMeta,
  clearMetaState,
  setCollabConfig,
  setVideo,
  applyWorkingCopyFromDraft,
  createMutationToken,
  finalizeMutation,
  undoMutation,
  redoMutation,
  markDraft,
  markCommitted,
  markSaved,
  currentVideo,
  commitMetaCounts,
  setAnnotationStatus,
  setAnnotationStatuses,
} from "./state.js";
import {
  clamp,
  frameToTime,
  frameToSeekTime,
  timeToFrame,
  findSegmentIndexByFrame,
  splitSegmentAtFrame,
  deleteSegmentAtIndex,
  setSegmentStart,
  setSegmentEnd,
  moveBoundary,
  validateSegmentTopology,
  formatTime,
  formatRange,
  formatDuration,
  createTimelineWindow,
  clampTimelineWindow,
  ensureFrameInTimelineWindow,
  getTimelineWindowEnd,
  frameToTimelineRatio,
  ratioToTimelineFrame,
} from "./timeline.js";
import {
  getSegmentCompleteness,
  getVideoStats,
  sanitizeTasks,
  filterSkillOptions,
  ensureSkillOption,
  filterActionTextOptions,
  ensureActionTextOption,
} from "./editor.js";
import { buildDraftKey, saveDraft, loadDraft, clearDraft } from "./draft.js";
import { registerShortcuts } from "./shortcuts.js";

const ui = {
  prevVideoBtn: document.getElementById("prevVideoBtn"),
  nextVideoBtn: document.getElementById("nextVideoBtn"),
  videoSelect: document.getElementById("videoSelect"),
  videoJumpInput: document.getElementById("videoJumpInput"),
  videoPosition: document.getElementById("videoPosition"),
  statusDraft: document.getElementById("statusDraft"),
  statusCommitted: document.getElementById("statusCommitted"),
  statusSaved: document.getElementById("statusSaved"),
  statusHint: document.getElementById("statusHint"),
  annotatorCluster: document.getElementById("annotatorCluster"),
  currentAnnotatorName: document.getElementById("currentAnnotatorName"),
  switchAnnotatorBtn: document.getElementById("switchAnnotatorBtn"),
  advancedToggleBtn: document.getElementById("advancedToggleBtn"),
  advancedBar: document.getElementById("advancedBar"),
  inputJsonField: document.getElementById("inputJsonField"),
  loadJsonBtn: document.getElementById("loadJsonBtn"),
  undoBtn: document.getElementById("undoBtn"),
  redoBtn: document.getElementById("redoBtn"),
  applyBtn: document.getElementById("applyBtn"),
  saveBtn: document.getElementById("saveBtn"),

  videoWrap: document.getElementById("videoWrap"),
  videoGrid: document.getElementById("videoGrid"),
  activeViewBadge: document.getElementById("activeViewBadge"),
  segmentBadge: document.getElementById("segmentBadge"),
  activeSegmentRange: document.getElementById("activeSegmentRange"),
  playbackTime: document.getElementById("playbackTime"),
  videoError: document.getElementById("videoError"),

  playPauseBtn: document.getElementById("playPauseBtn"),
  playIcon: document.getElementById("playIcon"),
  pauseIcon: document.getElementById("pauseIcon"),
  jumpBack10Btn: document.getElementById("jumpBack10Btn"),
  prevFrameBtn: document.getElementById("prevFrameBtn"),
  nextFrameBtn: document.getElementById("nextFrameBtn"),
  jumpForward10Btn: document.getElementById("jumpForward10Btn"),
  autoStopToggle: document.getElementById("autoStopToggle"),
  loopSegmentToggle: document.getElementById("loopSegmentToggle"),
  playbackRateSelect: document.getElementById("playbackRateSelect"),
  timelineZoomSelect: document.getElementById("timelineZoomSelect"),
  playSegmentBtn: document.getElementById("playSegmentBtn"),

  sliderWrap: document.getElementById("sliderWrap"),
  frameSlider: document.getElementById("frameSlider"),
  chapterBar: document.getElementById("chapterBar"),
  boundaryMarkers: document.getElementById("boundaryMarkers"),
  boundaryHandles: document.getElementById("boundaryHandles"),
  currentFrameNeedle: document.getElementById("currentFrameNeedle"),
  frameLabel: document.getElementById("frameLabel"),
  selectionHint: document.getElementById("selectionHint"),

  timelineSummary: document.getElementById("timelineSummary"),
  episodeValue: document.getElementById("episodeValue"),
  episodeValueInline: document.getElementById("episodeValueInline"),
  framesValueInline: document.getElementById("framesValueInline"),
  pathValue: document.getElementById("pathValue"),
  framesValue: document.getElementById("framesValue"),
  fpsValue: document.getElementById("fpsValue"),
  segmentCountValue: document.getElementById("segmentCountValue"),
  incompleteCountValue: document.getElementById("incompleteCountValue"),
  avgSegmentLenValue: document.getElementById("avgSegmentLenValue"),
  tasksInput: document.getElementById("tasksInput"),
  itemError: document.getElementById("itemError"),

  segmentRows: document.getElementById("segmentRows"),
  prevIncompleteBtn: document.getElementById("prevIncompleteBtn"),
  nextIncompleteBtn: document.getElementById("nextIncompleteBtn"),
  addSegmentBtn: document.getElementById("addSegmentBtn"),
  setStartBtn: document.getElementById("setStartBtn"),
  setEndBtn: document.getElementById("setEndBtn"),
  jumpSegmentStartBtn: document.getElementById("jumpSegmentStartBtn"),
  jumpSegmentEndBtn: document.getElementById("jumpSegmentEndBtn"),

  activeEditorEmpty: document.getElementById("activeEditorEmpty"),
  activeEditorBody: document.getElementById("activeEditorBody"),
  activeSegmentMeta: document.getElementById("activeSegmentMeta"),
  activeStartValue: document.getElementById("activeStartValue"),
  activeEndValue: document.getElementById("activeEndValue"),
  activeDurationValue: document.getElementById("activeDurationValue"),
  activeCurrentFrameValue: document.getElementById("activeCurrentFrameValue"),
  modelReviewPanel: document.getElementById("modelReviewPanel"),
  modelReviewVerdict: document.getElementById("modelReviewVerdict"),
  modelReviewCount: document.getElementById("modelReviewCount"),
  modelReviewIssues: document.getElementById("modelReviewIssues"),
  actionTextFilterInput: document.getElementById("actionTextFilterInput"),
  activeActionSelect: document.getElementById("activeActionSelect"),
  activeActionInput: document.getElementById("activeActionInput"),
  skillFilterInput: document.getElementById("skillFilterInput"),
  activeSkillSelect: document.getElementById("activeSkillSelect"),
  skillConfigHint: document.getElementById("skillConfigHint"),

  inputPath: document.getElementById("inputPath"),
  outputPath: document.getElementById("outputPath"),
  toastContainer: document.getElementById("toastContainer"),
  infoPanel: document.getElementById("infoPanel"),
  infoPanelToggle: document.getElementById("infoPanelToggle"),
  infoPanelBody: document.getElementById("infoPanelBody"),
  mainLayout: document.getElementById("mainLayout"),
  panelResizeHandle: document.getElementById("panelResizeHandle"),
  annotationPanel: document.getElementById("annotationPanel"),
  annotatorModal: document.getElementById("annotatorModal"),
  annotatorModalMessage: document.getElementById("annotatorModalMessage"),
  annotatorOptionList: document.getElementById("annotatorOptionList"),
  annotatorModalCloseBtn: document.getElementById("annotatorModalCloseBtn"),
  itemStatusCard: document.getElementById("itemStatusCard"),
  itemStatusIcon: document.getElementById("itemStatusIcon"),
  itemStatusBadge: document.getElementById("itemStatusBadge"),
  itemStatusDescription: document.getElementById("itemStatusDescription"),
  itemStatusNote: document.getElementById("itemStatusNote"),
  itemStatusNormalActions: document.getElementById("itemStatusNormalActions"),
  itemStatusDiscardedActions: document.getElementById("itemStatusDiscardedActions"),
  discardItemBtn: document.getElementById("discardItemBtn"),
  restoreItemBtn: document.getElementById("restoreItemBtn"),
};

const local = {
  isScrubbing: false,
  deleteConfirmIdx: null,
  deleteConfirmTimer: null,
  dragBoundary: null,
  draftSaveTimer: null,
  actionEditToken: null,
  taskEditToken: null,
  destroyShortcuts: null,
  viewPlayers: new Map(),
  viewPlayback: new Map(),
  driverViewKey: "",
  activeViewKey: "",
  hasPlayableViews: false,
  syncingFromPrimary: false,
  autoStopSegment: true,
  loopSegment: false,
  segmentPlaybackActive: false,
  playheadFrame: 0,
  playbackStableFrame: 0,
  settledFrame: 0,
  isPlaybackStable: false,
  isPlaybackRefreshing: false,
  isFrameSettled: false,
  isSyncSettling: false,
  syncRequestId: 0,
  syncTargetFrame: null,
  pendingViews: new Set(),
  syncPollTimer: null,
  syncTimeoutTimer: null,
  playbackRefreshRequestId: 0,
  playbackRefreshTargetFrame: null,
  playbackRefreshPendingViews: new Set(),
  playbackRefreshPollTimer: null,
  playbackRefreshTimeoutTimer: null,
  playbackBoundaryWatchHandle: null,
  playbackBoundaryWatchMode: "",
  pendingAnnotationAction: null,
  playbackSyncTimer: null,
  playbackRate: 1,
  annotationPanelWidth: 420,
  metaRequestController: null,
  videoRequestController: null,
  switchInputController: null,
  isSelectingAnnotator: false,
  isSaving: false,
  isUpdatingDiscard: false,
  auditActiveIndex: -1,
  auditOpenedAt: null,
  metaRequestId: 0,
  switchInputRequestId: 0,
  dragRenderRaf: null,
  // Prefetch cache: Map<index, { payload, error, loading, controllers }>
  prefetchCache: new Map(),
  prefetchTimer: null,
};

const playerSeekState = new WeakMap();
const SHORTCUT_FLASH_MS = 140;
const PANEL_WIDTH_STORAGE_KEY = "annotation_ui.annotationPanelWidth";
const PANEL_WIDTH_DEFAULT = 420;
const PANEL_WIDTH_MIN = 360;
const PANEL_WIDTH_MAX = 720;
const TIMELINE_WINDOW_PRESETS = new Set(["global", "60s", "10s"]);
const PLAYBACK_SYNC_INTERVAL_MS = 80;
const PLAYBACK_SYNC_DRIFT_FRAMES = 1;
const PLAYBACK_STOP_TOLERANCE_FRAMES = 0.5;
const PLAYBACK_REFRESH_POLL_MS = 30;
const PLAYBACK_REFRESH_TIMEOUT_MS = 1200;
const SETTLE_POLL_MS = 30;
const SETTLE_TIMEOUT_MS = 2000;
function isCurrentDiscarded() {
  return Boolean(currentVideo()?.discarded);
}

function readStoredAnnotationPanelWidth() {
  try {
    const raw = window.localStorage.getItem(PANEL_WIDTH_STORAGE_KEY);
    const width = Number.parseInt(raw || "", 10);
    if (!Number.isFinite(width)) return PANEL_WIDTH_DEFAULT;
    return clamp(width, PANEL_WIDTH_MIN, PANEL_WIDTH_MAX);
  } catch (_) {
    return PANEL_WIDTH_DEFAULT;
  }
}

function applyAnnotationPanelWidth(width) {
  const safeWidth = clamp(Number(width) || PANEL_WIDTH_DEFAULT, PANEL_WIDTH_MIN, PANEL_WIDTH_MAX);
  local.annotationPanelWidth = safeWidth;
  document.documentElement.style.setProperty("--annotation-panel-width", `${safeWidth}px`);
  try {
    window.localStorage.setItem(PANEL_WIDTH_STORAGE_KEY, String(safeWidth));
  } catch (_) {
    // ignore storage errors in restricted browsers
  }
}

function syncAdvancedToggle() {
  if (!ui.advancedToggleBtn || !ui.advancedBar) return;
  const expanded = !ui.advancedBar.classList.contains("hidden");
  ui.advancedToggleBtn.setAttribute("aria-expanded", String(expanded));
}

function setMultiUserUiMode(enabled) {
  if (ui.annotatorCluster) ui.annotatorCluster.classList.toggle("hidden", !enabled);
  if (ui.advancedToggleBtn) ui.advancedToggleBtn.classList.toggle("hidden", enabled);
  if (enabled && ui.advancedBar) {
    ui.advancedBar.classList.add("hidden");
  }
  syncAdvancedToggle();
}

function renderAnnotatorState() {
  const enabled = Boolean(state.collab.enabled);
  setMultiUserUiMode(enabled);
  if (!enabled) return;

  if (ui.currentAnnotatorName) {
    ui.currentAnnotatorName.textContent = state.collab.annotatorName || "未选择";
  }
  if (ui.annotatorModalCloseBtn) {
    ui.annotatorModalCloseBtn.classList.toggle("hidden", !state.collab.annotatorId);
  }
}

function renderAnnotatorOptions() {
  if (!ui.annotatorOptionList) return;
  ui.annotatorOptionList.innerHTML = "";
  if (!state.collab.annotators.length) {
    ui.annotatorOptionList.innerHTML = '<div class="annotator-option-empty">当前没有可用的标注员配置</div>';
    return;
  }

  state.collab.annotators.forEach(item => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "annotator-option";
    if (item.id === state.collab.annotatorId) {
      button.classList.add("active");
    }
    button.disabled = local.isSelectingAnnotator || local.isSaving;
    button.dataset.annotatorId = item.id;
    button.innerHTML = `
      <span class="annotator-option-name"></span>
      <span class="annotator-option-meta"></span>
      <span class="annotator-option-id"></span>
    `;
    const nameEl = button.querySelector(".annotator-option-name");
    const metaEl = button.querySelector(".annotator-option-meta");
    const idEl = button.querySelector(".annotator-option-id");
    if (nameEl) nameEl.textContent = item.name || item.id;
    if (metaEl) {
      metaEl.textContent = item.role === "reviewer"
        ? `${item.count} 条任务 · 全量复核`
        : `${item.count} 条任务 · 从第 ${item.start_index + 1} 条开始`;
    }
    if (idEl) idEl.textContent = item.id;
    button.addEventListener("click", async () => {
      await handleAnnotatorSelection(item.id);
    });
    ui.annotatorOptionList.appendChild(button);
  });
}

function showAnnotatorModal(message = "请选择你的标注身份后开始标注。") {
  if (!ui.annotatorModal) return;
  if (ui.annotatorModalMessage) ui.annotatorModalMessage.textContent = message;
  renderAnnotatorOptions();
  ui.annotatorModal.classList.remove("hidden");
  ui.annotatorModal.setAttribute("aria-hidden", "false");
}

function hideAnnotatorModal() {
  if (!ui.annotatorModal) return;
  ui.annotatorModal.classList.add("hidden");
  ui.annotatorModal.setAttribute("aria-hidden", "true");
}

function isAnnotatorSelectionRequired(error) {
  return Boolean(error?.status === 409 && error?.payload?.requires_annotator_selection);
}

function resetLoadedDataset(message = "无视频") {
  clearPrefetchCache();
  clearMetaState();
  setPathInfo("", "");
  renderMetaSelect();
  clearCurrentView(message);
}

async function refreshCollabConfig({ silent = false } = {}) {
  try {
    const payload = await getCollabConfig();
    setCollabConfig(payload);
    renderAnnotatorState();
    return payload;
  } catch (error) {
    if (!silent) {
      showToast(`加载协作配置失败: ${error.message}`, "error");
    }
    return null;
  }
}

async function handleAnnotatorSelectionRequired(message = "请选择标注员后继续。") {
  await refreshCollabConfig({ silent: true });
  renderAnnotatorState();
  resetLoadedDataset(message);
  showAnnotatorModal(message);
}

function syncVideoJumpInput() {
  if (!ui.videoJumpInput) return;
  const hasMeta = state.meta.length > 0 && state.currentIndex >= 0;
  ui.videoJumpInput.disabled = !hasMeta || local.isSaving;
  ui.videoJumpInput.placeholder = hasMeta ? `1-${state.meta.length}` : "跳到 #";
  if (!hasMeta) {
    ui.videoJumpInput.value = "";
    return;
  }
  if (document.activeElement === ui.videoJumpInput) return;
  ui.videoJumpInput.value = String(state.currentIndex + 1);
}

function updateWorkflowControls() {
  const hasMeta = state.meta.length > 0 && state.currentIndex >= 0;
  const hasVideo = Boolean(currentVideo());
  const busy = local.isSaving || state.isLoadingVideo || local.isSelectingAnnotator;
  if (ui.prevVideoBtn) ui.prevVideoBtn.disabled = busy || !hasMeta;
  if (ui.nextVideoBtn) ui.nextVideoBtn.disabled = busy || !hasMeta;
  if (ui.videoSelect) ui.videoSelect.disabled = busy || !state.meta.length;
  if (ui.saveBtn) ui.saveBtn.disabled = busy || !hasVideo;
  if (ui.applyBtn) ui.applyBtn.disabled = busy || !hasVideo || isCurrentDiscarded();
  if (ui.switchAnnotatorBtn) {
    ui.switchAnnotatorBtn.disabled = busy || !state.collab.enabled;
  }
  syncVideoJumpInput();
}

function isAbortError(error) {
  return error?.name === "AbortError";
}

function formatStatusTimestamp(value) {
  if (!value) return "";
  try {
    return new Date(value).toLocaleTimeString("zh-CN", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch (_) {
    return "";
  }
}

function getTimelinePreset() {
  const preset = String(ui.timelineZoomSelect?.value || "global");
  return TIMELINE_WINDOW_PRESETS.has(preset) ? preset : "global";
}

function getTimelineWindow(video = currentVideo()) {
  if (!video) return { preset: "global", startFrame: 0, span: 0 };
  if (!state.timelineWindow) {
    state.timelineWindow = createTimelineWindow(video, getTimelinePreset(), state.currentFrame);
  }
  return clampTimelineWindow(video, state.timelineWindow);
}

function syncTimelineSliderWindow(video = currentVideo()) {
  if (!video) {
    ui.frameSlider.min = "0";
    ui.frameSlider.max = "0";
    ui.frameSlider.value = "0";
    if (ui.timelineZoomSelect) ui.timelineZoomSelect.value = getTimelinePreset();
    return;
  }

  const timelineWindow = getTimelineWindow(video);
  const maxFrame = getTimelineWindowEnd(video, timelineWindow);
  ui.frameSlider.min = String(timelineWindow.startFrame);
  ui.frameSlider.max = String(maxFrame);
  ui.frameSlider.value = String(clamp(state.currentFrame, timelineWindow.startFrame, maxFrame));
  if (ui.timelineZoomSelect) {
    ui.timelineZoomSelect.value = String(timelineWindow.preset || "global");
  }
}

function setTimelineWindow(preset, centerFrame = state.currentFrame) {
  const video = currentVideo();
  if (!video) return;
  const safePreset = TIMELINE_WINDOW_PRESETS.has(String(preset)) ? String(preset) : "global";
  state.timelineWindow = createTimelineWindow(video, safePreset, centerFrame);
  syncTimelineSliderWindow(video);
}

function ensureVisibleTimelineFrame(frame) {
  const video = currentVideo();
  if (!video) return;
  state.timelineWindow = ensureFrameInTimelineWindow(video, getTimelineWindow(video), frame);
  syncTimelineSliderWindow(video);
}

function setInfoPanelVisible(visible) {
  if (!ui.infoPanel || !ui.infoPanelToggle) return;
  ui.infoPanel.classList.toggle("hidden", !visible);
  ui.infoPanelToggle.setAttribute("aria-expanded", String(visible));
}

function getSelectedSegment(video = currentVideo()) {
  if (!video || !Array.isArray(video.action_config)) return null;
  return video.action_config[state.selectedSegIdx] || null;
}

function getSegmentLastFrame(segment) {
  if (!segment) return 0;
  return Math.max(Number(segment.start_frame) || 0, (Number(segment.end_frame) || 0) - 1);
}

function getStopAnchorFrame(segment) {
  return getSegmentLastFrame(segment);
}

function shouldConstrainPlaybackToSelectedSegment() {
  return Boolean(local.segmentPlaybackActive || local.autoStopSegment || local.loopSegment);
}

function syncPlaybackOptionControls() {
  if (ui.autoStopToggle) ui.autoStopToggle.checked = local.autoStopSegment;
  if (ui.loopSegmentToggle) ui.loopSegmentToggle.checked = local.loopSegment;
  if (ui.playbackRateSelect) ui.playbackRateSelect.value = String(local.playbackRate);
}

function applyPlaybackRate(rate) {
  const nextRate = [0.5, 1, 2].includes(Number(rate)) ? Number(rate) : 1;
  local.playbackRate = nextRate;
  syncPlaybackOptionControls();
  for (const player of local.viewPlayers.values()) {
    if (!player) continue;
    player.playbackRate = nextRate;
  }
}

function focusSegmentField(index, fieldName) {
  const video = currentVideo();
  if (!video || index < 0 || index >= video.action_config.length) return false;
  const segment = video.action_config[index];
  state.selectedSegIdx = index;
  renderAfterMutation(false);
  setCurrentFrame(segment.start_frame, true);
  const target = fieldName === "skill" ? ui.activeSkillSelect : ui.activeActionInput;
  if (!target) return false;
  window.setTimeout(() => {
    target.focus();
    if (target === ui.activeActionInput) {
      ui.activeActionInput.setSelectionRange(
        ui.activeActionInput.value.length,
        ui.activeActionInput.value.length,
      );
    }
  }, 40);
  return true;
}

function findIncompleteSegmentIndex(direction = 1) {
  const video = currentVideo();
  if (!video || !Array.isArray(video.action_config) || !video.action_config.length) return -1;
  const total = video.action_config.length;
  const start = clamp(state.selectedSegIdx, 0, total - 1);
  for (let step = 1; step <= total; step += 1) {
    const offset = direction > 0 ? step : -step;
    const index = (start + offset + total) % total;
    const completeness = getSegmentCompleteness(video.action_config[index]);
    if (!completeness.complete) return index;
  }
  return -1;
}

function updateIncompleteNavButtons() {
  const enabled = currentVideo() && currentVideo().action_config?.some(segment => !getSegmentCompleteness(segment).complete);
  if (ui.prevIncompleteBtn) ui.prevIncompleteBtn.disabled = !enabled;
  if (ui.nextIncompleteBtn) ui.nextIncompleteBtn.disabled = !enabled;
}

function goToIncompleteSegment(direction = 1) {
  const index = findIncompleteSegmentIndex(direction);
  if (index < 0) {
    showToast("当前视频没有未完成片段", "info");
    return;
  }
  const video = currentVideo();
  const completeness = getSegmentCompleteness(video.action_config[index]);
  const targetField = completeness.missingAction ? "action" : "skill";
  focusSegmentField(index, targetField);
}

function focusFirstValidationTarget(error) {
  const fieldErrors = Array.isArray(error?.fieldErrors)
    ? error.fieldErrors
    : Array.isArray(error?.payload?.field_errors)
      ? error.payload.field_errors
      : [];

  for (const item of fieldErrors) {
    const field = String(item?.field || "");
    if (field === "tasks") {
      ui.tasksInput.focus();
      return true;
    }
    const matched = field.match(/^action_config\[(\d+)\]\.(action_text|skill|start_frame|end_frame)$/);
    if (!matched) continue;
    const index = Number.parseInt(matched[1], 10);
    const fieldName = matched[2] === "skill" ? "skill" : "action";
    return focusSegmentField(index, fieldName);
  }

  return false;
}

function showToast(message, type = "info", duration = 3000) {
  const icons = { success: "✓", error: "✕", info: "ℹ" };
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type] || "ℹ"}</span><span class="toast-msg"></span>`;
  const msgEl = toast.querySelector(".toast-msg");
  if (msgEl) msgEl.textContent = message;
  ui.toastContainer.appendChild(toast);

  const remove = () => {
    toast.classList.add("toast-out");
    toast.addEventListener("animationend", () => toast.remove(), { once: true });
  };

  const timer = window.setTimeout(remove, duration);
  toast.addEventListener("click", () => {
    window.clearTimeout(timer);
    remove();
  });
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function isTextEditingTarget(target) {
  if (!(target instanceof HTMLElement)) return false;
  // SELECT and numeric INPUT are handled by their own target checks below.
  if (target.tagName === "SELECT") return false;
  if (isNumericInputTarget(target)) return false;
  return ["INPUT", "TEXTAREA"].includes(target.tagName) || target.isContentEditable;
}

function isSelectTarget(target) {
  return target instanceof HTMLElement && target.tagName === "SELECT";
}

function isNumericInputTarget(target) {
  if (!(target instanceof HTMLElement) || target.tagName !== "INPUT") return false;
  const inputMode = (target.getAttribute("inputmode") || "").toLowerCase();
  const inputType = (target.type || "text").toLowerCase();
  return inputMode === "numeric" || inputType === "number";
}

function getDraftKey(video) {
  return buildDraftKey(state.inputJson, video || currentVideo(), state.collab.annotatorId);
}

function normalizeUrl(rawUrl) {
  if (!rawUrl) return "";
  try {
    return new URL(rawUrl, window.location.origin).toString();
  } catch (_) {
    return String(rawUrl);
  }
}

function makeViewStreamUrl(index, viewKey) {
  const encoded = encodeURIComponent(String(viewKey || ""));
  return `/api/stream/${index}/${encoded}`;
}

function getViewPlayer(viewKey) {
  return local.viewPlayers.get(viewKey) || null;
}

function getDriverPlayer() {
  return getViewPlayer(local.driverViewKey);
}

function listPlayableViewKeys() {
  const out = [];
  for (const [viewKey, cfg] of local.viewPlayback.entries()) {
    if (cfg && !cfg.unavailable) out.push(viewKey);
  }
  return out;
}

function isDriverPlaying() {
  const driver = getDriverPlayer();
  return Boolean(driver && !driver.paused && !driver.ended);
}

function setPlayheadFrameValue(frame) {
  const video = currentVideo();
  if (!video) {
    local.playheadFrame = 0;
    state.currentFrame = 0;
    return 0;
  }
  const next = clamp(frame, 0, Math.max(video.frames - 1, 0));
  local.playheadFrame = next;
  state.currentFrame = next;
  return next;
}

function markFrameSettled(frame) {
  local.settledFrame = frame;
  local.isFrameSettled = true;
}

function markPlaybackStable(frame) {
  local.playbackStableFrame = frame;
  local.isPlaybackStable = true;
}

function clearFrameSettled() {
  local.isFrameSettled = false;
}

function clearPlaybackStable() {
  local.isPlaybackStable = false;
}

function clearSyncTimers() {
  window.clearInterval(local.syncPollTimer);
  window.clearTimeout(local.syncTimeoutTimer);
  local.syncPollTimer = null;
  local.syncTimeoutTimer = null;
}

function clearPlaybackRefreshTimers() {
  window.clearInterval(local.playbackRefreshPollTimer);
  window.clearTimeout(local.playbackRefreshTimeoutTimer);
  local.playbackRefreshPollTimer = null;
  local.playbackRefreshTimeoutTimer = null;
}

function clearPendingAnnotationAction() {
  local.pendingAnnotationAction = null;
}

function getPlayerSeekMeta(player) {
  let meta = playerSeekState.get(player);
  if (!meta) {
    meta = {
      issued: 0,
      completed: 0,
      pending: 0,
    };
    playerSeekState.set(player, meta);
  }
  return meta;
}

function markPlayerSeekIssued(player) {
  if (!player) return 0;
  const meta = getPlayerSeekMeta(player);
  meta.issued += 1;
  meta.pending = meta.issued;
  return meta.pending;
}

function markPlayerSeekCompleted(player) {
  if (!player) return;
  const meta = getPlayerSeekMeta(player);
  const token = meta.pending || meta.issued;
  if (!token) return;
  window.requestAnimationFrame(() => {
    const nextMeta = getPlayerSeekMeta(player);
    nextMeta.completed = Math.max(nextMeta.completed, token);
    if (nextMeta.pending === token) {
      nextMeta.pending = 0;
    }
  });
}

function hasPlayerCompletedLatestSeek(player) {
  if (!player) return false;
  const meta = getPlayerSeekMeta(player);
  return meta.completed >= meta.issued && meta.pending === 0;
}

function cancelFrameSettle({ keepSettled = true, clearPendingAction = false } = {}) {
  local.syncRequestId += 1;
  clearSyncTimers();
  local.isSyncSettling = false;
  local.syncTargetFrame = null;
  local.pendingViews = new Set();
  if (!keepSettled) {
    clearFrameSettled();
  }
  if (clearPendingAction) {
    clearPendingAnnotationAction();
  }
}

function cancelPlaybackRefresh({ keepStable = true, clearPendingAction = false } = {}) {
  local.playbackRefreshRequestId += 1;
  clearPlaybackRefreshTimers();
  local.isPlaybackRefreshing = false;
  local.playbackRefreshTargetFrame = null;
  local.playbackRefreshPendingViews = new Set();
  if (!keepStable) {
    clearPlaybackStable();
  }
  if (clearPendingAction) {
    clearPendingAnnotationAction();
  }
}

function isPlayerSettledAtFrame(player, targetFrame) {
  const video = currentVideo();
  if (!video || !player || player.readyState < 2 || player.seeking) return false;
  if (!hasPlayerCompletedLatestSeek(player)) return false;
  return timeToFrame(video, player.currentTime || 0) === targetFrame;
}

function getFrameDistanceFromTarget(timeSec, targetFrame) {
  const video = currentVideo();
  if (!video || !Number.isFinite(video.fps) || video.fps <= 0) return Infinity;
  const targetTime = frameToSeekTime(video, targetFrame);
  return Math.abs(((timeSec || 0) - targetTime) * video.fps);
}

function isPlayerWithinFrameTolerance(player, targetFrame, toleranceFrames = PLAYBACK_STOP_TOLERANCE_FRAMES) {
  if (!player || player.readyState < 2 || player.seeking) return false;
  if (!hasPlayerCompletedLatestSeek(player)) return false;
  return getFrameDistanceFromTarget(player.currentTime || 0, targetFrame) <= toleranceFrames;
}

function collectPendingViewKeys(targetFrame) {
  const pending = new Set();
  for (const viewKey of listPlayableViewKeys()) {
    const player = getViewPlayer(viewKey);
    if (!isPlayerSettledAtFrame(player, targetFrame)) {
      pending.add(viewKey);
    }
  }
  local.pendingViews = pending;
  return pending;
}

function collectPlaybackRefreshPendingViewKeys(targetFrame, toleranceFrames = PLAYBACK_STOP_TOLERANCE_FRAMES) {
  const pending = new Set();
  for (const viewKey of listPlayableViewKeys()) {
    const player = getViewPlayer(viewKey);
    if (!isPlayerWithinFrameTolerance(player, targetFrame, toleranceFrames)) {
      pending.add(viewKey);
    }
  }
  local.playbackRefreshPendingViews = pending;
  return pending;
}

function startPlaybackSyncLoop() {
  if (local.playbackSyncTimer !== null) return;
  local.playbackSyncTimer = window.setInterval(() => {
    if (local.isSyncSettling || local.isPlaybackRefreshing || !isDriverPlaying()) return;
    syncOtherPlayersToDriver(false);
  }, PLAYBACK_SYNC_INTERVAL_MS);
}

function stopPlaybackSyncLoop() {
  window.clearInterval(local.playbackSyncTimer);
  local.playbackSyncTimer = null;
}

function stopPlaybackBoundaryWatch() {
  const driver = getDriverPlayer();
  if (local.playbackBoundaryWatchMode === "video") {
    if (driver && typeof driver.cancelVideoFrameCallback === "function" && local.playbackBoundaryWatchHandle !== null) {
      try {
        driver.cancelVideoFrameCallback(local.playbackBoundaryWatchHandle);
      } catch (_) {
        // ignore cancellation errors on detached players
      }
    }
  } else if (local.playbackBoundaryWatchMode === "raf" && local.playbackBoundaryWatchHandle !== null) {
    window.cancelAnimationFrame(local.playbackBoundaryWatchHandle);
  }
  local.playbackBoundaryWatchHandle = null;
  local.playbackBoundaryWatchMode = "";
}

function getPlaybackBoundaryTarget(video = currentVideo()) {
  const segment = shouldConstrainPlaybackToSelectedSegment() ? getSelectedSegment(video) : null;
  if (!video || !segment) return null;
  return {
    segment,
    startFrame: Number(segment.start_frame) || 0,
    stopAnchorFrame: getStopAnchorFrame(segment),
  };
}

function schedulePlaybackBoundaryWatch() {
  const driver = getDriverPlayer();
  if (!driver || local.playbackBoundaryWatchHandle !== null) return;

  const tick = () => {
    local.playbackBoundaryWatchHandle = null;
    local.playbackBoundaryWatchMode = "";
    if (!isDriverPlaying() || local.isSyncSettling || local.isPlaybackRefreshing) return;
    if (checkPlaybackBoundaryNow()) return;
    schedulePlaybackBoundaryWatch();
  };

  if (typeof driver.requestVideoFrameCallback === "function") {
    local.playbackBoundaryWatchMode = "video";
    local.playbackBoundaryWatchHandle = driver.requestVideoFrameCallback(() => tick());
    return;
  }

  local.playbackBoundaryWatchMode = "raf";
  local.playbackBoundaryWatchHandle = window.requestAnimationFrame(() => tick());
}

function startPlaybackBoundaryWatch() {
  if (local.playbackBoundaryWatchHandle !== null) return;
  if (!isDriverPlaying()) return;
  if (!getPlaybackBoundaryTarget()) return;
  schedulePlaybackBoundaryWatch();
}

function syncOtherPlayersToDriver(force = false) {
  const driver = getDriverPlayer();
  const video = currentVideo();
  if (!driver || !video) return;
  const driverTime = driver.currentTime;
  const driverFrame = timeToFrame(video, driverTime || 0);
  for (const [viewKey, player] of local.viewPlayers.entries()) {
    if (!player || viewKey === local.driverViewKey) continue;
    const cfg = local.viewPlayback.get(viewKey);
    if (!cfg || cfg.unavailable) continue;
    const driftFrames = Math.abs(timeToFrame(video, player.currentTime || 0) - driverFrame);
    if (force || driftFrames > PLAYBACK_SYNC_DRIFT_FRAMES) {
      if (!player.seeking && Math.abs((player.currentTime || 0) - driverTime) <= 0.0005) continue;
      try {
        markPlayerSeekIssued(player);
        player.currentTime = driverTime;
      } catch (_) {
        // ignore seek errors for partially loaded players
      }
    }
  }
}

function nudgePlayersTowardFrame(targetFrame, toleranceFrames = PLAYBACK_STOP_TOLERANCE_FRAMES) {
  const video = currentVideo();
  if (!video) return;
  const targetTime = frameToSeekTime(video, targetFrame);
  for (const viewKey of listPlayableViewKeys()) {
    const player = getViewPlayer(viewKey);
    if (!player) continue;
    if (isPlayerWithinFrameTolerance(player, targetFrame, toleranceFrames)) continue;
    if (!player.seeking && Math.abs((player.currentTime || 0) - targetTime) <= 0.0005) continue;
    try {
      markPlayerSeekIssued(player);
      player.currentTime = targetTime;
    } catch (_) {
      // ignore seek errors for partially loaded players
    }
  }
}

function setAllPlayersTime(targetTime, force = false) {
  for (const [viewKey, player] of local.viewPlayers.entries()) {
    const cfg = local.viewPlayback.get(viewKey);
    if (!player || !cfg || cfg.unavailable) continue;
    if (Math.abs((player.currentTime || 0) - targetTime) <= 0.0005 && !player.seeking) continue;
    if (!force && Math.abs((player.currentTime || 0) - targetTime) <= 0.02) continue;
    try {
      markPlayerSeekIssued(player);
      player.currentTime = targetTime;
    } catch (_) {
      // ignore seek errors for partially loaded players
    }
  }
}

async function playAllPlayers() {
  const video = currentVideo();
  if (!video) return;
  cancelPlaybackRefresh({ keepStable: false, clearPendingAction: false });
  clearPlaybackStable();
  const driver = getDriverPlayer();
  const targetTime = driver
    ? (driver.currentTime || 0)
    : frameToSeekTime(video, state.currentFrame);
  setAllPlayersTime(targetTime, true);
  clearFrameSettled();
  const playables = listPlayableViewKeys();
  const startTasks = playables.map(viewKey => {
    const player = getViewPlayer(viewKey);
    if (!player) return Promise.resolve();
    player.playbackRate = local.playbackRate;
    return player.play().catch(() => undefined);
  });
  await Promise.allSettled(startTasks);
  syncOtherPlayersToDriver(true);
  startPlaybackSyncLoop();
  startPlaybackBoundaryWatch();
  updateInteractionAvailability();
}

function pauseAllPlayers() {
  stopPlaybackSyncLoop();
  stopPlaybackBoundaryWatch();
  const playables = listPlayableViewKeys();
  for (const viewKey of playables) {
    const player = getViewPlayer(viewKey);
    if (!player) continue;
    player.pause();
  }
}

function flushPendingAnnotationAction(frame) {
  const pending = local.pendingAnnotationAction;
  clearPendingAnnotationAction();
  if (!pending || typeof pending.run !== "function") return;
  pending.run(frame);
}

async function requestFrameSettle(frame, options = {}) {
  const {
    resumeAfterSettle = false,
    clearPendingActionOnFailure = true,
    failureMessage = "视角未完全对齐，未写入边界",
  } = options;
  const video = currentVideo();
  if (!video) return false;

  const safeFrame = clamp(frame, 0, Math.max(video.frames - 1, 0));
  setPlayheadFrameValue(safeFrame);
  ensureVisibleTimelineFrame(safeFrame);
  clearFrameSettled();
  clearPlaybackStable();

  if (!local.hasPlayableViews) {
    markFrameSettled(safeFrame);
    renderFrameOnly();
    updateInteractionAvailability();
    return true;
  }

  const requestId = local.syncRequestId + 1;
  local.syncRequestId = requestId;
  local.isSyncSettling = true;
  local.syncTargetFrame = safeFrame;
  cancelPlaybackRefresh({ keepStable: false, clearPendingAction: false });
  stopPlaybackBoundaryWatch();
  clearSyncTimers();
  stopPlaybackSyncLoop();
  collectPendingViewKeys(safeFrame);
  renderFrameOnly();
  updateInteractionAvailability();
  renderStatus();

  pauseAllPlayers();
  syncPlayPauseBtn(false);

  const targetTime = frameToSeekTime(video, safeFrame);
  setAllPlayersTime(targetTime, true);

  return new Promise(resolve => {
    const finish = success => {
      if (requestId !== local.syncRequestId) {
        resolve(false);
        return;
      }

      const pending = collectPendingViewKeys(safeFrame);
      cancelFrameSettle({
        keepSettled: false,
        clearPendingAction: !success && clearPendingActionOnFailure,
      });
      setPlayheadFrameValue(safeFrame);

      if (!success) {
        renderFrameOnly();
        updateInteractionAvailability();
        renderStatus();
        if (failureMessage) {
          const suffix = pending.size ? `: ${[...pending].join(", ")}` : "";
          showToast(`${failureMessage}${suffix}`, "error");
        }
        resolve(false);
        return;
      }

      markFrameSettled(safeFrame);
      setAllPlayersTime(targetTime, true);
      syncOtherPlayersToDriver(true);
      renderFrameOnly();
      updateInteractionAvailability();
      renderStatus();
      flushPendingAnnotationAction(safeFrame);
      if (resumeAfterSettle) {
        void playAllPlayers();
      }
      resolve(true);
    };

    const check = () => {
      if (requestId !== local.syncRequestId) {
        clearSyncTimers();
        resolve(false);
        return;
      }
      const pending = collectPendingViewKeys(safeFrame);
      if (pending.size === 0) {
        finish(true);
      }
    };

    local.syncPollTimer = window.setInterval(check, SETTLE_POLL_MS);
    local.syncTimeoutTimer = window.setTimeout(() => finish(false), SETTLE_TIMEOUT_MS);
    check();
  });
}

function continueAfterPlaybackStop(frame, options = {}) {
  const {
    resumeFrame = null,
    resumeFailureMessage = "视角未完全对齐，单段循环已暂停",
  } = options;
  if (local.pendingAnnotationAction) {
    void requestFrameSettle(frame, {
      failureMessage: "视角未完全对齐，未写入边界",
    });
    return;
  }
  if (Number.isInteger(resumeFrame)) {
    void requestFrameSettle(resumeFrame, {
      resumeAfterSettle: true,
      clearPendingActionOnFailure: false,
      failureMessage: resumeFailureMessage,
    });
  }
}

async function finalizePlaybackStop(frame, options = {}) {
  const {
    toleranceFrames = PLAYBACK_STOP_TOLERANCE_FRAMES,
    clearPendingActionOnFailure = false,
    failureMessage = "视角未稳定停在目标边界附近",
    resumeFrame = null,
    resumeFailureMessage = "视角未完全对齐，单段循环已暂停",
  } = options;
  const video = currentVideo();
  if (!video) return false;

  const safeFrame = clamp(frame, 0, Math.max(video.frames - 1, 0));
  setPlayheadFrameValue(safeFrame);
  ensureVisibleTimelineFrame(safeFrame);
  clearFrameSettled();
  clearPlaybackStable();

  if (!local.hasPlayableViews) {
    markPlaybackStable(safeFrame);
    renderFrameOnly();
    updateInteractionAvailability();
    renderStatus();
    continueAfterPlaybackStop(safeFrame, { resumeFrame, resumeFailureMessage });
    return true;
  }

  const requestId = local.playbackRefreshRequestId + 1;
  local.playbackRefreshRequestId = requestId;
  local.isPlaybackRefreshing = true;
  local.playbackRefreshTargetFrame = safeFrame;
  clearPlaybackRefreshTimers();
  stopPlaybackBoundaryWatch();
  stopPlaybackSyncLoop();
  collectPlaybackRefreshPendingViewKeys(safeFrame, toleranceFrames);
  renderFrameOnly();
  updateInteractionAvailability();
  renderStatus();

  pauseAllPlayers();
  syncPlayPauseBtn(false);

  nudgePlayersTowardFrame(safeFrame, toleranceFrames);

  return new Promise(resolve => {
    const finish = success => {
      if (requestId !== local.playbackRefreshRequestId) {
        resolve(false);
        return;
      }

      const pending = collectPlaybackRefreshPendingViewKeys(safeFrame, toleranceFrames);
      cancelPlaybackRefresh({
        keepStable: false,
        clearPendingAction: !success && clearPendingActionOnFailure,
      });
      setPlayheadFrameValue(safeFrame);

      if (!success) {
        renderFrameOnly();
        updateInteractionAvailability();
        renderStatus();
        if (failureMessage) {
          const suffix = pending.size ? `: ${[...pending].join(", ")}` : "";
          showToast(`${failureMessage}${suffix}`, "warning");
        }
        if (local.pendingAnnotationAction) {
          void requestFrameSettle(safeFrame, {
            failureMessage: "视角未完全对齐，未写入边界",
          });
        }
        resolve(false);
        return;
      }

      markPlaybackStable(safeFrame);
      renderFrameOnly();
      updateInteractionAvailability();
      renderStatus();
      continueAfterPlaybackStop(safeFrame, { resumeFrame, resumeFailureMessage });
      resolve(true);
    };

    const check = () => {
      if (requestId !== local.playbackRefreshRequestId) {
        clearPlaybackRefreshTimers();
        resolve(false);
        return;
      }
      const pending = collectPlaybackRefreshPendingViewKeys(safeFrame, toleranceFrames);
      if (pending.size === 0) {
        finish(true);
      }
    };

    local.playbackRefreshPollTimer = window.setInterval(check, PLAYBACK_REFRESH_POLL_MS);
    local.playbackRefreshTimeoutTimer = window.setTimeout(() => finish(false), PLAYBACK_REFRESH_TIMEOUT_MS);
    check();
  });
}

function checkPlaybackBoundaryNow() {
  const video = currentVideo();
  const driver = getDriverPlayer();
  const target = getPlaybackBoundaryTarget(video);
  if (!video || !driver || !target || local.isSyncSettling || local.isPlaybackRefreshing) return false;

  const frame = timeToFrame(video, driver.currentTime || 0);
  if (frame < target.stopAnchorFrame) return false;
  if (local.loopSegment) {
    void finalizePlaybackStop(target.stopAnchorFrame, {
      resumeFrame: target.startFrame,
      resumeFailureMessage: "视角未完全对齐，单段循环已暂停",
      failureMessage: "视角未稳定停在段尾，单段循环已暂停",
      clearPendingActionOnFailure: false,
    });
    return true;
  }

  local.segmentPlaybackActive = false;
  void finalizePlaybackStop(target.stopAnchorFrame, {
    failureMessage: "视角未稳定停在段尾，已停在目标边界附近",
    clearPendingActionOnFailure: false,
  });
  return true;
}

function renderStatus() {
  const mode = state.statusMode;
  ui.statusDraft.classList.toggle("status-inactive", mode !== "draft");
  ui.statusCommitted.classList.toggle("status-inactive", mode !== "committed");
  ui.statusSaved.classList.toggle("status-inactive", mode !== "saved");
  if (!ui.statusHint) return;
  updateWorkflowControls();

  if (state.isLoadingVideo) {
    ui.statusHint.textContent = "正在加载视频 / 视角 / 播放配置";
    return;
  }

  if (state.collab.enabled && !state.collab.annotatorId) {
    ui.statusHint.textContent = "请选择标注员";
    return;
  }

  if (!currentVideo()) {
    ui.statusHint.textContent = state.meta.length ? "请选择一个视频" : "无视频";
    return;
  }

  if (local.isPlaybackRefreshing) {
    ui.statusHint.textContent = "已到段尾，正在刷新最终帧…";
    return;
  }

  if (local.isSyncSettling) {
    ui.statusHint.textContent = "正在对齐视角，分段暂不可用";
    return;
  }

  if (mode === "draft") {
    ui.statusHint.textContent = "本地草稿未应用";
    return;
  }

  if (mode === "committed") {
    const appliedAt = formatStatusTimestamp(state.lastAppliedAt);
    ui.statusHint.textContent = appliedAt
      ? `已应用到会话，未保存到文件 · ${appliedAt}`
      : "已应用到会话，未保存到文件";
    return;
  }

  const savedAt = formatStatusTimestamp(state.lastSavedAt);
  ui.statusHint.textContent = savedAt ? `已保存到文件 · ${savedAt}` : "来自当前文件";
}

function currentActiveMs() {
  if (!local.auditOpenedAt) return 0;
  return Math.max(0, Date.now() - local.auditOpenedAt);
}

function buildAuditContext() {
  const activeMs = currentActiveMs();
  return {
    active_ms: activeMs,
    lead_time_ms: activeMs,
  };
}

function markMetaStatusOpened(index) {
  const current = state.meta[index]?.annotation_status;
  if (!current || current.status === "unopened") {
    setAnnotationStatus(index, {
      ...current,
      status: "opened",
    });
    renderMetaSelect();
  }
}

function trackItemOpen(index) {
  if (index < 0 || index >= state.meta.length) return;
  local.auditActiveIndex = index;
  local.auditOpenedAt = Date.now();
  markMetaStatusOpened(index);
  sendAuditEvent({
    event_type: "item_open",
    index,
    reason: "load",
  });
}

function trackCurrentItemClose(reason = "leave") {
  if (local.auditActiveIndex < 0) return;
  const index = local.auditActiveIndex;
  const activeMs = currentActiveMs();
  local.auditActiveIndex = -1;
  local.auditOpenedAt = null;
  sendAuditEvent({
    event_type: "item_close",
    index,
    active_ms: activeMs,
    lead_time_ms: activeMs,
    reason,
  });
}

function renderUndoRedoButtons() {
  ui.undoBtn.disabled = isCurrentDiscarded() || state.undoStack.length === 0;
  ui.redoBtn.disabled = isCurrentDiscarded() || state.redoStack.length === 0;
}

function markDraftAndSchedule() {
  markDraft();
  renderStatus();
  scheduleDraftSave();
}

function scheduleDraftSave() {
  window.clearTimeout(local.draftSaveTimer);
  local.draftSaveTimer = window.setTimeout(() => {
    local.draftSaveTimer = null;
    persistDraftNow();
  }, 300);
}

function persistDraftNow() {
  window.clearTimeout(local.draftSaveTimer);
  local.draftSaveTimer = null;
  const video = currentVideo();
  if (!video || (!state.dirty && !state.hasUnsavedFileChanges)) return;
  saveDraft(getDraftKey(video), {
    savedAt: Date.now(),
    serverHash: snapshotHash(state.serverSnapshot),
    workingCopy: cloneDeep(video),
  });
}

function clearCurrentDraft() {
  const video = currentVideo();
  if (!video) return;
  clearDraft(getDraftKey(video));
}

function clearDraftByKey(key) {
  clearDraft(key);
}

function renderShortcutFlash(tag) {
  if (!tag) return;
  const item = document.querySelector(`.shortcut-group[data-shortcut="${tag}"]`);
  if (!item) return;
  item.classList.add("shortcut-active");
  window.setTimeout(() => item.classList.remove("shortcut-active"), SHORTCUT_FLASH_MS);
}

function setPathInfo(inputPath, outputPath) {
  if (typeof inputPath === "string") {
    ui.inputPath.textContent = inputPath;
    ui.inputJsonField.value = inputPath;
  }
  if (typeof outputPath === "string") {
    ui.outputPath.textContent = outputPath;
  }
}

function setItemError(message) {
  if (!message) {
    ui.itemError.textContent = "";
    ui.itemError.classList.add("hidden");
    return;
  }
  ui.itemError.textContent = message;
  ui.itemError.classList.remove("hidden");
}

function formatFieldErrors(error) {
  const fieldErrors = Array.isArray(error?.fieldErrors)
    ? error.fieldErrors
    : Array.isArray(error?.payload?.field_errors)
      ? error.payload.field_errors
      : [];
  if (!fieldErrors.length) return "";
  return fieldErrors
    .slice(0, 3)
    .map(item => `${item.field || "field"}: ${item.message || "invalid"}`)
    .join(" | ");
}

function setValidationError(error) {
  const detail = formatFieldErrors(error);
  if (detail) {
    setItemError(detail);
    return;
  }
  setItemError(error?.message || "");
}

function isPlaybackDisabled() {
  const video = currentVideo();
  return state.isLoadingVideo || !video || !!video.unavailable || !local.hasPlayableViews;
}

function canRunBoundaryMutation() {
  return !isCurrentDiscarded() && !isPlaybackDisabled() && !local.isSyncSettling;
}

function setAnnotationEditingAvailability() {
  const disabled = state.isLoadingVideo || !currentVideo() || isCurrentDiscarded();
  const controls = [
    ui.tasksInput,
    ui.actionTextFilterInput,
    ui.activeActionSelect,
    ui.activeActionInput,
    ui.skillFilterInput,
    ui.activeSkillSelect,
    ui.addSegmentBtn,
    ui.setStartBtn,
    ui.setEndBtn,
    ui.applyBtn,
  ];
  controls.forEach(control => {
    if (control) control.disabled = disabled;
  });
  if (ui.undoBtn) ui.undoBtn.disabled = disabled || state.undoStack.length === 0;
  if (ui.redoBtn) ui.redoBtn.disabled = disabled || state.redoStack.length === 0;
  ui.boundaryHandles?.querySelectorAll(".boundary-handle").forEach(handle => {
    handle.disabled = disabled || !canDragBoundary();
  });
  ui.segmentRows?.querySelectorAll('[data-role="delete"]').forEach(button => {
    button.disabled = disabled;
  });
}

function canDragBoundary() {
  return canRunBoundaryMutation() && !isDriverPlaying() && local.isFrameSettled;
}

function syncAwareSelectionHint(defaultText) {
  if (local.pendingAnnotationAction) {
    return local.isPlaybackRefreshing ? "刷新最终帧后执行…" : "暂停并对齐后执行…";
  }
  if (local.isPlaybackRefreshing) return "刷新最终帧…";
  if (local.isSyncSettling) return "对齐中…";
  return defaultText;
}

function setControlAvailability(enabled) {
  if (!enabled) {
    pauseAllPlayers();
    cancelPlaybackRefresh({ keepStable: false, clearPendingAction: false });
    local.segmentPlaybackActive = false;
  }
  const controls = [
    ui.jumpBack10Btn,
    ui.playPauseBtn,
    ui.prevFrameBtn,
    ui.nextFrameBtn,
    ui.jumpForward10Btn,
    ui.playSegmentBtn,
    ui.frameSlider,
    ui.autoStopToggle,
    ui.loopSegmentToggle,
    ui.playbackRateSelect,
    ui.timelineZoomSelect,
    ui.addSegmentBtn,
    ui.setStartBtn,
    ui.setEndBtn,
    ui.jumpSegmentStartBtn,
    ui.jumpSegmentEndBtn,
  ];
  controls.forEach(control => {
    if (control) control.disabled = !enabled;
  });
  ui.videoWrap.classList.toggle("video-disabled", !enabled);
  ui.chapterBar.classList.toggle("disabled", !enabled);
  ui.boundaryMarkers.classList.toggle("disabled", !enabled);
  ui.boundaryHandles.classList.toggle("disabled", !enabled);
}

function updateInteractionAvailability() {
  const enabled = !state.isLoadingVideo && !isPlaybackDisabled();
  setControlAvailability(enabled);
  if (!enabled) {
    setAnnotationEditingAvailability();
    return;
  }

  const settling = local.isSyncSettling || local.isPlaybackRefreshing;
  const transportControls = [
    ui.jumpBack10Btn,
    ui.playPauseBtn,
    ui.prevFrameBtn,
    ui.nextFrameBtn,
    ui.jumpForward10Btn,
    ui.frameSlider,
    ui.playSegmentBtn,
    ui.jumpSegmentStartBtn,
    ui.jumpSegmentEndBtn,
    ui.timelineZoomSelect,
    ui.playbackRateSelect,
    ui.autoStopToggle,
    ui.loopSegmentToggle,
  ];
  transportControls.forEach(control => {
    if (control) control.disabled = settling;
  });

  const mutationControls = [ui.addSegmentBtn, ui.setStartBtn, ui.setEndBtn];
  mutationControls.forEach(control => {
    if (control) control.disabled = !canRunBoundaryMutation();
  });

  const canDrag = canDragBoundary();
  ui.boundaryHandles.classList.toggle("disabled", !canDrag);
  ui.boundaryHandles.querySelectorAll(".boundary-handle").forEach(handle => {
    handle.disabled = !canDrag;
  });
  setAnnotationEditingAvailability();
}

function showVideoError(message) {
  ui.videoError.textContent = message;
  ui.videoError.classList.remove("hidden");
}

function hideVideoError() {
  ui.videoError.classList.add("hidden");
}

function applyVideoGridLayout(viewCount) {
  ui.videoGrid.classList.remove("video-grid-1", "video-grid-2", "video-grid-3", "video-grid-4");
  const safeCount = Math.max(1, Math.min(4, Number(viewCount) || 1));
  ui.videoGrid.classList.add(`video-grid-${safeCount}`);
}

function setActiveView(viewKey, options = {}) {
  const { silent = false } = options;
  if (!viewKey) return;
  if (!local.viewPlayback.has(viewKey)) return;

  local.activeViewKey = viewKey;
  const cfg = local.viewPlayback.get(viewKey) || {};
  const tileNodes = ui.videoGrid.querySelectorAll(".video-tile");
  tileNodes.forEach(tile => {
    tile.classList.toggle("active", tile.dataset.viewKey === viewKey);
  });

  if (ui.activeViewBadge) {
    ui.activeViewBadge.textContent = `view ${viewKey}`;
  }

  if (!silent && cfg.unavailable) {
    showToast(`视角 ${viewKey} 不可播放`, "info", 1800);
  }
}

function handleViewPlaybackError(viewKey, errorNode) {
  const cfg = local.viewPlayback.get(viewKey);
  const player = getViewPlayer(viewKey);
  if (!cfg || !player) return;

  const canFallback = Boolean(cfg.fallbackUrl)
    && cfg.fallbackAttempts < cfg.fallbackRetryMax
    && normalizeUrl(player.currentSrc) !== normalizeUrl(cfg.fallbackUrl);

  if (canFallback) {
    cfg.fallbackAttempts += 1;
    player.src = cfg.fallbackUrl;
    player.load();
    showToast(`视角 ${viewKey} 直连失败，已切换代理播放`, "info", 2600);
    return;
  }

  cfg.unavailable = true;
  local.hasPlayableViews = listPlayableViewKeys().length > 0;
  const safeKey = (typeof CSS !== "undefined" && typeof CSS.escape === "function")
    ? CSS.escape(viewKey)
    : String(viewKey).replace(/["\\]/g, "\\$&");
  const tile = ui.videoGrid.querySelector(`.video-tile[data-view-key="${safeKey}"]`);
  if (tile) tile.classList.add("unavailable");
  if (errorNode) {
    errorNode.classList.remove("hidden");
    errorNode.textContent = `视角 ${viewKey} 视频加载错误`;
  }

  if (local.driverViewKey === viewKey) {
    local.driverViewKey = listPlayableViewKeys()[0] || "";
    const driver = getDriverPlayer();
    syncPlayPauseBtn(driver ? !driver.paused : false);
  }
  if (local.activeViewKey === viewKey) {
    const nextActive = local.driverViewKey || viewKey;
    setActiveView(nextActive, { silent: true });
  }

  if (!local.hasPlayableViews) {
    setControlAvailability(false);
    showVideoError("所有视角均不可播放");
  }
  updateInteractionAvailability();
}

function renderVideoTiles(views, index, primaryViewKey) {
  ui.videoGrid.innerHTML = "";
  applyVideoGridLayout(1);
  local.viewPlayers = new Map();
  local.viewPlayback = new Map();
  local.hasPlayableViews = false;
  local.driverViewKey = "";
  local.activeViewKey = "";

  const list = Array.isArray(views) ? views : [];
  const safeList = list.length ? list.slice(0, 4) : [];
  applyVideoGridLayout(safeList.length || 1);

  safeList.forEach(view => {
    const viewKey = String(view?.key || "").trim() || "main";
    const viewPath = String(view?.path || "");
    const viewUnavailable = Boolean(view?.unavailable);
    const viewError = String(view?.error || "");
    const playback = view && typeof view.playback === "object" ? view.playback : {};
    const streamUrl = makeViewStreamUrl(index, viewKey);
    const primaryUrl = String(playback.primary_url || streamUrl);
    const fallbackUrl = String(playback.fallback_url || "");
    const fallbackRetryRaw = Number(playback.fallback_retry_max);
    const fallbackRetryMax = Number.isFinite(fallbackRetryRaw) && fallbackRetryRaw >= 0
      ? Math.floor(fallbackRetryRaw)
      : 0;
    const tile = document.createElement("div");
    tile.className = "video-tile";
    tile.dataset.viewKey = viewKey;
    if (viewUnavailable) tile.classList.add("unavailable");

    const header = document.createElement("div");
    header.className = "video-tile-header";
    const nameEl = document.createElement("span");
    nameEl.className = "video-tile-name";
    nameEl.textContent = viewKey;
    nameEl.title = `${viewKey}: ${viewPath}`;
    const statusEl = document.createElement("span");
    statusEl.className = "video-tile-status";
    statusEl.textContent = viewUnavailable ? "Unavailable" : "Ready";
    header.appendChild(nameEl);
    header.appendChild(statusEl);

    const playerWrap = document.createElement("div");
    playerWrap.className = "video-tile-player-wrap";

    const player = document.createElement("video");
    player.preload = "metadata";
    player.playsInline = true;
    player.dataset.viewKey = viewKey;
    player.playbackRate = local.playbackRate;
    playerWrap.appendChild(player);

    const tileError = document.createElement("div");
    tileError.className = "video-tile-error hidden";
    tileError.textContent = viewError || `视角 ${viewKey} 不可播放`;
    if (viewUnavailable) {
      tileError.classList.remove("hidden");
    }
    playerWrap.appendChild(tileError);

    tile.appendChild(header);
    tile.appendChild(playerWrap);
    ui.videoGrid.appendChild(tile);

    local.viewPlayers.set(viewKey, player);
    local.viewPlayback.set(viewKey, {
      key: viewKey,
      path: viewPath,
      unavailable: viewUnavailable,
      error: viewError,
      primaryUrl,
      fallbackUrl,
      fallbackAttempts: 0,
      fallbackRetryMax,
    });

    if (!viewUnavailable) {
      local.hasPlayableViews = true;
      player.src = primaryUrl;
      player.load();
    }

    tile.addEventListener("click", () => {
      setActiveView(viewKey);
    });

    // When a non-driver view finishes loading metadata, sync it to the driver's
    // current position so all views start at the same frame.
    player.addEventListener("loadedmetadata", () => {
      try {
        if (local.isSyncSettling && Number.isInteger(local.syncTargetFrame)) {
          const video = currentVideo();
          if (video) {
            markPlayerSeekIssued(player);
            player.currentTime = frameToSeekTime(video, local.syncTargetFrame);
            return;
          }
        }
        if (local.isPlaybackRefreshing && Number.isInteger(local.playbackRefreshTargetFrame)) {
          const video = currentVideo();
          if (video) {
            markPlayerSeekIssued(player);
            player.currentTime = frameToSeekTime(video, local.playbackRefreshTargetFrame);
            return;
          }
        }
        if (viewKey === local.driverViewKey) return;
        const driver = getDriverPlayer();
        if (!driver) return;
        if (!player.seeking && Math.abs((player.currentTime || 0) - (driver.currentTime || 0)) <= 0.0005) return;
        markPlayerSeekIssued(player);
        player.currentTime = driver.currentTime;
      } catch (_) {
        // ignore seek errors on partially loaded players
      }
    });

    player.addEventListener("play", () => {
      if (viewKey !== local.driverViewKey) return;
      clearFrameSettled();
      clearPlaybackStable();
      syncPlayPauseBtn(true);
      startPlaybackSyncLoop();
      startPlaybackBoundaryWatch();
      updateInteractionAvailability();
    });
    player.addEventListener("pause", () => {
      if (viewKey !== local.driverViewKey) return;
      syncPlayPauseBtn(false);
      stopPlaybackSyncLoop();
      stopPlaybackBoundaryWatch();
      updateInteractionAvailability();
    });
    player.addEventListener("ended", () => {
      if (viewKey !== local.driverViewKey) return;
      local.segmentPlaybackActive = false;
      clearPlaybackStable();
      syncPlayPauseBtn(false);
      stopPlaybackSyncLoop();
      stopPlaybackBoundaryWatch();
      updateInteractionAvailability();
    });
    player.addEventListener("error", () => {
      handleViewPlaybackError(viewKey, tileError);
    });
    player.addEventListener("seeked", () => {
      markPlayerSeekCompleted(player);
      if (viewKey !== local.driverViewKey) return;
      if (local.isSyncSettling || local.isPlaybackRefreshing) return;
      const video = currentVideo();
      if (!video) return;
      const frame = timeToFrame(video, player.currentTime);
      if (frame !== state.currentFrame) {
        setCurrentFrame(frame, false);
      }
    });
    player.addEventListener("timeupdate", () => {
      if (viewKey !== local.driverViewKey) return;
      const video = currentVideo();
      if (!video || local.isScrubbing || local.syncingFromPrimary || local.isSyncSettling || local.isPlaybackRefreshing) return;
      local.syncingFromPrimary = true;
      try {
        const frame = timeToFrame(video, player.currentTime);
        if (frame !== state.currentFrame) {
          setCurrentFrame(frame, false);
        }
        syncOtherPlayersToDriver(false);
      } finally {
        local.syncingFromPrimary = false;
      }
    });
  });

  if (safeList.length === 3) {
    const placeholder = document.createElement("div");
    placeholder.className = "video-tile-placeholder";
    ui.videoGrid.appendChild(placeholder);
  }

  const preferredDriver = String(primaryViewKey || "").trim();
  if (preferredDriver && local.viewPlayback.has(preferredDriver) && !local.viewPlayback.get(preferredDriver)?.unavailable) {
    local.driverViewKey = preferredDriver;
  } else {
    local.driverViewKey = listPlayableViewKeys()[0] || (safeList[0]?.key || "");
  }

  const initialActive = local.driverViewKey || (safeList[0]?.key || "");
  setActiveView(initialActive, { silent: true });
  syncPlayPauseBtn(false);
  updateInteractionAvailability();
}

function renderVideoPosition() {
  if (!state.meta.length) {
    ui.videoPosition.textContent = "- / -";
    syncVideoJumpInput();
    return;
  }
  if (state.currentIndex < 0) {
    ui.videoPosition.textContent = `- / ${state.meta.length}`;
    syncVideoJumpInput();
    return;
  }
  ui.videoPosition.textContent = `${state.currentIndex + 1} / ${state.meta.length}`;
  syncVideoJumpInput();
}

function renderMetaSelect() {
  ui.videoSelect.innerHTML = "";
  state.meta.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    const filename = String(item.path || "").split("/").pop();
    option.textContent = `ep ${item.episode_index} | ${filename}${getAnnotationStatusLabel(item)}${getReviewMetaLabel(item)}`;
    if (item.discarded) option.textContent += " [已丢弃]";
    if (item.unavailable) option.textContent += " [N/A]";
    if (item.error) option.textContent += " [ERR]";
    ui.videoSelect.appendChild(option);
  });
  if (state.currentIndex >= 0) {
    ui.videoSelect.value = String(state.currentIndex);
  }
}

function renderItemStatus() {
  const video = currentVideo();
  const discarded = Boolean(video?.discarded);
  ui.itemStatusCard?.classList.toggle("discarded", discarded);
  if (ui.itemStatusIcon) ui.itemStatusIcon.textContent = discarded ? "!" : "✓";
  if (ui.itemStatusBadge) ui.itemStatusBadge.textContent = discarded ? "已丢弃" : "正常";
  if (ui.itemStatusDescription) {
    ui.itemStatusDescription.textContent = discarded
      ? "此条已丢弃 · 分段编辑已锁定"
      : "该条数据正常参与标注与合并。";
  }
  const note = String(video?.discard_note || "").trim();
  if (ui.itemStatusNote) {
    ui.itemStatusNote.textContent = note ? `备注：${note}` : "";
    ui.itemStatusNote.classList.toggle("hidden", !discarded || !note);
  }
  ui.itemStatusNormalActions?.classList.toggle("hidden", discarded || !video);
  ui.itemStatusDiscardedActions?.classList.toggle("hidden", !discarded || !video);
  if (ui.discardItemBtn) ui.discardItemBtn.disabled = local.isUpdatingDiscard || !video;
  if (ui.restoreItemBtn) ui.restoreItemBtn.disabled = local.isUpdatingDiscard || !video;
  setAnnotationEditingAvailability();
  updateWorkflowControls();
}

function renderFrameLabel() {
  const video = currentVideo();
  if (!video) {
    ui.frameLabel.textContent = "f -/-";
    ui.playbackTime.textContent = "00:00.000";
    return;
  }
  if (local.isSyncSettling) {
    ui.frameLabel.textContent = `f ${state.currentFrame}/${video.frames - 1} · 对齐中…`;
  } else if (local.isPlaybackRefreshing) {
    ui.frameLabel.textContent = `f ${state.currentFrame}/${video.frames - 1} · 刷新最终帧…`;
  } else {
    ui.frameLabel.textContent = `f ${state.currentFrame}/${video.frames - 1}`;
  }
  ui.playbackTime.textContent = formatTime(frameToTime(video, state.currentFrame));
}

function updateNeedlePosition() {
  const video = currentVideo();
  if (!video) {
    ui.currentFrameNeedle.style.left = "0%";
    return;
  }
  const ratio = frameToTimelineRatio(video, getTimelineWindow(video), state.currentFrame);
  ui.currentFrameNeedle.style.left = `${ratio * 100}%`;
}

function renderFrameOnly() {
  syncTimelineSliderWindow();
  renderFrameLabel();
  renderActiveSegmentFrameState();
  renderActiveSegmentValidationState();
  renderSegmentsHighlight();
  highlightActiveChapter();
  highlightActiveBoundary();
  updateNeedlePosition();
}

function syncPlayPauseBtn(isPlaying) {
  ui.playIcon.classList.toggle("hidden", isPlaying);
  ui.pauseIcon.classList.toggle("hidden", !isPlaying);
}

function renderGeneralInfo() {
  const video = currentVideo();
  if (!video) {
    ui.episodeValue.textContent = "-";
    if (ui.episodeValueInline) ui.episodeValueInline.textContent = "-";
    if (ui.activeViewBadge) ui.activeViewBadge.textContent = "view -";
    ui.pathValue.textContent = "-";
    ui.framesValue.textContent = "-";
    if (ui.framesValueInline) ui.framesValueInline.textContent = "-";
    ui.fpsValue.textContent = "-";
    ui.segmentCountValue.textContent = "-";
    ui.incompleteCountValue.textContent = "-";
    ui.avgSegmentLenValue.textContent = "-";
    ui.timelineSummary.textContent = "-";
    ui.tasksInput.value = "";
    setItemError("");
    renderItemStatus();
    return;
  }

  const stats = getVideoStats(video);
  ui.episodeValue.textContent = String(video.episode_index);
  if (ui.episodeValueInline) ui.episodeValueInline.textContent = String(video.episode_index);
  const views = Array.isArray(video.views) ? video.views : [];
  if (views.length > 0) {
    ui.pathValue.textContent = views
      .map(view => `${view.key}: ${view.path || "-"}`)
      .join("\n");
  } else {
    ui.pathValue.textContent = String(video.path || "-");
  }
  ui.framesValue.textContent = String(video.frames);
  if (ui.framesValueInline) ui.framesValueInline.textContent = `${video.frames}f`;
  ui.fpsValue.textContent = String(video.fps);
  ui.segmentCountValue.textContent = String(stats.totalSegments);
  ui.incompleteCountValue.textContent = String(stats.incompleteCount);
  ui.avgSegmentLenValue.textContent = `${stats.avgSegmentFrames.toFixed(1)}f`;
  ui.timelineSummary.textContent = `${stats.totalSegments} 段 · ${stats.incompleteCount} 未完成`;
  ui.tasksInput.value = (video.tasks || []).join("\n");
  setItemError(video.error || "");
  updateIncompleteNavButtons();
  renderItemStatus();
}

function segmentColor(index) {
  const paletteKey = `--seg-${Math.abs(Number(index) || 0) % 8}`;
  const value = getComputedStyle(document.documentElement).getPropertyValue(paletteKey).trim();
  return value || "#60a5fa";
}

function isSkillOutsideOptions(skill) {
  const normalized = String(skill || "").trim();
  if (!normalized) return false;
  return !state.skillOptions.includes(normalized);
}

function getModelReview(video = currentVideo()) {
  if (!video || !video.model_review || typeof video.model_review !== "object") return null;
  return video.model_review;
}

function getReviewIssues(video = currentVideo()) {
  const review = getModelReview(video);
  return Array.isArray(review?.issues) ? review.issues : [];
}

function issueMatchesSegment(issue, segment, index) {
  if (!issue || !segment) return false;
  const rawSegId = Number(issue.seg_id);
  if (!Number.isFinite(rawSegId)) return false;
  return rawSegId === Number(segment.seg_id) || rawSegId === Number(index);
}

function getSegmentReviewIssues(video, segment, index) {
  return getReviewIssues(video).filter(issue => issueMatchesSegment(issue, segment, index));
}

function getSegmentReviewSeverity(video, segment, index) {
  const issues = getSegmentReviewIssues(video, segment, index);
  if (!issues.length) return "";
  const hasActionable = issues.some(issue => String(issue.issue_type || "") !== "unclear_visual");
  return hasActionable ? "incorrect" : "uncertain";
}

function formatIssueType(issueType) {
  const labels = {
    boundary_error: "边界错误",
    wrong_object: "物体错误",
    wrong_action: "动作错误",
    wrong_bin: "垃圾桶错误",
    missing_action: "缺失动作",
    extra_action: "多余动作",
    naming_inconsistent: "命名不一致",
    format_error: "格式错误",
    unclear_visual: "画面不确定",
  };
  return labels[String(issueType || "")] || "模型问题";
}

function renderSegmentReviewBadgeHtml(video, segment, index) {
  const issues = getSegmentReviewIssues(video, segment, index);
  if (!issues.length) return "";
  const severity = getSegmentReviewSeverity(video, segment, index);
  const label = severity === "uncertain" ? `存疑 ${issues.length}` : `问题 ${issues.length}`;
  return `<span class="segment-review-badge ${severity}">${escapeHtml(label)}</span>`;
}

function getReviewMetaLabel(item) {
  const verdict = String(item?.review_verdict || "unreviewed");
  const issueCount = Number(item?.review_issue_count || 0);
  if (state.appMode !== "model-review" && verdict === "unreviewed") return "";
  if (verdict === "incorrect") return ` [问题 ${issueCount}]`;
  if (verdict === "uncertain") return ` [存疑 ${Math.max(issueCount, 1)}]`;
  if (verdict === "unreviewed") return " [未复审]";
  return "";
}

function getAnnotationStatusLabel(item) {
  const status = String(item?.annotation_status?.status || "unopened");
  const labels = {
    unopened: "未打开",
    opened: "已打开",
    modified: "已修改",
    saved: "已保存",
    validation_failed: "校验失败",
    reviewed: "已复审",
    legacy_output_without_audit: "旧输出",
  };
  return ` [${labels[status] || status}]`;
}

function hideModelReviewPanel() {
  if (!ui.modelReviewPanel) return;
  ui.modelReviewPanel.classList.add("hidden");
  ui.modelReviewPanel.classList.remove("review-incorrect", "review-uncertain");
  if (ui.modelReviewIssues) ui.modelReviewIssues.innerHTML = "";
}

function renderModelReviewPanel(segment = getSelectedSegment()) {
  const video = currentVideo();
  if (!ui.modelReviewPanel || !video || !segment) {
    hideModelReviewPanel();
    return;
  }

  const issues = getSegmentReviewIssues(video, segment, state.selectedSegIdx);
  if (!issues.length) {
    hideModelReviewPanel();
    return;
  }

  const severity = getSegmentReviewSeverity(video, segment, state.selectedSegIdx);
  ui.modelReviewPanel.classList.remove("hidden", "review-incorrect", "review-uncertain");
  ui.modelReviewPanel.classList.add(severity === "uncertain" ? "review-uncertain" : "review-incorrect");
  if (ui.modelReviewVerdict) {
    ui.modelReviewVerdict.textContent = severity === "uncertain" ? "模型复审：存疑" : "模型复审：错误";
  }
  if (ui.modelReviewCount) {
    ui.modelReviewCount.textContent = `${issues.length} 个问题`;
  }
  if (!ui.modelReviewIssues) return;

  ui.modelReviewIssues.innerHTML = "";
  issues.forEach(issue => {
    const issueEl = document.createElement("div");
    issueEl.className = "model-review-issue";

    const title = document.createElement("div");
    title.className = "model-review-issue-title";
    const confidence = Number(issue.confidence || 0);
    title.innerHTML =
      `<span>${escapeHtml(formatIssueType(issue.issue_type))}</span>` +
      `<span>${Number.isFinite(confidence) ? Math.round(confidence * 100) : 0}%</span>`;
    issueEl.appendChild(title);

    const reason = document.createElement("div");
    reason.className = "model-review-issue-text";
    reason.textContent = String(issue.reason || "模型未给出原因");
    issueEl.appendChild(reason);

    const fix = String(issue.suggested_fix || "").trim();
    if (fix) {
      const fixEl = document.createElement("div");
      fixEl.className = "model-review-issue-fix";
      fixEl.textContent = `建议：${fix}`;
      issueEl.appendChild(fixEl);
    }

    const evidenceFrames = Array.isArray(issue.evidence_frames) ? issue.evidence_frames : [];
    if (evidenceFrames.length) {
      const frameList = document.createElement("div");
      frameList.className = "model-review-frame-list";
      evidenceFrames.forEach(rawFrame => {
        const frame = Number(rawFrame);
        if (!Number.isFinite(frame)) return;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "model-review-frame-btn";
        button.textContent = `f${frame}`;
        button.title = `跳到证据帧 ${frame}`;
        button.addEventListener("click", event => {
          event.preventDefault();
          event.stopPropagation();
          setCurrentFrame(frame, true);
        });
        frameList.appendChild(button);
      });
      issueEl.appendChild(frameList);
    }

    ui.modelReviewIssues.appendChild(issueEl);
  });
}

function renderActiveSegmentValidationState(segment = getSelectedSegment()) {
  if (!segment) {
    ui.activeActionInput.classList.remove("field-invalid");
    ui.activeSkillSelect.classList.remove("field-invalid", "field-warning");
    if (ui.skillConfigHint) ui.skillConfigHint.classList.add("hidden");
    hideModelReviewPanel();
    return;
  }

  const completeness = getSegmentCompleteness(segment);
  const selectedSkill = String(segment.skill || "").trim();
  ui.activeActionInput.classList.toggle("field-invalid", completeness.missingAction);
  ui.activeSkillSelect.classList.toggle("field-invalid", completeness.missingSkill);
  ui.activeSkillSelect.classList.toggle(
    "field-warning",
    !completeness.missingSkill && isSkillOutsideOptions(selectedSkill),
  );
  if (ui.skillConfigHint) {
    ui.skillConfigHint.classList.toggle(
      "hidden",
      completeness.missingSkill || !isSkillOutsideOptions(selectedSkill),
    );
  }
}

function renderActiveSegmentFrameState() {
  const video = currentVideo();
  const segment = getSelectedSegment(video);
  if (!video || !segment) {
    ui.activeCurrentFrameValue.textContent = "-";
    ui.selectionHint.textContent = syncAwareSelectionHint("选择片段");
    return;
  }

  const currentSegIndex = findSegmentIndexByFrame(video, state.currentFrame);
  ui.activeCurrentFrameValue.textContent = String(state.currentFrame);
  ui.selectionHint.textContent = syncAwareSelectionHint(
    currentSegIndex === state.selectedSegIdx
      ? `编辑 seg ${segment.seg_id}`
      : `选择 seg ${segment.seg_id}`,
  );
}

function updateSegmentRow(index) {
  const video = currentVideo();
  if (!video || index < 0 || index >= video.action_config.length) return;

  const row = ui.segmentRows.querySelector(`.segment-row[data-seg-idx="${index}"]`);
  if (!row) {
    renderSegments();
    return;
  }

  const segment = video.action_config[index];
  const safeSegId = Number.isFinite(Number(segment?.seg_id)) ? Number(segment.seg_id) : index;
  const completeness = getSegmentCompleteness(segment);
  const actionText = String(segment.action_text || "").trim();
  const skillText = String(segment.skill || "").trim();
  const skillLabel = skillText || "未设 skill";
  const currentSegIndex = findSegmentIndexByFrame(video, state.currentFrame);
  const reviewSeverity = getSegmentReviewSeverity(video, segment, index);

  row.classList.toggle("active", index === state.selectedSegIdx);
  row.classList.toggle("current", index === currentSegIndex);
  row.classList.toggle("incomplete", !completeness.complete);
  row.classList.toggle("missing-action", completeness.missingAction);
  row.classList.toggle("missing-skill", completeness.missingSkill);
  row.classList.toggle("skill-outside-options", isSkillOutsideOptions(segment.skill));
  row.classList.toggle("segment-review-incorrect", reviewSeverity === "incorrect");
  row.classList.toggle("segment-review-uncertain", reviewSeverity === "uncertain");

  const segIdNode = row.querySelector(".segment-inline-id");
  if (segIdNode) {
    const segIdText = `seg ${safeSegId}`;
    segIdNode.textContent = segIdText;
    segIdNode.setAttribute("title", segIdText);
  }

  const skillNode = row.querySelector(".segment-inline-skill");
  if (skillNode) {
    skillNode.className = [
      "segment-inline-skill",
      !skillText ? "missing" : "",
      isSkillOutsideOptions(segment.skill) ? "outside" : "",
    ].filter(Boolean).join(" ");
    skillNode.textContent = truncate(skillLabel, 16);
    skillNode.setAttribute("title", skillLabel);
  }

  const reviewBadge = row.querySelector(".segment-review-badge");
  if (reviewBadge) reviewBadge.remove();
  const topRow = row.querySelector(".segment-row-top");
  if (topRow) {
    const badgeHtml = renderSegmentReviewBadgeHtml(video, segment, index);
    if (badgeHtml) topRow.insertAdjacentHTML("beforeend", badgeHtml);
  }

  const titleNode = row.querySelector(".segment-title");
  if (titleNode) {
    titleNode.innerHTML = actionText
      ? escapeHtml(actionText)
      : '<span class="segment-title-placeholder">未填写动作描述</span>';
    titleNode.setAttribute("title", actionText || "未填写动作描述");
  }
}

function renderActiveSegmentEditor() {
  const video = currentVideo();
  if (!video) {
    ui.activeEditorEmpty.classList.remove("hidden");
    ui.activeEditorBody.classList.add("hidden");
    ui.segmentBadge.textContent = "seg -";
    ui.activeSegmentMeta.textContent = "seg -";
    ui.activeSegmentRange.textContent = "f - → -";
    ui.selectionHint.textContent = syncAwareSelectionHint("选择片段");
    ui.activeActionInput.classList.remove("field-invalid");
    ui.activeSkillSelect.classList.remove("field-invalid", "field-warning");
    if (ui.skillConfigHint) ui.skillConfigHint.classList.add("hidden");
    hideModelReviewPanel();
    return;
  }

  const segment = video.action_config[state.selectedSegIdx];
  if (!segment) {
    ui.activeEditorEmpty.classList.remove("hidden");
    ui.activeEditorBody.classList.add("hidden");
    ui.activeActionInput.classList.remove("field-invalid");
    ui.activeSkillSelect.classList.remove("field-invalid", "field-warning");
    if (ui.skillConfigHint) ui.skillConfigHint.classList.add("hidden");
    return;
  }

  ui.activeEditorEmpty.classList.add("hidden");
  ui.activeEditorBody.classList.remove("hidden");

  const currentSegIndex = findSegmentIndexByFrame(video, state.currentFrame);
  const selectedSkill = String(segment.skill || "").trim();
  const activeMetaLabel = selectedSkill
    ? `seg ${segment.seg_id} · ${truncate(selectedSkill, 20)}`
    : `seg ${segment.seg_id}`;
  ui.activeSegmentMeta.textContent = activeMetaLabel;
  ui.segmentBadge.textContent = `seg ${segment.seg_id}`;
  ui.activeSegmentRange.textContent = formatRange(segment.start_frame, segment.end_frame);
  ui.activeStartValue.textContent = String(segment.start_frame);
  ui.activeEndValue.textContent = String(segment.end_frame);
  ui.activeDurationValue.textContent = formatDuration(video, segment);
  ui.activeCurrentFrameValue.textContent = String(state.currentFrame);
  ui.selectionHint.textContent = syncAwareSelectionHint(
    currentSegIndex === state.selectedSegIdx ? `编辑 seg ${segment.seg_id}` : `选择 seg ${segment.seg_id}`,
  );

  const selectedActionText = String(segment.action_text || "");
  if (document.activeElement !== ui.activeActionInput) {
    ui.activeActionInput.value = selectedActionText;
  }

  renderActionSelectOptions(ui.actionTextFilterInput.value, selectedActionText);
  ui.activeActionSelect.value = selectedActionText;
  if (selectedActionText && ui.activeActionSelect.value !== selectedActionText) {
    renderActionSelectOptions(ui.actionTextFilterInput.value, selectedActionText);
    ui.activeActionSelect.value = selectedActionText;
  }

  renderSkillSelectOptions(ui.skillFilterInput.value, selectedSkill);
  ui.activeSkillSelect.value = selectedSkill;
  if (selectedSkill && ui.activeSkillSelect.value !== selectedSkill) {
    renderSkillSelectOptions(ui.skillFilterInput.value, selectedSkill);
    ui.activeSkillSelect.value = selectedSkill;
  }

  renderActiveSegmentValidationState(segment);
  renderModelReviewPanel(segment);
}

function renderActionSelectOptions(filterText, selectedActionText) {
  const withSelected = ensureActionTextOption(state.actionTextOptions, selectedActionText);
  const options = filterActionTextOptions(withSelected, filterText);
  if (selectedActionText && !options.includes(selectedActionText)) {
    options.unshift(selectedActionText);
  }

  ui.activeActionSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "-- 选择 action text --";
  ui.activeActionSelect.appendChild(placeholder);

  for (const actionText of options) {
    const option = document.createElement("option");
    option.value = actionText;
    option.textContent = actionText;
    ui.activeActionSelect.appendChild(option);
  }
}

function renderSkillSelectOptions(filterText, selectedSkill) {
  const withSelected = ensureSkillOption(state.skillOptions, selectedSkill);
  const options = filterSkillOptions(withSelected, filterText);
  if (selectedSkill && !options.includes(selectedSkill)) {
    options.unshift(selectedSkill);
  }

  ui.activeSkillSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "-- 选择 skill --";
  ui.activeSkillSelect.appendChild(placeholder);

  for (const skill of options) {
    const option = document.createElement("option");
    option.value = skill;
    option.textContent = skill;
    ui.activeSkillSelect.appendChild(option);
  }
}

function renderSegments() {
  const video = currentVideo();
  if (!video) {
    ui.segmentRows.innerHTML = '<div class="segment-empty">暂无片段数据</div>';
    return;
  }

  ui.segmentRows.innerHTML = "";
  const currentSegIndex = findSegmentIndexByFrame(video, state.currentFrame);

  video.action_config.forEach((segment, index) => {
    const safeSegId = Number.isFinite(Number(segment?.seg_id)) ? Number(segment.seg_id) : index;
    const completeness = getSegmentCompleteness(segment);
    const actionText = String(segment.action_text || "").trim();
    const skillText = String(segment.skill || "").trim();
    const skillLabel = skillText || "未设 skill";
    const skillClasses = [
      "segment-inline-skill",
      !skillText ? "missing" : "",
      isSkillOutsideOptions(segment.skill) ? "outside" : "",
    ].filter(Boolean).join(" ");
    const row = document.createElement("div");
    row.className = "segment-row";
    row.dataset.segIdx = String(index);
    if (index === state.selectedSegIdx) row.classList.add("active");
    if (index === currentSegIndex) row.classList.add("current");
    if (!completeness.complete) row.classList.add("incomplete");
    if (completeness.missingAction) row.classList.add("missing-action");
    if (completeness.missingSkill) row.classList.add("missing-skill");
    if (isSkillOutsideOptions(segment.skill)) {
      row.classList.add("skill-outside-options");
    }
    const reviewSeverity = getSegmentReviewSeverity(video, segment, index);
    if (reviewSeverity === "incorrect") row.classList.add("segment-review-incorrect");
    if (reviewSeverity === "uncertain") row.classList.add("segment-review-uncertain");

    const titleHtml = actionText
      ? escapeHtml(actionText)
      : '<span class="segment-title-placeholder">未填写动作描述</span>';
    const actionTitle = escapeHtml(actionText || "未填写动作描述");
    const skillTitle = escapeHtml(skillLabel);
    const reviewBadgeHtml = renderSegmentReviewBadgeHtml(video, segment, index);

    row.innerHTML =
      `<div class="segment-color-bar" style="background:${segmentColor(safeSegId)}"></div>` +
      '<div class="segment-content">' +
      `<div class="segment-row-top"><span class="segment-inline-id" title="seg ${safeSegId}">seg ${safeSegId}</span><span class="${skillClasses}" title="${skillTitle}">${escapeHtml(truncate(skillLabel, 16))}</span>${reviewBadgeHtml}</div>` +
      `<div class="segment-title" title="${actionTitle}">${titleHtml}</div>` +
      "</div>" +
      '<div class="segment-side-actions">' +
      '<button type="button" class="icon-btn" data-role="jump" title="跳到起始帧">↦</button>' +
      `<button type="button" class="icon-btn ${local.deleteConfirmIdx === index ? "confirm-delete" : "danger"}" data-role="delete" title="${local.deleteConfirmIdx === index ? "点击确认删除" : "删除片段"}">${local.deleteConfirmIdx === index ? "确认?" : "✕"}</button>` +
      "</div>";

    row.addEventListener("click", event => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      const roleElement = target.closest("[data-role]");
      const role = roleElement ? roleElement.dataset.role : null;

      if (role === "jump") {
        event.stopPropagation();
        selectSegment(index, { jumpToStart: true });
        return;
      }

      if (role === "delete") {
        event.stopPropagation();
        if (local.deleteConfirmIdx === index) {
          window.clearTimeout(local.deleteConfirmTimer);
          local.deleteConfirmIdx = null;
          handleDeleteSegment(index);
        } else {
          window.clearTimeout(local.deleteConfirmTimer);
          local.deleteConfirmIdx = index;
          renderSegments();
          local.deleteConfirmTimer = window.setTimeout(() => {
            local.deleteConfirmIdx = null;
            renderSegments();
          }, 3000);
        }
        return;
      }

      selectSegment(index, { jumpToStart: true, focusEditor: true });
    });

    ui.segmentRows.appendChild(row);
  });
}

function renderSegmentsHighlight() {
  const video = currentVideo();
  if (!video) return;
  const currentSegIndex = findSegmentIndexByFrame(video, state.currentFrame);
  ui.segmentRows.querySelectorAll(".segment-row").forEach(row => {
    const idx = Number(row.dataset.segIdx);
    row.classList.toggle("active", idx === state.selectedSegIdx);
    row.classList.toggle("current", idx === currentSegIndex);
  });
}

function renderPlaybackChapters() {
  const video = currentVideo();
  if (!video) {
    ui.chapterBar.innerHTML = "";
    return;
  }

  ui.chapterBar.innerHTML = "";
  video.action_config.forEach((segment, index) => {
    const chapter = document.createElement("button");
    const reviewSeverity = getSegmentReviewSeverity(video, segment, index);
    chapter.type = "button";
    chapter.className = "chapter-seg";
    if (reviewSeverity === "incorrect") chapter.classList.add("review-incorrect");
    if (reviewSeverity === "uncertain") chapter.classList.add("review-uncertain");
    chapter.style.width = `${((segment.end_frame - segment.start_frame) / video.frames) * 100}%`;
    chapter.style.background = segmentColor(segment.seg_id);
    chapter.dataset.segId = String(segment.seg_id);
    const reviewTitle = reviewSeverity
      ? ` | 模型${reviewSeverity === "uncertain" ? "存疑" : "判错"}`
      : "";
    chapter.title = `seg ${segment.seg_id} [${segment.start_frame},${segment.end_frame}) | ${segment.skill || ""} | ${segment.action_text || ""}${reviewTitle}`;
    chapter.innerHTML = "";
    const label = document.createElement("span");
    label.className = "chapter-seg-label";
    label.textContent = `${segment.seg_id}: ${truncate(segment.action_text || "Untitled", 18)}`;
    chapter.appendChild(label);
    chapter.addEventListener("click", () => {
      selectSegment(index);
      setCurrentFrame(segment.start_frame, true);
    });
    ui.chapterBar.appendChild(chapter);
  });

  highlightActiveChapter();
}

function renderBoundaryMarkers() {
  const video = currentVideo();
  if (!video) {
    ui.boundaryMarkers.innerHTML = "";
    return;
  }

  const timelineWindow = getTimelineWindow(video);
  const startFrame = timelineWindow.startFrame;
  const endFrame = getTimelineWindowEnd(video, timelineWindow);
  ui.boundaryMarkers.innerHTML = "";

  video.action_config.forEach((segment, index) => {
    if (segment.start_frame <= 0 || segment.start_frame >= video.frames) return;
    if (segment.start_frame < startFrame || segment.start_frame > endFrame) return;

    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = "boundary-marker";
    marker.style.left = `${frameToTimelineRatio(video, timelineWindow, segment.start_frame) * 100}%`;
    marker.dataset.frame = String(segment.start_frame);
    marker.title = `跳到边界帧 ${segment.start_frame}`;
    marker.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      selectSegment(index);
      setCurrentFrame(segment.start_frame, true);
    });

    ui.boundaryMarkers.appendChild(marker);
  });

  highlightActiveBoundary();
}

function startBoundaryDrag(event, leftIndex) {
  event.preventDefault();
  event.stopPropagation();
  const video = currentVideo();
  if (!video) return;
  if (!canDragBoundary()) {
    showToast("请先暂停并等待所有视角对齐后再拖拽边界", "info", 1800);
    return;
  }

  local.dragBoundary = {
    leftIndex,
    token: createMutationToken("boundary_drag"),
    changed: false,
    lastBoundary: null,
  };

  const handleEl = event.currentTarget;
  if (handleEl instanceof HTMLElement) {
    handleEl.classList.add("dragging");
  }

  const onMove = moveEvent => {
    if (!local.dragBoundary) return;
    const frame = getFrameFromClientX(moveEvent.clientX);
    const result = moveBoundary(video, leftIndex, frame);
    if (result.ok) {
      local.dragBoundary.changed = true;
      local.dragBoundary.lastBoundary = result.boundary;
      if (local.dragRenderRaf !== null) return;
      local.dragRenderRaf = window.requestAnimationFrame(() => {
        local.dragRenderRaf = null;
        renderBoundaryMarkers();
        renderBoundaryHandles();
        renderFrameOnly();
      });
    }
  };

  const stop = () => {
    const drag = local.dragBoundary;
    local.dragBoundary = null;

    document.removeEventListener("pointermove", onMove);
    document.removeEventListener("pointerup", stop);

    if (handleEl instanceof HTMLElement) {
      handleEl.classList.remove("dragging");
    }

    if (local.dragRenderRaf !== null) {
      window.cancelAnimationFrame(local.dragRenderRaf);
      local.dragRenderRaf = null;
    }

    if (!drag || !drag.changed) return;

    if (finalizeMutation(drag.token)) {
      markDraftAndSchedule();
      renderUndoRedoButtons();
      renderStatus();
    }
    renderAfterMutation(false);
    if (Number.isFinite(drag.lastBoundary)) {
      showToast(`边界已更新到 f${drag.lastBoundary}`, "success", 1800);
    }
  };

  document.addEventListener("pointermove", onMove);
  document.addEventListener("pointerup", stop);
}

function renderBoundaryHandles() {
  const video = currentVideo();
  if (!video) {
    ui.boundaryHandles.innerHTML = "";
    return;
  }

  const timelineWindow = getTimelineWindow(video);
  const startFrame = timelineWindow.startFrame;
  const endFrame = getTimelineWindowEnd(video, timelineWindow);
  ui.boundaryHandles.innerHTML = "";

  video.action_config.forEach((segment, index) => {
    if (index >= video.action_config.length - 1) return;
    const boundaryFrame = segment.end_frame;
    if (boundaryFrame < startFrame || boundaryFrame > endFrame) return;

    const handle = document.createElement("button");
    handle.type = "button";
    handle.className = "boundary-handle";
    handle.dataset.boundaryIndex = String(index);
    handle.disabled = !canDragBoundary();
    handle.style.left = `${frameToTimelineRatio(video, timelineWindow, boundaryFrame) * 100}%`;
    handle.title = `拖拽边界 @f${boundaryFrame}`;
    handle.addEventListener("pointerdown", event => startBoundaryDrag(event, index));

    ui.boundaryHandles.appendChild(handle);
  });
}

function highlightActiveChapter() {
  const video = currentVideo();
  if (!video) return;
  const activeByFrame = findSegmentIndexByFrame(video, state.currentFrame);
  const selected = state.selectedSegIdx;
  ui.chapterBar.querySelectorAll(".chapter-seg").forEach((chapter, index) => {
    chapter.classList.toggle("active", index === activeByFrame || index === selected);
  });
}

function highlightActiveBoundary() {
  ui.boundaryMarkers.querySelectorAll(".boundary-marker").forEach(marker => {
    marker.classList.toggle("active", Number(marker.dataset.frame) === state.currentFrame);
  });
}

function truncate(text, length = 40) {
  if (!text) return "";
  return text.length <= length ? text : `${text.slice(0, length - 3)}...`;
}

function getFrameFromClientX(clientX) {
  const video = currentVideo();
  if (!video) return 0;
  const rect = ui.frameSlider.getBoundingClientRect();
  if (rect.width <= 0) return state.currentFrame;
  const ratio = (clamp(clientX, rect.left, rect.right) - rect.left) / rect.width;
  return ratioToTimelineFrame(video, getTimelineWindow(video), ratio);
}

function isTimelineBoundaryTarget(target) {
  return target instanceof Element && Boolean(target.closest(".boundary-handle, .boundary-marker"));
}

function startTimelineScrub(event) {
  if (event.button !== undefined && event.button !== 0) return;
  if (isTimelineBoundaryTarget(event.target)) return;
  if (isPlaybackDisabled() || local.isSyncSettling || local.isPlaybackRefreshing) return;

  event.preventDefault();
  local.isScrubbing = true;

  let lastFrame = null;
  const applyClientX = clientX => {
    const frame = getFrameFromClientX(clientX);
    if (frame === lastFrame) return;
    lastFrame = frame;
    setCurrentFrame(frame, true);
  };

  applyClientX(event.clientX);

  const onMove = moveEvent => {
    if (moveEvent.pointerId !== event.pointerId) return;
    moveEvent.preventDefault();
    applyClientX(moveEvent.clientX);
  };

  const stop = stopEvent => {
    if (stopEvent.pointerId !== event.pointerId) return;
    document.removeEventListener("pointermove", onMove);
    document.removeEventListener("pointerup", stop);
    document.removeEventListener("pointercancel", stop);
    if (ui.sliderWrap?.hasPointerCapture?.(event.pointerId)) {
      ui.sliderWrap.releasePointerCapture(event.pointerId);
    }
    local.isScrubbing = false;
  };

  ui.sliderWrap?.setPointerCapture?.(event.pointerId);
  document.addEventListener("pointermove", onMove);
  document.addEventListener("pointerup", stop);
  document.addEventListener("pointercancel", stop);
}

function getDriverFrameOrPlayhead() {
  const video = currentVideo();
  if (!video) return 0;
  const driver = getDriverPlayer();
  if (!driver) return state.currentFrame;
  return timeToFrame(video, driver.currentTime || 0);
}

function queueBoundaryMutation(label, run) {
  if (isPlaybackDisabled()) return;

  if (!local.isSyncSettling && !isDriverPlaying() && local.isFrameSettled) {
    run(local.settledFrame);
    return;
  }

  local.pendingAnnotationAction = { label, run };
  renderFrameOnly();
  updateInteractionAvailability();

  if (local.isSyncSettling) {
    showToast("对齐中，完成后执行…", "info", 1200);
    return;
  }

  if (local.isPlaybackRefreshing) {
    showToast("正在刷新最终帧，随后执行…", "info", 1400);
    return;
  }

  showToast("暂停并对齐后执行…", "info", 1400);
  const targetFrame = local.isPlaybackStable ? local.playbackStableFrame : getDriverFrameOrPlayhead();
  void requestFrameSettle(targetFrame, {
    failureMessage: "视角未完全对齐，未写入边界",
  });
}

function setCurrentFrame(frame, syncVideoTime) {
  const video = currentVideo();
  if (!video) return;

  const next = setPlayheadFrameValue(frame);
  ensureVisibleTimelineFrame(next);
  renderFrameOnly();

  if (syncVideoTime) {
    clearFrameSettled();
    clearPlaybackStable();
    updateInteractionAvailability();
    void requestFrameSettle(next, {
      failureMessage: "视角未完全对齐，跳转未完成",
      clearPendingActionOnFailure: false,
    });
  }
}

async function togglePlayPause() {
  if (isPlaybackDisabled() || local.isSyncSettling || local.isPlaybackRefreshing) return;

  const driver = getDriverPlayer();
  if (!driver) return;

  if (driver.paused) {
    const segment = shouldConstrainPlaybackToSelectedSegment() ? getSelectedSegment() : null;
    if (segment) {
      const startFrame = Number(segment.start_frame);
      const endFrame = getSegmentLastFrame(segment);
      if (state.currentFrame < startFrame || state.currentFrame >= endFrame) {
        setPlayheadFrameValue(startFrame);
      }
    }
    const ok = await requestFrameSettle(state.currentFrame, {
      failureMessage: "视角未完全对齐，无法开始播放",
      clearPendingActionOnFailure: false,
    });
    if (!ok) return;
    try {
      await playAllPlayers();
    } catch (error) {
      showToast(`播放失败: ${error.message}`, "error");
    }
  } else {
    local.segmentPlaybackActive = false;
    await finalizePlaybackStop(getDriverFrameOrPlayhead(), {
      failureMessage: "视角未稳定停在当前帧附近",
      clearPendingActionOnFailure: false,
    });
  }
}

async function handlePlaySegment() {
  if (isPlaybackDisabled() || local.isPlaybackRefreshing) return;
  const video = currentVideo();
  const segment = getSelectedSegment(video);
  if (!video || !segment) return;

  const startFrame = Number(segment.start_frame);
  const endFrame = getSegmentLastFrame(segment);
  const targetFrame = state.currentFrame < startFrame || state.currentFrame >= endFrame
    ? startFrame
    : state.currentFrame;

  local.segmentPlaybackActive = true;
  const ok = await requestFrameSettle(targetFrame, {
    failureMessage: "视角未完全对齐，无法开始片段播放",
    clearPendingActionOnFailure: false,
  });
  if (!ok) {
    local.segmentPlaybackActive = false;
    return;
  }
  try {
    await playAllPlayers();
  } catch (error) {
    showToast(`播放失败: ${error.message}`, "error");
  }
}

function selectSegment(index, options = {}) {
  const video = currentVideo();
  if (!video || !video.action_config.length) return;

  state.selectedSegIdx = clamp(index, 0, video.action_config.length - 1);
  renderSegments();
  renderActiveSegmentEditor();
  renderPlaybackChapters();
  renderBoundaryMarkers();
  renderBoundaryHandles();
  updateIncompleteNavButtons();
  syncPlaybackOptionControls();

  if (options.jumpToStart) {
    const segment = video.action_config[state.selectedSegIdx];
    if (segment) setCurrentFrame(segment.start_frame, true);
  }

  if (options.focusEditor) {
    window.setTimeout(() => {
      ui.activeActionInput.focus();
      ui.activeActionInput.setSelectionRange(ui.activeActionInput.value.length, ui.activeActionInput.value.length);
    }, 30);
  }

  const activeRow = ui.segmentRows.querySelector(`.segment-row[data-seg-idx="${state.selectedSegIdx}"]`);
  if (activeRow) {
    activeRow.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function renderAfterMutation(updateStatus = true) {
  const video = currentVideo();
  if (!video) return;

  const topology = validateSegmentTopology(video);
  if (!topology.ok) {
    showToast(`结构异常: ${topology.error}`, "error");
  }

  renderGeneralInfo();
  renderSegments();
  renderActiveSegmentEditor();
  renderPlaybackChapters();
  ensureVisibleTimelineFrame(state.currentFrame);
  renderBoundaryMarkers();
  renderBoundaryHandles();
  renderFrameOnly();
  renderUndoRedoButtons();
  updateIncompleteNavButtons();
  syncPlaybackOptionControls();
  updateInteractionAvailability();
  if (updateStatus) renderStatus();
}

function runMutation(type, mutate, onSuccess) {
  const video = currentVideo();
  if (!video) return;
  if (video.discarded) {
    showToast("该条已丢弃，请先恢复后再编辑分段", "info");
    return;
  }

  const token = createMutationToken(type);
  const result = mutate(video);
  if (!result || !result.ok) {
    if (result && result.error) showToast(result.error, "error");
    return;
  }

  if (finalizeMutation(token)) {
    markDraftAndSchedule();
  }
  renderAfterMutation();
  if (typeof onSuccess === "function") onSuccess(result);
}

function handleSplitSegment() {
  if (isPlaybackDisabled()) return;
  queueBoundaryMutation("split", frame => {
    runMutation("split", video => splitSegmentAtFrame(video, frame), result => {
      if (typeof result.newIndex === "number") {
        state.selectedSegIdx = result.newIndex;
        renderAfterMutation();
        ui.activeActionInput.focus();
        showToast(`在 f${frame} 切分成功`, "success");
      }
    });
  });
}

function handleDeleteSegment(index) {
  runMutation("delete", video => deleteSegmentAtIndex(video, index), () => {
    const video = currentVideo();
    if (video) {
      state.selectedSegIdx = clamp(index, 0, video.action_config.length - 1);
      showToast("片段已删除，可在 3 秒内按 Ctrl+Z 回退", "info");
    }
  });
}

function handleSetStart() {
  if (isPlaybackDisabled()) return;
  const index = state.selectedSegIdx;
  queueBoundaryMutation("set_start", frame => {
    runMutation("set_start", video => setSegmentStart(video, index, frame), result => {
      const segment = getSelectedSegment();
      showToast(`seg ${segment?.seg_id ?? index} 起始帧更新为 f${result.boundary}`, "success");
    });
  });
}

function handleSetEnd() {
  if (isPlaybackDisabled()) return;
  const index = state.selectedSegIdx;
  queueBoundaryMutation("set_end", frame => {
    runMutation("set_end", video => setSegmentEnd(video, index, frame), result => {
      const segment = getSelectedSegment();
      showToast(`seg ${segment?.seg_id ?? index} 结束帧更新为 f${result.boundary}`, "success");
    });
  });
}

function jumpToSegmentStart() {
  const video = currentVideo();
  if (!video || isPlaybackDisabled() || local.isSyncSettling || local.isPlaybackRefreshing) return;
  const segment = video.action_config[state.selectedSegIdx];
  if (!segment) return;
  setCurrentFrame(segment.start_frame, true);
}

function jumpToSegmentEnd() {
  const video = currentVideo();
  if (!video || isPlaybackDisabled() || local.isSyncSettling || local.isPlaybackRefreshing) return;
  const segment = video.action_config[state.selectedSegIdx];
  if (!segment) return;
  const endFrame = Math.max(segment.start_frame, segment.end_frame - 1);
  setCurrentFrame(endFrame, true);
}

function flushEditorMutationToken() {
  if (!local.actionEditToken) return;
  if (finalizeMutation(local.actionEditToken)) {
    markDraftAndSchedule();
  }
  local.actionEditToken = null;
  renderUndoRedoButtons();
  renderStatus();
}

function flushTaskMutationToken() {
  if (!local.taskEditToken) return;
  if (finalizeMutation(local.taskEditToken)) {
    markDraftAndSchedule();
  }
  local.taskEditToken = null;
  renderUndoRedoButtons();
  renderStatus();
}

function bindEditorInputs() {
  ui.activeActionInput.addEventListener("focus", () => {
    if (!currentVideo()) return;
    if (!local.actionEditToken) {
      local.actionEditToken = createMutationToken("action_edit");
    }
  });

  ui.activeActionInput.addEventListener("input", () => {
    const video = currentVideo();
    if (!video) return;
    const selectedIndex = state.selectedSegIdx;
    const segment = video.action_config[selectedIndex];
    if (!segment) return;

    segment.action_text = ui.activeActionInput.value;
    markDraftAndSchedule();
    updateSegmentRow(selectedIndex);
    renderActiveSegmentValidationState(segment);
  });

  ui.activeActionInput.addEventListener("blur", () => {
    flushEditorMutationToken();
  });

  ui.activeActionInput.addEventListener("keydown", event => {
    if (event.key === "Tab" && !event.shiftKey) {
      event.preventDefault();
      ui.activeSkillSelect.focus();
    }
  });

  ui.actionTextFilterInput.addEventListener("input", () => {
    const video = currentVideo();
    const segment = video ? video.action_config[state.selectedSegIdx] : null;
    const selectedActionText = segment?.action_text || "";
    renderActionSelectOptions(ui.actionTextFilterInput.value, selectedActionText);
    if (segment) {
      ui.activeActionSelect.value = selectedActionText;
    }
  });

  ui.activeActionSelect.addEventListener("change", () => {
    flushEditorMutationToken();
    runMutation("action_select", video => {
      const segment = video.action_config[state.selectedSegIdx];
      if (!segment) return { ok: false, error: "no segment selected" };
      segment.action_text = ui.activeActionSelect.value;
      return { ok: true };
    });
  });

  ui.skillFilterInput.addEventListener("input", () => {
    const video = currentVideo();
    const segment = video ? video.action_config[state.selectedSegIdx] : null;
    renderSkillSelectOptions(ui.skillFilterInput.value, segment?.skill || "");
    if (segment) {
      ui.activeSkillSelect.value = segment.skill || "";
    }
  });

  ui.activeSkillSelect.addEventListener("change", () => {
    runMutation("skill_edit", video => {
      const segment = video.action_config[state.selectedSegIdx];
      if (!segment) return { ok: false, error: "no segment selected" };
      segment.skill = ui.activeSkillSelect.value;
      return { ok: true };
    });
  });

  ui.activeSkillSelect.addEventListener("keydown", event => {
    if (event.key === "Tab" && !event.shiftKey) {
      event.preventDefault();
      const video = currentVideo();
      if (!video) return;
      const next = Math.min(video.action_config.length - 1, state.selectedSegIdx + 1);
      selectSegment(next, { jumpToStart: false, focusEditor: false });
      window.setTimeout(() => {
        ui.activeActionInput.focus();
        ui.activeActionInput.select();
      }, 30);
    } else if (event.key === "Tab" && event.shiftKey) {
      event.preventDefault();
      ui.activeActionInput.focus();
      ui.activeActionInput.setSelectionRange(ui.activeActionInput.value.length, ui.activeActionInput.value.length);
    }
  });

  ui.tasksInput.addEventListener("focus", () => {
    if (!currentVideo()) return;
    if (!local.taskEditToken) {
      local.taskEditToken = createMutationToken("task_edit");
    }
  });

  ui.tasksInput.addEventListener("input", () => {
    const video = currentVideo();
    if (!video) return;
    video.tasks = ui.tasksInput.value.split("\n");
    markDraftAndSchedule();
  });

  ui.tasksInput.addEventListener("blur", () => {
    flushTaskMutationToken();
  });
}

function bindInfoPanelToggle() {
  if (!ui.infoPanelToggle || !ui.infoPanel) return;

  ui.infoPanelToggle.addEventListener("click", () => {
    const visible = ui.infoPanel.classList.contains("hidden");
    setInfoPanelVisible(visible);
  });
}

function bindAdvancedToggle() {
  if (!ui.advancedToggleBtn || !ui.advancedBar) return;
  syncAdvancedToggle();
  ui.advancedToggleBtn.addEventListener("click", () => {
    ui.advancedBar.classList.toggle("hidden");
    syncAdvancedToggle();
  });
}

function bindPanelResizeHandle() {
  if (!ui.panelResizeHandle) return;

  const startResize = startX => {
    if (window.innerWidth <= 860) return;
    const initialWidth = local.annotationPanelWidth;
    ui.panelResizeHandle.classList.add("dragging");

    const onMove = moveEvent => {
      const delta = moveEvent.clientX - startX;
      applyAnnotationPanelWidth(initialWidth - delta);
    };

    const stop = () => {
      ui.panelResizeHandle.classList.remove("dragging");
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", stop);
    };

    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", stop);
  };

  ui.panelResizeHandle.addEventListener("pointerdown", event => {
    event.preventDefault();
    startResize(event.clientX);
  });

  ui.panelResizeHandle.addEventListener("keydown", event => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      applyAnnotationPanelWidth(local.annotationPanelWidth + 24);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      applyAnnotationPanelWidth(local.annotationPanelWidth - 24);
    }
  });
}

async function commitCurrentVideoEdits(options = {}) {
  const { silent = false } = options;
  const video = currentVideo();
  if (!video) return true;
  if (video.discarded) {
    if (!silent) showToast("该条已丢弃，无需应用分段", "info");
    return true;
  }

  flushEditorMutationToken();
  flushTaskMutationToken();

  const topology = validateSegmentTopology(video);
  if (!topology.ok) {
    if (!silent) showToast(`提交失败: ${topology.error}`, "error");
    return false;
  }

  const taskLines = sanitizeTasks(ui.tasksInput.value.split("\n"));
  if (!taskLines.length) {
    ui.tasksInput.focus();
    if (!silent) showToast("Task List 不能为空", "error");
    return false;
  }
  const inputTaskCount = ui.tasksInput.value
    .split("\n")
    .map(line => line.trim())
    .filter(Boolean).length;
  if (inputTaskCount !== taskLines.length && !silent) {
    showToast("Task List 已自动去重并清理空行", "info");
  }
  video.tasks = taskLines;
  ui.tasksInput.value = taskLines.join("\n");

  const stats = getVideoStats(video);
  if (stats.incompleteCount > 0) {
    goToIncompleteSegment(1);
    if (!silent) {
      showToast(`提交失败: 仍有 ${stats.incompleteCount} 个片段未填写 action/skill`, "error");
    }
    return false;
  }

  const payload = {
    tasks: taskLines,
    action_config: video.action_config.map(segment => ({
      seg_id: segment.seg_id,
      start_frame: segment.start_frame,
      end_frame: segment.end_frame,
      action_text: String(segment.action_text || "").trim(),
      skill: String(segment.skill || "").trim(),
    })),
    _audit: buildAuditContext(),
  };

  try {
    const commitIndex = state.currentIndex;
    const response = await updateAnnotation(commitIndex, payload);
    invalidatePrefetchIndex(commitIndex);
    if (Array.isArray(response?.skill_options)) {
      state.skillOptions = response.skill_options.slice();
    }
    if (Array.isArray(response?.action_text_options)) {
      state.actionTextOptions = response.action_text_options.slice();
    }
    markCommitted();
    renderStatus();
    persistDraftNow();
    setItemError("");
    commitMetaCounts(payload.tasks.length, payload.action_config.length);
    if (response?.annotation_status) {
      setAnnotationStatus(commitIndex, response.annotation_status);
    }
    renderMetaSelect();
    renderVideoPosition();
    updateIncompleteNavButtons();
    if (!silent) showToast("当前条已应用，尚未保存到文件", "success");
    return true;
  } catch (error) {
    if (isAnnotatorSelectionRequired(error)) {
      await handleAnnotatorSelectionRequired("请选择标注员后继续提交当前条。");
      return false;
    }
    setValidationError(error);
    focusFirstValidationTarget(error);
    if (!silent) showToast(`提交失败: ${error.message}`, "error");
    return false;
  }
}

async function ensureSwitchable() {
  if (!currentVideo() || !state.dirty) return true;
  const ok = await commitCurrentVideoEdits({ silent: true });
  if (ok) return true;
  persistDraftNow();
  showToast("当前条未通过校验，仅保存为本地草稿", "info", 3600);
  return true;
}

async function handleJumpToVideo() {
  if (!ui.videoJumpInput || !state.meta.length) return;
  if (local.isSaving) {
    showToast("正在保存，请稍候", "info");
    renderVideoPosition();
    return;
  }

  const raw = ui.videoJumpInput.value.trim();
  if (!/^\d+$/.test(raw)) {
    showToast("请输入有效编号", "info");
    renderVideoPosition();
    return;
  }

  const targetNumber = Number.parseInt(raw, 10);
  if (targetNumber < 1 || targetNumber > state.meta.length) {
    showToast(`编号超出范围: 1-${state.meta.length}`, "info");
    renderVideoPosition();
    return;
  }

  const targetIndex = targetNumber - 1;
  if (targetIndex === state.currentIndex) {
    renderVideoPosition();
    return;
  }

  if (!await ensureSwitchable()) {
    renderVideoPosition();
    return;
  }

  await loadVideoByIndex(targetIndex);
}

function clearCurrentView(message = "无视频") {
  trackCurrentItemClose("clear");
  cancelFrameSettle({ keepSettled: false, clearPendingAction: true });
  cancelPlaybackRefresh({ keepStable: false, clearPendingAction: true });
  stopPlaybackSyncLoop();
  stopPlaybackBoundaryWatch();
  state.currentIndex = -1;
  state.serverSnapshot = null;
  state.workingCopy = null;
  state.currentFrame = 0;
  state.selectedSegIdx = 0;
  state.statusMode = state.hasUnsavedFileChanges ? "committed" : "saved";
  state.dirty = false;
  if (!state.hasUnsavedFileChanges) {
    state.lastAppliedAt = null;
    state.lastSavedAt = null;
  }
  state.isLoadingVideo = false;
  state.timelineWindow = null;
  state.undoStack = [];
  state.redoStack = [];
  local.segmentPlaybackActive = false;
  local.playheadFrame = 0;
  local.playbackStableFrame = 0;
  local.settledFrame = 0;

  ui.videoGrid.innerHTML = "";
  applyVideoGridLayout(1);
  local.viewPlayers = new Map();
  local.viewPlayback = new Map();
  local.driverViewKey = "";
  local.activeViewKey = "";
  local.hasPlayableViews = false;
  if (ui.activeViewBadge) ui.activeViewBadge.textContent = "view -";
  ui.frameSlider.value = "0";
  ui.frameSlider.min = "0";
  ui.frameSlider.max = "0";

  ui.chapterBar.innerHTML = "";
  ui.boundaryMarkers.innerHTML = "";
  ui.boundaryHandles.innerHTML = "";
  ui.segmentRows.innerHTML = '<div class="segment-empty">暂无片段数据</div>';

  ui.activeEditorEmpty.classList.remove("hidden");
  ui.activeEditorBody.classList.add("hidden");
  hideModelReviewPanel();
  ui.segmentBadge.textContent = "seg -";
  ui.activeSegmentMeta.textContent = "seg -";
  ui.activeSegmentRange.textContent = "f - → -";
  ui.selectionHint.textContent = syncAwareSelectionHint("选择片段");

  ui.tasksInput.value = "";
  ui.actionTextFilterInput.value = "";
  ui.activeActionSelect.innerHTML = "";
  ui.skillFilterInput.value = "";
  ui.activeSkillSelect.innerHTML = "";

  setControlAvailability(false);
  renderGeneralInfo();
  renderFrameLabel();
  updateNeedlePosition();
  updateIncompleteNavButtons();
  syncPlaybackOptionControls();
  if (message) {
    showVideoError(message);
  } else {
    hideVideoError();
  }
  updateInteractionAvailability();
  renderStatus();
  renderUndoRedoButtons();
  renderVideoPosition();
}

// ─── Prefetch helpers ────────────────────────────────────────────────────────

const PREFETCH_MAX_CACHE = 4;

function prefetchEvict() {
  // Keep only the entries closest to the current index; evict the rest.
  if (local.prefetchCache.size <= PREFETCH_MAX_CACHE) return;
  const current = state.currentIndex;
  const sorted = [...local.prefetchCache.keys()].sort(
    (a, b) => Math.abs(a - current) - Math.abs(b - current),
  );
  for (let i = PREFETCH_MAX_CACHE; i < sorted.length; i++) {
    local.prefetchCache.delete(sorted[i]);
  }
}

function invalidatePrefetchIndex(index) {
  if (!Number.isInteger(index)) return;
  const cached = local.prefetchCache.get(index);
  cached?.controller?.abort();
  local.prefetchCache.delete(index);
}

function clearPrefetchCache() {
  window.clearTimeout(local.prefetchTimer);
  local.prefetchTimer = null;
  local.prefetchCache.forEach(entry => {
    entry?.controller?.abort();
  });
  local.prefetchCache.clear();
}

async function prefetchVideoIndex(index) {
  if (index < 0 || index >= state.meta.length) return;
  if (local.prefetchCache.has(index)) return; // already cached or in-flight

  const controller = new AbortController();
  const entry = { payload: null, error: null, loading: true, controller };
  local.prefetchCache.set(index, entry);

  try {
    const payload = await getVideo(index, { signal: controller.signal });
    entry.payload = payload;
    entry.loading = false;

    // Also pre-load video metadata for each view so the browser has the
    // duration/dimensions ready before the user navigates to this index.
    const views = Array.isArray(payload.views) && payload.views.length
      ? payload.views
      : [{ key: String(payload.primary_view_key || "main"), path: String(payload.path || ""), unavailable: Boolean(payload.unavailable) }];
    for (const view of views.slice(0, 4)) {
      if (view.unavailable) continue;
      const viewKey = String(view?.key || "main");
      const playback = view && typeof view.playback === "object" ? view.playback : {};
      const streamUrl = makeViewStreamUrl(index, viewKey);
      const primaryUrl = String(playback.primary_url || streamUrl);
      const probe = document.createElement("video");
      probe.preload = "metadata";
      probe.muted = true;
      probe.src = primaryUrl;
      probe.load();
      // Detach after metadata loaded or on error to free the element.
      const cleanup = () => { probe.src = ""; probe.load(); };
      probe.addEventListener("loadedmetadata", cleanup, { once: true });
      probe.addEventListener("error", cleanup, { once: true });
    }
  } catch (err) {
    if (err?.name === "AbortError") {
      local.prefetchCache.delete(index);
      return;
    }
    entry.error = err;
    entry.loading = false;
  }

  prefetchEvict();
}

function schedulePrefetch(currentIndex) {
  window.clearTimeout(local.prefetchTimer);
  local.prefetchTimer = window.setTimeout(() => {
    local.prefetchTimer = null;
    // Prefetch next and previous videos with a small delay so the current
    // video's own network requests get priority.
    prefetchVideoIndex(currentIndex + 1);
    prefetchVideoIndex(currentIndex - 1);
  }, 800);
}

// ─── Video loader ─────────────────────────────────────────────────────────────

async function loadVideoByIndex(index) {
  if (index < 0 || index >= state.meta.length) return;
  if (local.auditActiveIndex >= 0 && local.auditActiveIndex !== index) {
    trackCurrentItemClose("switch");
  }
  cancelFrameSettle({ keepSettled: false, clearPendingAction: true });
  cancelPlaybackRefresh({ keepStable: false, clearPendingAction: true });
  stopPlaybackSyncLoop();
  stopPlaybackBoundaryWatch();
  if (local.videoRequestController) {
    local.videoRequestController.abort();
  }
  const requestId = state.activeVideoRequestId + 1;
  state.activeVideoRequestId = requestId;
  state.isLoadingVideo = true;
  const controller = new AbortController();
  local.videoRequestController = controller;
  setControlAvailability(false);
  showVideoError("正在加载视频 / 视角 / 播放配置");
  renderStatus();

  try {
    // Use prefetch cache when available to skip the network round-trip.
    let payload;
    const cached = local.prefetchCache.get(index);
    if (cached && !cached.loading && cached.payload) {
      payload = cached.payload;
      local.prefetchCache.delete(index);
    } else {
      // If a prefetch is in-flight for this index, abort it and fetch fresh
      // with our own controller so we can cancel if the user navigates again.
      if (cached?.loading) {
        cached.controller?.abort();
        local.prefetchCache.delete(index);
      }
      payload = await getVideo(index, { signal: controller.signal });
    }
    if (controller.signal.aborted || requestId !== state.activeVideoRequestId) return;
    setVideo(index, payload);
    if (payload?.annotation_status) {
      setAnnotationStatus(index, payload.annotation_status);
    }
    if (local.auditActiveIndex !== index) {
      trackItemOpen(index);
    }

    const draft = loadDraft(getDraftKey(payload));
    if (draft && draft.workingCopy && draft.serverHash === snapshotHash(state.serverSnapshot)) {
      draft.workingCopy.model_review = payload.model_review || null;
      draft.workingCopy.app_mode = payload.app_mode || state.appMode;
      applyWorkingCopyFromDraft(draft.workingCopy);
      showToast("已恢复本地草稿", "info");
    }

    const video = currentVideo();
    if (!video) return;

    state.timelineWindow = createTimelineWindow(video, getTimelinePreset(), 0);
    ui.videoSelect.value = String(index);
    syncTimelineSliderWindow(video);

    const views = Array.isArray(payload.views) && payload.views.length
      ? payload.views
      : [{
        key: String(payload.primary_view_key || "main"),
        path: String(payload.path || ""),
        unavailable: Boolean(payload.unavailable),
        error: String(payload.error || ""),
        playback: payload && typeof payload.playback === "object" ? payload.playback : {},
      }];
    const primaryViewKey = String(payload.primary_view_key || views[0]?.key || "main");
    renderVideoTiles(views, index, primaryViewKey);
    setPlayheadFrameValue(0);
    clearPlaybackStable();
    clearFrameSettled();

    renderGeneralInfo();
    renderSegments();
    renderActiveSegmentEditor();
    renderPlaybackChapters();
    renderBoundaryMarkers();
    renderBoundaryHandles();
    renderFrameOnly();
    renderVideoPosition();
    renderStatus();
    renderUndoRedoButtons();

    if (video.unavailable || !local.hasPlayableViews) {
      setControlAvailability(false);
      showVideoError(video.error || "所有视角不可播放");
    } else {
      setControlAvailability(true);
      updateInteractionAvailability();
      if (video.error) {
        showVideoError(video.error);
      } else {
        hideVideoError();
      }
      void requestFrameSettle(0, {
        failureMessage: "初始帧未完全对齐",
        clearPendingActionOnFailure: false,
      });
    }

    // Schedule background prefetch of adjacent videos after current one loads.
    schedulePrefetch(index);
  } catch (error) {
    if (isAbortError(error) || requestId !== state.activeVideoRequestId) return;
    if (isAnnotatorSelectionRequired(error)) {
      await handleAnnotatorSelectionRequired("请选择标注员后继续加载视频。");
      return;
    }
    showToast(`加载视频失败: ${error.message}`, "error");
    showVideoError(`加载失败: ${error.message}`);
  } finally {
    if (local.videoRequestController === controller) {
      local.videoRequestController = null;
    }
    if (requestId === state.activeVideoRequestId) {
      state.isLoadingVideo = false;
      renderStatus();
    }
  }
}

async function handleAnnotatorSelection(annotatorId) {
  const safeAnnotatorId = String(annotatorId || "").trim();
  if (!safeAnnotatorId || local.isSelectingAnnotator || local.isSaving) return false;
  if (safeAnnotatorId === state.collab.annotatorId && state.collab.annotatorId) {
    hideAnnotatorModal();
    return true;
  }

  local.isSelectingAnnotator = true;
  renderAnnotatorOptions();
  if (ui.annotatorModalCloseBtn) ui.annotatorModalCloseBtn.disabled = true;
  try {
    await selectAnnotator(safeAnnotatorId);
    const collabPayload = await refreshCollabConfig({ silent: true });
    if (!collabPayload || !state.collab.annotatorId) {
      showToast("切换标注员失败：未获取到有效的标注员信息", "error");
      showAnnotatorModal("请选择有效的标注员。");
      return false;
    }

    resetLoadedDataset("正在加载标注数据");
    hideAnnotatorModal();
    const ok = await fetchAndRenderMeta();
    if (!ok) {
      showAnnotatorModal("选择标注员后加载数据失败，请重试。");
      return false;
    }
    if (state.meta.length > 0) {
      await loadVideoByIndex(0);
    } else {
      clearCurrentView("无视频");
    }
    showToast(`已切换到 ${state.collab.annotatorName || state.collab.annotatorId}`, "success");
    return true;
  } catch (error) {
    showToast(`选择标注员失败: ${error.message}`, "error");
    showAnnotatorModal("请选择有效的标注员。");
    return false;
  } finally {
    local.isSelectingAnnotator = false;
    renderAnnotatorOptions();
    if (ui.annotatorModalCloseBtn) ui.annotatorModalCloseBtn.disabled = false;
  }
}

async function ensureCollaborationReady() {
  const collabPayload = await refreshCollabConfig();
  if (!collabPayload) return false;
  if (!state.collab.enabled) {
    hideAnnotatorModal();
    return true;
  }
  if (!state.collab.annotatorId) {
    resetLoadedDataset("请选择标注员以开始标注");
    showAnnotatorModal("请选择你的标注身份后开始标注。");
    return false;
  }
  hideAnnotatorModal();
  return true;
}

async function fetchAndRenderMeta() {
  if (local.metaRequestController) {
    local.metaRequestController.abort();
  }
  const requestId = local.metaRequestId + 1;
  local.metaRequestId = requestId;
  const controller = new AbortController();
  local.metaRequestController = controller;
  try {
    const payload = await getMeta({ signal: controller.signal });
    if (controller.signal.aborted || requestId !== local.metaRequestId) return false;
    if (state.collab.enabled) {
      state.collab.annotatorId = String(payload.annotator_id || state.collab.annotatorId || "");
      state.collab.annotatorName = String(payload.annotator_name || state.collab.annotatorName || "");
      state.collab.annotatorRole = String(payload.annotator_role || state.collab.annotatorRole || "");
      renderAnnotatorState();
    }
    setMeta(
      payload.videos || [],
      payload.skill_options || [],
      payload.action_text_options || [],
      payload.input_json,
      payload.output_json,
      payload.app_mode,
      payload.model_review_json,
      payload.annotation_statuses,
    );
    setPathInfo(payload.input_json, payload.output_json);

    const loadWarnings = Array.isArray(payload.load_warnings) ? payload.load_warnings : [];
    if (loadWarnings.length > 0) {
      showToast(`加载告警: 跳过 ${loadWarnings.length} 条不可编辑记录`, "info", 4500);
      console.warn("annotation_ui load warnings:", loadWarnings);
    }

    renderMetaSelect();
    renderVideoPosition();
    if (!state.meta.length) {
      clearCurrentView("无视频");
    }
    return true;
  } catch (error) {
    if (isAbortError(error) || requestId !== local.metaRequestId) return false;
    if (isAnnotatorSelectionRequired(error)) {
      await handleAnnotatorSelectionRequired("请选择标注员后开始标注。");
      return false;
    }
    showToast(`加载元数据失败: ${error.message}`, "error");
    return false;
  } finally {
    if (local.metaRequestController === controller) {
      local.metaRequestController = null;
    }
  }
}

async function switchVideo(direction) {
  if (local.isSaving) {
    showToast("正在保存，请稍候", "info");
    return;
  }

  if (!await ensureSwitchable()) return;

  const next = state.currentIndex + direction;
  if (next < 0 || next >= state.meta.length) {
    showToast("已是第一/最后一个视频", "info");
    return;
  }

  await loadVideoByIndex(next);
}

async function handleSwitchInput() {
  if (local.isSaving) {
    showToast("正在保存，请稍候", "info");
    return;
  }

  if (state.collab.enabled) {
    showToast("多人模式下不允许切换输入源", "info");
    return;
  }
  const inputPath = ui.inputJsonField.value.trim();
  if (!inputPath) {
    showToast("请输入 JSON 路径", "error");
    return;
  }

  if (!await ensureSwitchable()) return;

  if (local.switchInputController) {
    local.switchInputController.abort();
  }
  const requestId = local.switchInputRequestId + 1;
  local.switchInputRequestId = requestId;
  const controller = new AbortController();
  local.switchInputController = controller;
  state.isLoadingVideo = true;
  renderStatus();
  setControlAvailability(false);
  showVideoError("正在切换输入源");

  try {
    await switchInput(inputPath, state.outputJson, { signal: controller.signal });
    if (controller.signal.aborted || requestId !== local.switchInputRequestId) return;
    const ok = await fetchAndRenderMeta();
    if (ok && state.meta.length > 0) {
      await loadVideoByIndex(0);
    } else if (ok) {
      clearCurrentView("无视频");
    }
  } catch (error) {
    if (isAbortError(error) || requestId !== local.switchInputRequestId) return;
    showToast(`切换失败: ${error.message}`, "error");
    showVideoError(`切换失败: ${error.message}`);
  } finally {
    if (local.switchInputController === controller) {
      local.switchInputController = null;
    }
    if (
      requestId === local.switchInputRequestId
      && !local.videoRequestController
      && !local.metaRequestController
    ) {
      state.isLoadingVideo = false;
      renderStatus();
    }
  }
}

async function handleSave() {
  if (local.isSaving) return;
  if (state.isLoadingVideo) {
    showToast("正在加载视频，请稍候", "info");
    return;
  }

  const saveIndex = state.currentIndex;
  const video = currentVideo();
  const draftKey = video ? getDraftKey(video) : "";
  local.isSaving = true;
  updateWorkflowControls();
  renderStatus();

  try {
    const committed = await commitCurrentVideoEdits({ silent: true });
    if (!committed) {
      showToast("保存已中止：当前条应用失败", "error");
      return;
    }

    invalidatePrefetchIndex(saveIndex);
    const result = await saveAll({ _audit: buildAuditContext() });
    invalidatePrefetchIndex(saveIndex);
    markSaved();
    if (Array.isArray(result?.annotation_statuses)) {
      setAnnotationStatuses(result.annotation_statuses);
      renderMetaSelect();
    }
    renderStatus();
    if (draftKey) {
      clearDraftByKey(draftKey);
    } else {
      clearCurrentDraft();
    }
    showToast(`已保存到 ${result.saved_path || "输出文件"}`, "success", 4000);
  } catch (error) {
    if (isAnnotatorSelectionRequired(error)) {
      await handleAnnotatorSelectionRequired("请选择标注员后继续保存。");
      return;
    }
    setValidationError(error);
    showToast(`保存失败: ${error.message}`, "error");
  } finally {
    local.isSaving = false;
    updateWorkflowControls();
    renderStatus();
  }
}

function handleUndo() {
  if (isCurrentDiscarded()) return;
  flushEditorMutationToken();
  flushTaskMutationToken();
  if (!undoMutation()) {
    showToast("没有可撤销操作", "info");
    return;
  }
  markDraftAndSchedule();
  renderAfterMutation();
}

function handleRedo() {
  if (isCurrentDiscarded()) return;
  flushEditorMutationToken();
  flushTaskMutationToken();
  if (!redoMutation()) {
    showToast("没有可重做操作", "info");
    return;
  }
  markDraftAndSchedule();
  renderAfterMutation();
}

function applyDiscardResponse(response) {
  const video = currentVideo();
  if (!video) return;
  const fields = ["discarded", "discard_reason", "discard_note", "discarded_by", "discarded_at"];
  fields.forEach(field => {
    if (field === "discarded" || response?.[field]) video[field] = response?.[field] ?? false;
    else delete video[field];
  });
  state.serverSnapshot = cloneDeep(video);
  const meta = state.meta[state.currentIndex];
  if (meta) {
    meta.discarded = Boolean(video.discarded);
    meta.discard_reason = String(video.discard_reason || "");
  }
  if (response?.annotation_status) setAnnotationStatus(state.currentIndex, response.annotation_status);
  clearCurrentDraft();
  markSaved();
  renderItemStatus();
  renderMetaSelect();
  renderSegments();
  renderActiveSegmentEditor();
  renderStatus();
  renderUndoRedoButtons();
  updateInteractionAvailability();
}

async function restoreCurrentItem() {
  if (local.isUpdatingDiscard || !isCurrentDiscarded()) return;
  local.isUpdatingDiscard = true;
  renderItemStatus();
  try {
    const index = state.currentIndex;
    const response = await updateDiscard(index, {
      discarded: false,
      _audit: buildAuditContext(),
    });
    invalidatePrefetchIndex(index);
    applyDiscardResponse(response);
    showToast("已恢复此条并保存", "success");
  } catch (error) {
    showToast(`恢复失败: ${error.message}`, "error");
  } finally {
    local.isUpdatingDiscard = false;
    renderItemStatus();
  }
}

async function discardCurrentItemDirectly() {
  if (local.isUpdatingDiscard || !currentVideo() || isCurrentDiscarded()) return;
  local.isUpdatingDiscard = true;
  renderItemStatus();
  try {
    const index = state.currentIndex;
    const response = await updateDiscard(index, {
      discarded: true,
      _audit: buildAuditContext(),
    });
    invalidatePrefetchIndex(index);
    applyDiscardResponse(response);
    if (index < state.meta.length - 1) {
      showToast("已丢弃并保存，正在进入下一条", "success");
      await loadVideoByIndex(index + 1);
    } else {
      showToast("已丢弃并保存（当前为最后一条）", "success");
    }
  } catch (error) {
    showToast(`丢弃失败: ${error.message}`, "error");
  } finally {
    local.isUpdatingDiscard = false;
    renderItemStatus();
  }
}

function bindEvents() {
  ui.prevVideoBtn.addEventListener("click", () => switchVideo(-1));
  ui.nextVideoBtn.addEventListener("click", () => switchVideo(1));
  ui.prevIncompleteBtn.addEventListener("click", () => goToIncompleteSegment(-1));
  ui.nextIncompleteBtn.addEventListener("click", () => goToIncompleteSegment(1));

  ui.videoSelect.addEventListener("change", async () => {
    const index = Number.parseInt(ui.videoSelect.value, 10);
    if (Number.isNaN(index) || index === state.currentIndex) return;
    if (local.isSaving) {
      ui.videoSelect.value = String(state.currentIndex);
      showToast("正在保存，请稍候", "info");
      return;
    }
    if (!await ensureSwitchable()) {
      ui.videoSelect.value = String(state.currentIndex);
      return;
    }
    await loadVideoByIndex(index);
  });
  ui.videoJumpInput.addEventListener("keydown", async event => {
    if (event.key === "Enter") {
      event.preventDefault();
      await handleJumpToVideo();
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      renderVideoPosition();
      ui.videoJumpInput.blur();
    }
  });
  ui.videoJumpInput.addEventListener("blur", () => {
    renderVideoPosition();
  });

  ui.loadJsonBtn.addEventListener("click", handleSwitchInput);
  ui.inputJsonField.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSwitchInput();
    }
  });
  ui.switchAnnotatorBtn?.addEventListener("click", async () => {
    if (!state.collab.enabled) return;
    if (local.isSaving) {
      showToast("正在保存，请稍候", "info");
      return;
    }
    if (!await ensureSwitchable()) return;
    showAnnotatorModal("切换标注员后将加载该标注员负责的任务分片。");
  });
  ui.annotatorModalCloseBtn?.addEventListener("click", () => {
    hideAnnotatorModal();
  });
  ui.discardItemBtn?.addEventListener("click", () => void discardCurrentItemDirectly());
  ui.restoreItemBtn?.addEventListener("click", () => void restoreCurrentItem());
  ui.undoBtn.addEventListener("click", handleUndo);
  ui.redoBtn.addEventListener("click", handleRedo);
  ui.applyBtn.addEventListener("click", () => commitCurrentVideoEdits());
  ui.saveBtn.addEventListener("click", handleSave);

  ui.jumpBack10Btn.addEventListener("click", () => {
    if (isPlaybackDisabled()) return;
    setCurrentFrame(state.currentFrame - 10, true);
  });
  ui.playPauseBtn.addEventListener("click", () => togglePlayPause());
  ui.prevFrameBtn.addEventListener("click", () => {
    if (isPlaybackDisabled()) return;
    setCurrentFrame(state.currentFrame - 1, true);
  });
  ui.nextFrameBtn.addEventListener("click", () => {
    if (isPlaybackDisabled()) return;
    setCurrentFrame(state.currentFrame + 1, true);
  });
  ui.jumpForward10Btn.addEventListener("click", () => {
    if (isPlaybackDisabled()) return;
    setCurrentFrame(state.currentFrame + 10, true);
  });
  ui.playSegmentBtn.addEventListener("click", () => {
    void handlePlaySegment();
  });
  ui.autoStopToggle.addEventListener("change", () => {
    local.autoStopSegment = ui.autoStopToggle.checked;
    syncPlaybackOptionControls();
    if (isDriverPlaying()) {
      stopPlaybackBoundaryWatch();
      startPlaybackBoundaryWatch();
    }
  });
  ui.loopSegmentToggle.addEventListener("change", () => {
    local.loopSegment = ui.loopSegmentToggle.checked;
    syncPlaybackOptionControls();
    if (isDriverPlaying()) {
      stopPlaybackBoundaryWatch();
      startPlaybackBoundaryWatch();
    }
  });
  ui.playbackRateSelect.addEventListener("change", () => {
    applyPlaybackRate(Number(ui.playbackRateSelect.value));
  });
  ui.timelineZoomSelect?.addEventListener("change", () => {
    const video = currentVideo();
    if (!video) return;
    setTimelineWindow(ui.timelineZoomSelect.value, state.currentFrame);
    renderBoundaryMarkers();
    renderBoundaryHandles();
    renderFrameOnly();
  });

  ui.addSegmentBtn.addEventListener("click", handleSplitSegment);
  ui.setStartBtn.addEventListener("click", handleSetStart);
  ui.setEndBtn.addEventListener("click", handleSetEnd);
  ui.jumpSegmentStartBtn.addEventListener("click", jumpToSegmentStart);
  ui.jumpSegmentEndBtn.addEventListener("click", jumpToSegmentEnd);

  ui.videoWrap.addEventListener("click", event => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.closest(".transport-bar")) return;
    if (target.closest(".video-tile-player-wrap") || target === ui.videoWrap || target.closest(".video-overlay")) {
      togglePlayPause();
    }
  });

  ui.sliderWrap?.addEventListener("pointerdown", startTimelineScrub);
  ui.frameSlider.addEventListener("mousedown", () => {
    local.isScrubbing = true;
  });
  ui.frameSlider.addEventListener("touchstart", () => {
    local.isScrubbing = true;
  }, { passive: true });
  ui.frameSlider.addEventListener("input", () => {
    const frame = Number.parseInt(ui.frameSlider.value, 10);
    if (Number.isNaN(frame)) return;
    setCurrentFrame(frame, true);
  });
  ui.frameSlider.addEventListener("mouseup", () => {
    local.isScrubbing = false;
  });
  ui.frameSlider.addEventListener("touchend", () => {
    local.isScrubbing = false;
  });

  window.addEventListener("beforeunload", event => {
    flushEditorMutationToken();
    flushTaskMutationToken();
    trackCurrentItemClose("unload");
    if (state.dirty || state.hasUnsavedFileChanges) {
      persistDraftNow();
      event.preventDefault();
      event.returnValue = "";
    }
  });
}

function bindShortcuts() {
  local.destroyShortcuts = registerShortcuts({
    isTextEditingTarget,
    isSelectTarget,
    isNumericInputTarget,
    onCommand: async (command, event, payload) => {
      switch (command) {
        case "togglePlay":
          renderShortcutFlash("play");
          await togglePlayPause();
          break;
        case "framePrev":
          renderShortcutFlash("frame");
          if (isPlaybackDisabled() || local.isSyncSettling || local.isPlaybackRefreshing) break;
          setCurrentFrame(state.currentFrame - (event.ctrlKey || event.metaKey ? 10 : 1), true);
          break;
        case "frameNext":
          renderShortcutFlash("frame");
          if (isPlaybackDisabled() || local.isSyncSettling || local.isPlaybackRefreshing) break;
          setCurrentFrame(state.currentFrame + (event.ctrlKey || event.metaKey ? 10 : 1), true);
          break;
        case "split":
          renderShortcutFlash("split");
          handleSplitSegment();
          break;
        case "setStart":
          renderShortcutFlash("boundary");
          handleSetStart();
          break;
        case "setEnd":
          renderShortcutFlash("boundary");
          handleSetEnd();
          break;
        case "jumpSegStart":
          renderShortcutFlash("jumpEdge");
          if (local.isSyncSettling || local.isPlaybackRefreshing) break;
          jumpToSegmentStart();
          break;
        case "jumpSegEnd":
          renderShortcutFlash("jumpEdge");
          if (local.isSyncSettling || local.isPlaybackRefreshing) break;
          jumpToSegmentEnd();
          break;
        case "nextSegment": {
          renderShortcutFlash("segment");
          const video = currentVideo();
          if (!video) break;
          const next = Math.min(video.action_config.length - 1, state.selectedSegIdx + 1);
          selectSegment(next, { jumpToStart: false, focusEditor: false });
          break;
        }
        case "prevSegment": {
          renderShortcutFlash("segment");
          const video = currentVideo();
          if (!video) break;
          const prev = Math.max(0, state.selectedSegIdx - 1);
          selectSegment(prev, { jumpToStart: false, focusEditor: false });
          break;
        }
        case "jumpSegment": {
          renderShortcutFlash("jump");
          const video = currentVideo();
          if (!video) break;
          const index = clamp(payload?.index || 0, 0, video.action_config.length - 1);
          selectSegment(index, { jumpToStart: true, focusEditor: false });
          break;
        }
        case "commit":
          renderShortcutFlash("commit");
          if (local.isSaving) {
            showToast("正在保存，请稍候", "info");
            break;
          }
          await commitCurrentVideoEdits();
          break;
        case "save":
          renderShortcutFlash("save");
          await handleSave();
          break;
        case "undo":
          renderShortcutFlash("undo");
          handleUndo();
          break;
        case "redo":
          renderShortcutFlash("redo");
          handleRedo();
          break;
        case "nextVideo":
          await switchVideo(1);
          break;
        case "prevVideo":
          await switchVideo(-1);
          break;
        default:
          break;
      }
    },
  });
}

async function init() {
  applyAnnotationPanelWidth(readStoredAnnotationPanelWidth());
  syncPlaybackOptionControls();
  clearCurrentView();
  bindAdvancedToggle();
  bindPanelResizeHandle();
  bindInfoPanelToggle();
  setInfoPanelVisible(false);
  bindEditorInputs();
  bindEvents();
  bindShortcuts();
  renderAnnotatorState();

  const ready = await ensureCollaborationReady();
  if (!ready) return;

  const ok = await fetchAndRenderMeta();
  if (ok && state.meta.length > 0) {
    await loadVideoByIndex(0);
  }
}

init();
