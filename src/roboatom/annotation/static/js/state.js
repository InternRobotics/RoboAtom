export const MAX_HISTORY = 200;

export const state = {
  meta: [],
  annotationStatuses: [],
  skillOptions: [],
  actionTextOptions: [],
  inputJson: "",
  outputJson: "",
  appMode: "annotation",
  modelReviewJson: "",
  collab: {
    enabled: false,
    annotatorId: "",
    annotatorName: "",
    annotatorRole: "",
    annotators: [],
  },
  currentIndex: -1,
  serverSnapshot: null,
  workingCopy: null,
  currentFrame: 0,
  selectedSegIdx: 0,
  statusMode: "saved", // draft | committed | saved
  dirty: false,
  hasUnsavedFileChanges: false,
  lastAppliedAt: null,
  lastSavedAt: null,
  isLoadingVideo: false,
  activeVideoRequestId: 0,
  timelineWindow: null,
  undoStack: [],
  redoStack: [],
};

export function cloneDeep(value) {
  if (typeof structuredClone === "function") {
    try {
      return structuredClone(value);
    } catch (_) {
      // fallback below
    }
  }
  return JSON.parse(JSON.stringify(value));
}

export function snapshotHash(video) {
  if (!video) return "";
  return JSON.stringify({
    tasks: video.tasks || [],
    action_config: video.action_config || [],
    discarded: Boolean(video.discarded),
    discard_reason: String(video.discard_reason || ""),
    discard_note: String(video.discard_note || ""),
  });
}

function normalizeAnnotationStatus(item = {}) {
  return {
    status: String(item?.status || "unopened"),
    modified: Boolean(item?.modified),
    last_modified_by: String(item?.last_modified_by || ""),
    last_saved_at: item?.last_saved_at || null,
    change_count: Number.isFinite(Number(item?.change_count)) ? Number(item.change_count) : 0,
  };
}

export function setAnnotationStatuses(statuses) {
  const normalized = Array.isArray(statuses) ? statuses.map(normalizeAnnotationStatus) : [];
  state.annotationStatuses = normalized;
  state.meta = state.meta.map((item, index) => ({
    ...item,
    annotation_status: normalized[index] || normalizeAnnotationStatus(item.annotation_status),
  }));
}

export function setAnnotationStatus(index, status) {
  if (index < 0 || index >= state.meta.length) return;
  const normalized = normalizeAnnotationStatus(status);
  state.annotationStatuses[index] = normalized;
  state.meta[index] = {
    ...state.meta[index],
    annotation_status: normalized,
  };
}

export function setMeta(meta, skillOptions, actionTextOptions, inputJson, outputJson, appMode = "annotation", modelReviewJson = "", annotationStatuses = []) {
  const normalizedStatuses = Array.isArray(annotationStatuses)
    ? annotationStatuses.map(normalizeAnnotationStatus)
    : [];
  state.annotationStatuses = normalizedStatuses;
  state.meta = Array.isArray(meta)
    ? meta.map((item, index) => ({
      ...item,
      annotation_status: normalizedStatuses[index] || normalizeAnnotationStatus(item?.annotation_status),
    }))
    : [];
  state.skillOptions = Array.isArray(skillOptions) ? skillOptions.slice() : [];
  state.actionTextOptions = Array.isArray(actionTextOptions) ? actionTextOptions.slice() : [];
  if (typeof inputJson === "string") state.inputJson = inputJson;
  if (typeof outputJson === "string") state.outputJson = outputJson;
  if (typeof appMode === "string") state.appMode = appMode || "annotation";
  if (typeof modelReviewJson === "string") state.modelReviewJson = modelReviewJson;
}

export function clearMetaState() {
  state.meta = [];
  state.annotationStatuses = [];
  state.skillOptions = [];
  state.actionTextOptions = [];
  state.inputJson = "";
  state.outputJson = "";
  state.appMode = "annotation";
  state.modelReviewJson = "";
}

export function setCollabConfig(payload = {}) {
  const enabled = Boolean(payload?.enabled);
  state.collab.enabled = enabled;
  state.collab.annotatorId = enabled ? String(payload.current_annotator_id || "") : "";
  state.collab.annotatorName = enabled ? String(payload.current_annotator_name || "") : "";
  state.collab.annotatorRole = enabled ? String(payload.current_annotator_role || "") : "";
  state.collab.annotators = enabled && Array.isArray(payload.annotators)
    ? payload.annotators.map(item => ({
      id: String(item?.id || ""),
      name: String(item?.name || ""),
      role: String(item?.role || "annotator"),
      count: Number.isFinite(Number(item?.count)) ? Number(item.count) : 0,
      start_index: Number.isFinite(Number(item?.start_index)) ? Number(item.start_index) : 0,
    }))
    : [];
}

export function setVideo(index, payload) {
  state.currentIndex = index;
  state.serverSnapshot = cloneDeep(payload);
  state.workingCopy = cloneDeep(payload);
  state.currentFrame = 0;
  state.selectedSegIdx = 0;
  clearHistory();
  markLoadedFromDisk();
}

export function applyWorkingCopyFromDraft(workingCopy) {
  state.workingCopy = cloneDeep(workingCopy);
  state.currentFrame = 0;
  state.selectedSegIdx = 0;
  clearHistory();
  markDraft();
}

export function clearHistory() {
  state.undoStack = [];
  state.redoStack = [];
}

export function createMutationToken(type) {
  if (!state.workingCopy) return null;
  return {
    type,
    before: cloneDeep(state.workingCopy),
  };
}

export function finalizeMutation(token) {
  if (!token || !state.workingCopy) return false;
  const beforeHash = JSON.stringify(token.before);
  const afterHash = JSON.stringify(state.workingCopy);
  if (beforeHash === afterHash) return false;

  state.undoStack.push({
    type: token.type,
    before: token.before,
    after: cloneDeep(state.workingCopy),
  });
  if (state.undoStack.length > MAX_HISTORY) {
    state.undoStack.splice(0, state.undoStack.length - MAX_HISTORY);
  }
  state.redoStack = [];
  markDraft();
  return true;
}

export function undoMutation() {
  if (!state.undoStack.length || !state.workingCopy) return false;
  const entry = state.undoStack.pop();
  state.redoStack.push(entry);
  state.workingCopy = cloneDeep(entry.before);
  markDraft();
  return true;
}

export function redoMutation() {
  if (!state.redoStack.length || !state.workingCopy) return false;
  const entry = state.redoStack.pop();
  state.undoStack.push(entry);
  state.workingCopy = cloneDeep(entry.after);
  markDraft();
  return true;
}

export function markDraft() {
  state.dirty = true;
  state.hasUnsavedFileChanges = true;
  state.statusMode = "draft";
}

export function markCommitted() {
  state.dirty = false;
  state.hasUnsavedFileChanges = true;
  state.statusMode = "committed";
  state.lastAppliedAt = Date.now();
  if (state.workingCopy) {
    state.serverSnapshot = cloneDeep(state.workingCopy);
  }
}

export function markSaved() {
  const savedAt = Date.now();
  state.dirty = false;
  state.hasUnsavedFileChanges = false;
  state.statusMode = "saved";
  state.lastSavedAt = savedAt;
  if (!state.lastAppliedAt) {
    state.lastAppliedAt = savedAt;
  }
}

export function markLoadedFromDisk() {
  state.dirty = false;
  if (state.hasUnsavedFileChanges) {
    state.statusMode = "committed";
  } else {
    state.statusMode = "saved";
    state.lastAppliedAt = null;
    state.lastSavedAt = null;
  }
  state.isLoadingVideo = false;
  state.timelineWindow = null;
}

export function currentVideo() {
  return state.workingCopy;
}

export function commitMetaCounts(tasksCount, segmentsCount) {
  if (state.currentIndex < 0 || state.currentIndex >= state.meta.length) return;
  state.meta[state.currentIndex].tasks_count = tasksCount;
  state.meta[state.currentIndex].segments_count = segmentsCount;
}
