const PREFIX = "subtask_annotation_ui_draft_v1";

export function buildDraftKey(inputJsonPath, video, annotatorId = "") {
  const input = String(inputJsonPath || "");
  const annotator = String(annotatorId || "single");
  const episode = String(video?.episode_index ?? "unknown");
  const path = String(video?.path || "unknown");
  return `${PREFIX}:${annotator}:${input}:${episode}:${path}`;
}

export function saveDraft(key, payload) {
  if (!key) return;
  try {
    localStorage.setItem(key, JSON.stringify(payload));
  } catch (_) {
    // ignore storage errors
  }
}

export function loadDraft(key) {
  if (!key) return null;
  try {
    const text = localStorage.getItem(key);
    if (!text) return null;
    const payload = JSON.parse(text);
    if (!payload || typeof payload !== "object") return null;
    return payload;
  } catch (_) {
    return null;
  }
}

export function clearDraft(key) {
  if (!key) return;
  try {
    localStorage.removeItem(key);
  } catch (_) {
    // ignore storage errors
  }
}
