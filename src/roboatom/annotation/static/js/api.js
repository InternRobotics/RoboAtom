async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.error || `Request failed: ${response.status}`;
    const err = new Error(message);
    err.payload = payload;
    err.fieldErrors = Array.isArray(payload.field_errors) ? payload.field_errors : [];
    err.status = response.status;
    throw err;
  }
  return payload;
}

export async function getMeta(options = {}) {
  return fetchJson("/api/meta", options);
}

export async function getCollabConfig(options = {}) {
  return fetchJson("/api/collab/config", options);
}

export async function selectAnnotator(annotatorId, options = {}) {
  return fetchJson("/api/collab/select", {
    ...options,
    method: "POST",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    body: JSON.stringify({ annotator_id: annotatorId }),
  });
}

export async function getVideo(index, options = {}) {
  return fetchJson(`/api/video/${index}`, options);
}

export async function switchInput(inputJson, outputJson, options = {}) {
  return fetchJson("/api/input/switch", {
    ...options,
    method: "POST",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    body: JSON.stringify({ input_json: inputJson, output_json: outputJson || "" }),
  });
}

export async function updateAnnotation(index, payload) {
  return fetchJson(`/api/video/${index}/annotation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function updateDiscard(index, payload) {
  return fetchJson(`/api/video/${index}/discard`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function saveAll(payload = {}) {
  return fetchJson("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
}

export async function trackAuditEvent(payload = {}, options = {}) {
  return fetchJson("/api/audit/event", {
    ...options,
    method: "POST",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    body: JSON.stringify(payload || {}),
  });
}

export function sendAuditEvent(payload = {}) {
  const body = JSON.stringify(payload || {});
  if (typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
    try {
      const blob = new Blob([body], { type: "application/json" });
      if (navigator.sendBeacon("/api/audit/event", blob)) return;
    } catch (_) {
      // fallback below
    }
  }
  void fetch("/api/audit/event", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    keepalive: true,
  }).catch(() => {});
}
