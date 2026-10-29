export function getSegmentCompleteness(segment) {
  const missingAction = !String(segment?.action_text || "").trim();
  const missingSkill = !String(segment?.skill || "").trim();
  return {
    missingAction,
    missingSkill,
    complete: !missingAction && !missingSkill,
  };
}

export function getVideoStats(video) {
  if (!video || !Array.isArray(video.action_config) || !video.action_config.length) {
    return {
      totalSegments: 0,
      incompleteCount: 0,
      missingActionCount: 0,
      missingSkillCount: 0,
      avgSegmentFrames: 0,
    };
  }

  const segments = video.action_config;
  let incompleteCount = 0;
  let missingActionCount = 0;
  let missingSkillCount = 0;
  let totalFrames = 0;

  for (const segment of segments) {
    const completeness = getSegmentCompleteness(segment);
    if (!completeness.complete) incompleteCount += 1;
    if (completeness.missingAction) missingActionCount += 1;
    if (completeness.missingSkill) missingSkillCount += 1;
    totalFrames += Math.max(0, Number(segment.end_frame) - Number(segment.start_frame));
  }

  return {
    totalSegments: segments.length,
    incompleteCount,
    missingActionCount,
    missingSkillCount,
    avgSegmentFrames: segments.length ? totalFrames / segments.length : 0,
  };
}

export function sanitizeTasks(lines) {
  const seen = new Set();
  const out = [];
  for (const raw of lines) {
    const value = String(raw || "").trim();
    if (!value) continue;
    if (seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
}

function filterStringOptions(options, query) {
  const q = String(query || "").trim().toLowerCase();
  const list = Array.isArray(options) ? options : [];
  if (!q) return list.slice();
  return list.filter(item => String(item || "").toLowerCase().includes(q));
}

function ensureStringOption(options, value) {
  const list = Array.isArray(options) ? options.slice() : [];
  const normalized = String(value || "").trim();
  if (normalized && !list.includes(normalized)) {
    list.unshift(normalized);
  }
  return list;
}

export function filterSkillOptions(skillOptions, query) {
  return filterStringOptions(skillOptions, query);
}

export function ensureSkillOption(skillOptions, skill) {
  return ensureStringOption(skillOptions, skill);
}

export function filterActionTextOptions(actionTextOptions, query) {
  return filterStringOptions(actionTextOptions, query);
}

export function ensureActionTextOption(actionTextOptions, actionText) {
  return ensureStringOption(actionTextOptions, actionText);
}
