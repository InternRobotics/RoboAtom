export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function frameToTime(video, frame) {
  if (!video || !video.fps) return 0;
  return frame / video.fps;
}

export function frameToSeekTime(video, frame) {
  if (!video || !video.fps) return 0;
  const safeFrame = clamp(Math.round(frame), 0, Math.max(video.frames - 1, 0));
  return (safeFrame + 0.5) / video.fps;
}

export function timeToFrame(video, timeSec) {
  if (!video) return 0;
  return clamp(Math.floor(timeSec * video.fps), 0, Math.max(video.frames - 1, 0));
}

export function findSegmentIndexByFrame(video, frame) {
  if (!video || !Array.isArray(video.action_config)) return -1;
  return video.action_config.findIndex(seg => frame >= seg.start_frame && frame < seg.end_frame);
}

export function getNearBoundary(video, frame, threshold) {
  if (!video || !Array.isArray(video.action_config)) return null;
  const t = Number.isFinite(threshold) ? threshold : 2;
  let nearest = null;
  let minDistance = Infinity;
  for (const seg of video.action_config) {
    const distance = Math.abs(seg.start_frame - frame);
    if (distance < minDistance) {
      minDistance = distance;
      nearest = seg.start_frame;
    }
  }
  if (nearest === null || minDistance > t) return null;
  return { frame: nearest, distance: minDistance };
}

export function isBoundaryFrame(video, frame) {
  if (!video || !Array.isArray(video.action_config)) return false;
  return video.action_config.some(seg => seg.start_frame === frame);
}

export function reindexSegments(video) {
  if (!video || !Array.isArray(video.action_config)) return;
  video.action_config.forEach((seg, index) => {
    seg.seg_id = index;
  });
}

export function validateSegmentTopology(video) {
  if (!video) return { ok: false, error: "video is required" };
  const segments = video.action_config || [];
  const frames = Number(video.frames);
  if (!Number.isInteger(frames) || frames <= 0) {
    return { ok: false, error: "frames must be a positive integer" };
  }
  if (!segments.length) {
    return { ok: false, error: "at least one segment is required" };
  }

  if (segments[0].start_frame !== 0) {
    return { ok: false, error: "first segment must start at frame 0" };
  }
  if (segments[segments.length - 1].end_frame !== frames) {
    return { ok: false, error: `last segment must end at frame ${frames}` };
  }

  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i];
    if (seg.start_frame < 0 || seg.end_frame < 0) {
      return { ok: false, error: `segment[${i}] contains negative frames` };
    }
    if (seg.start_frame >= seg.end_frame) {
      return { ok: false, error: `segment[${i}] must satisfy start_frame < end_frame` };
    }
    if (seg.end_frame > frames) {
      return { ok: false, error: `segment[${i}] end_frame exceeds frames` };
    }
    if (i > 0 && segments[i - 1].end_frame !== seg.start_frame) {
      return { ok: false, error: `segment[${i - 1}] and segment[${i}] are not contiguous` };
    }
  }
  return { ok: true };
}

export function splitSegmentAtFrame(video, frame) {
  if (!video) return { ok: false, error: "video is required" };
  const index = findSegmentIndexByFrame(video, frame);
  if (index < 0) return { ok: false, error: "current frame is outside all segments" };
  const current = video.action_config[index];

  if (frame <= current.start_frame) {
    return { ok: false, error: "split point must be after segment start" };
  }
  if (frame >= current.end_frame - 1) {
    return { ok: false, error: "split point must be before segment end" };
  }

  const newSegment = {
    seg_id: index + 1,
    start_frame: frame,
    end_frame: current.end_frame,
    action_text: "",
    skill: "",
  };
  current.end_frame = frame;
  video.action_config.splice(index + 1, 0, newSegment);
  reindexSegments(video);
  return { ok: true, newIndex: index + 1 };
}

export function deleteSegmentAtIndex(video, index) {
  if (!video) return { ok: false, error: "video is required" };
  const segments = video.action_config;
  if (!segments || !segments.length) return { ok: false, error: "no segments" };
  if (segments.length <= 1) return { ok: false, error: "at least one segment must remain" };
  if (index < 0 || index >= segments.length) return { ok: false, error: "segment index out of range" };

  const removed = segments[index];
  if (index > 0) {
    segments[index - 1].end_frame = removed.end_frame;
  } else {
    segments[index + 1].start_frame = removed.start_frame;
  }
  segments.splice(index, 1);
  reindexSegments(video);
  return { ok: true, removed };
}

export function setSegmentStart(video, index, newStart) {
  if (!video) return { ok: false, error: "video is required" };
  const segments = video.action_config;
  if (index < 0 || index >= segments.length) return { ok: false, error: "segment index out of range" };
  if (index === 0) return { ok: false, error: "first segment start is fixed at 0" };

  const current = segments[index];
  const previous = segments[index - 1];
  const boundary = clamp(newStart, previous.start_frame + 1, current.end_frame - 1);

  if (boundary === current.start_frame) {
    return { ok: false, error: "start frame unchanged" };
  }

  current.start_frame = boundary;
  previous.end_frame = boundary;
  return { ok: true, boundary };
}

export function setSegmentEnd(video, index, newEnd) {
  if (!video) return { ok: false, error: "video is required" };
  const segments = video.action_config;
  if (index < 0 || index >= segments.length) return { ok: false, error: "segment index out of range" };
  if (index >= segments.length - 1) {
    return { ok: false, error: "last segment end is fixed at total frames" };
  }

  const current = segments[index];
  const next = segments[index + 1];
  const boundary = clamp(newEnd, current.start_frame + 1, next.end_frame - 1);

  if (boundary === current.end_frame) {
    return { ok: false, error: "end frame unchanged" };
  }

  current.end_frame = boundary;
  next.start_frame = boundary;
  return { ok: true, boundary };
}

export function moveBoundary(video, leftIndex, targetFrame) {
  if (!video) return { ok: false, error: "video is required" };
  const segments = video.action_config;
  if (leftIndex < 0 || leftIndex >= segments.length - 1) {
    return { ok: false, error: "boundary index out of range" };
  }

  const left = segments[leftIndex];
  const right = segments[leftIndex + 1];
  const boundary = clamp(targetFrame, left.start_frame + 1, right.end_frame - 1);
  if (boundary === left.end_frame) {
    return { ok: false, error: "boundary unchanged" };
  }

  left.end_frame = boundary;
  right.start_frame = boundary;
  return { ok: true, boundary };
}

export function getFrameInfo(video, frame) {
  const segmentIndex = findSegmentIndexByFrame(video, frame);
  return {
    segmentIndex,
    segmentId: segmentIndex >= 0 ? video.action_config[segmentIndex].seg_id : null,
    isBoundary: isBoundaryFrame(video, frame),
  };
}

export function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const whole = Math.floor(safe);
  const milli = Math.floor((safe - whole) * 1000);
  const hour = Math.floor(whole / 3600);
  const minute = Math.floor((whole % 3600) / 60);
  const second = whole % 60;
  const mm = String(minute).padStart(2, "0");
  const ss = String(second).padStart(2, "0");
  const mmm = String(milli).padStart(3, "0");
  if (hour > 0) {
    return `${hour}:${mm}:${ss}.${mmm}`;
  }
  return `${mm}:${ss}.${mmm}`;
}

export function formatRange(startFrame, endFrame) {
  return `f${startFrame}→${Math.max(startFrame, endFrame - 1)}`;
}

export function formatDuration(video, segment) {
  const frameCount = Math.max(0, segment.end_frame - segment.start_frame);
  if (!video || !video.fps) return `${frameCount}f`;
  return `${frameCount}f · ${formatTime(frameCount / video.fps)}`;
}

export function createTimelineWindow(video, preset = "global", centerFrame = 0) {
  if (!video) {
    return { preset: "global", startFrame: 0, span: 0 };
  }

  const maxFrame = Math.max(Number(video.frames) - 1, 0);
  if (preset === "global" || maxFrame <= 0) {
    return { preset: "global", startFrame: 0, span: maxFrame };
  }

  const seconds = preset === "10s" ? 10 : 60;
  const rawSpan = Math.round(seconds * Number(video.fps || 0));
  const span = clamp(rawSpan || maxFrame, 1, maxFrame);
  const halfSpan = Math.floor(span / 2);
  const startFrame = clamp(Math.round(centerFrame) - halfSpan, 0, Math.max(maxFrame - span, 0));
  return { preset, startFrame, span };
}

export function clampTimelineWindow(video, timelineWindow) {
  if (!video) {
    return { preset: "global", startFrame: 0, span: 0 };
  }

  const preset = String(timelineWindow?.preset || "global");
  const base = createTimelineWindow(video, preset, Number(timelineWindow?.startFrame || 0));
  if (preset === "global") return base;

  const maxFrame = Math.max(Number(video.frames) - 1, 0);
  const startFrame = clamp(Math.round(Number(timelineWindow?.startFrame || 0)), 0, Math.max(maxFrame - base.span, 0));
  return { preset, startFrame, span: base.span };
}

export function ensureFrameInTimelineWindow(video, timelineWindow, frame) {
  const nextWindow = clampTimelineWindow(video, timelineWindow);
  if (!video || nextWindow.preset === "global") return nextWindow;

  const safeFrame = clamp(Math.round(frame), 0, Math.max(Number(video.frames) - 1, 0));
  const endFrame = nextWindow.startFrame + nextWindow.span;
  if (safeFrame >= nextWindow.startFrame && safeFrame <= endFrame) {
    return nextWindow;
  }
  return createTimelineWindow(video, nextWindow.preset, safeFrame);
}

export function getTimelineWindowEnd(video, timelineWindow) {
  const nextWindow = clampTimelineWindow(video, timelineWindow);
  const maxFrame = Math.max(Number(video?.frames || 0) - 1, 0);
  return clamp(nextWindow.startFrame + nextWindow.span, 0, maxFrame);
}

export function frameToTimelineRatio(video, timelineWindow, frame) {
  const nextWindow = clampTimelineWindow(video, timelineWindow);
  const startFrame = nextWindow.startFrame;
  const endFrame = getTimelineWindowEnd(video, nextWindow);
  const denominator = Math.max(endFrame - startFrame, 1);
  return clamp((frame - startFrame) / denominator, 0, 1);
}

export function ratioToTimelineFrame(video, timelineWindow, ratio) {
  const nextWindow = clampTimelineWindow(video, timelineWindow);
  const startFrame = nextWindow.startFrame;
  const endFrame = getTimelineWindowEnd(video, nextWindow);
  const denominator = Math.max(endFrame - startFrame, 1);
  return clamp(Math.round(startFrame + clamp(ratio, 0, 1) * denominator), startFrame, endFrame);
}
