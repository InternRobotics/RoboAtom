function isDigitKey(key) {
  return /^[1-9]$/.test(key);
}

export function registerShortcuts(options) {
  const {
    isTextEditingTarget,
    isSelectTarget,
    isNumericInputTarget,
    onCommand,
  } = options;

  const handler = event => {
    const key = event.key;
    const inTextEditing = isTextEditingTarget(event.target);
    const inSelect = isSelectTarget ? isSelectTarget(event.target) : false;
    // Numeric-only inputs (e.g. video jump box): allow Space/N/P but let digits
    // and arrow keys pass through so the user can type numbers normally.
    const inNumericInput = isNumericInputTarget ? isNumericInputTarget(event.target) : false;

    // Global save/undo/redo should work everywhere.
    if ((event.ctrlKey || event.metaKey) && (key === "s" || key === "S")) {
      event.preventDefault();
      onCommand("save", event);
      return;
    }
    if ((event.ctrlKey || event.metaKey) && (key === "z" || key === "Z")) {
      event.preventDefault();
      onCommand(event.shiftKey ? "redo" : "undo", event);
      return;
    }
    if ((event.ctrlKey || event.metaKey) && (key === "y" || key === "Y")) {
      event.preventDefault();
      onCommand("redo", event);
      return;
    }
    if ((event.ctrlKey || event.metaKey) && key === "Enter") {
      event.preventDefault();
      onCommand("commit", event);
      return;
    }

    // In text editing (TEXTAREA / text INPUT): only Enter-to-commit, then bail.
    if (inTextEditing) {
      const targetTag = event.target instanceof HTMLElement ? event.target.tagName : "";
      if (key === "Enter" && targetTag !== "TEXTAREA" && !event.shiftKey) {
        event.preventDefault();
        onCommand("commit", event);
      }
      return;
    }

    // In numeric INPUT (e.g. video jump box): only Space/N/P shortcuts work.
    // Digits, arrow keys, Enter, Escape etc. all pass through to the input.
    if (inNumericInput) {
      if (key === " ") {
        event.preventDefault();
        onCommand("togglePlay", event);
        return;
      }
      if (key === "n" || key === "N") {
        event.preventDefault();
        onCommand("nextVideo", event);
        return;
      }
      if (key === "p" || key === "P") {
        event.preventDefault();
        onCommand("prevVideo", event);
        return;
      }
      // Everything else (digits, arrows, Enter, Escape) passes through natively.
      return;
    }

    // In SELECT element: allow playback/navigation shortcuts but skip
    // ArrowUp/ArrowDown to avoid conflicting with native SELECT behaviour.
    if (inSelect) {
      if (key === " ") {
        event.preventDefault();
        onCommand("togglePlay", event);
        return;
      }
      if (key === ",") {
        event.preventDefault();
        onCommand("framePrev", event);
        return;
      }
      if (key === ".") {
        event.preventDefault();
        onCommand("frameNext", event);
        return;
      }
      if (key === "[") {
        event.preventDefault();
        onCommand("setStart", event);
        return;
      }
      if (key === "]") {
        event.preventDefault();
        onCommand("setEnd", event);
        return;
      }
      if (key === "Tab") {
        event.preventDefault();
        onCommand(event.shiftKey ? "prevSegment" : "nextSegment", event);
        return;
      }
      if (key === "ArrowLeft") {
        event.preventDefault();
        onCommand(event.shiftKey ? "jumpSegStart" : "framePrev", event);
        return;
      }
      if (key === "ArrowRight") {
        event.preventDefault();
        onCommand(event.shiftKey ? "jumpSegEnd" : "frameNext", event);
        return;
      }
      if (key === "n" || key === "N") {
        event.preventDefault();
        onCommand("nextVideo", event);
        return;
      }
      if (key === "p" || key === "P") {
        event.preventDefault();
        onCommand("prevVideo", event);
        return;
      }
      // ArrowUp / ArrowDown: let SELECT handle natively — do NOT intercept.
      return;
    }

    // Normal (non-editing, non-select) state: all shortcuts active.
    if (key === " ") {
      event.preventDefault();
      onCommand("togglePlay", event);
      return;
    }
    if (key === ",") {
      event.preventDefault();
      onCommand("framePrev", event);
      return;
    }
    if (key === ".") {
      event.preventDefault();
      onCommand("frameNext", event);
      return;
    }
    if (key === "s" || key === "S") {
      event.preventDefault();
      onCommand("split", event);
      return;
    }
    if (key === "[") {
      event.preventDefault();
      onCommand("setStart", event);
      return;
    }
    if (key === "]") {
      event.preventDefault();
      onCommand("setEnd", event);
      return;
    }
    if (key === "Tab") {
      event.preventDefault();
      onCommand(event.shiftKey ? "prevSegment" : "nextSegment", event);
      return;
    }
    if (key === "ArrowLeft") {
      event.preventDefault();
      onCommand(event.shiftKey ? "jumpSegStart" : "framePrev", event);
      return;
    }
    if (key === "ArrowRight") {
      event.preventDefault();
      onCommand(event.shiftKey ? "jumpSegEnd" : "frameNext", event);
      return;
    }
    if (key === "ArrowUp") {
      event.preventDefault();
      onCommand("prevSegment", event);
      return;
    }
    if (key === "ArrowDown") {
      event.preventDefault();
      onCommand("nextSegment", event);
      return;
    }
    if (key === "n" || key === "N") {
      event.preventDefault();
      onCommand("nextVideo", event);
      return;
    }
    if (key === "p" || key === "P") {
      event.preventDefault();
      onCommand("prevVideo", event);
      return;
    }
    if (isDigitKey(key)) {
      event.preventDefault();
      onCommand("jumpSegment", event, { index: Number(key) - 1 });
    }
  };

  document.addEventListener("keydown", handler);
  return () => {
    document.removeEventListener("keydown", handler);
  };
}
