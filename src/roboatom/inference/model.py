"""RoboAtom-Anno generation with merged Qwen3.5 weights."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig

from .prompt import SYSTEM, camera_layout, user_prompt
from .video import preprocess_video


def encode_prompt(
    tokenizer: Any, content: str, grid: torch.Tensor, indices: list[int], merge_size: int,
) -> list[int]:
    """Encode Qwen's non-thinking template with original frame-index labels.

    Args:
        tokenizer: Model tokenizer.
        content: User instruction containing one video placeholder.
        grid: Temporal, height and width patch counts for the video.
        indices: Original frame indices, one per temporal visual block.
        merge_size: Spatial patch merge factor.

    Returns:
        Token IDs for the system and user turn, followed by the assistant prefix.
    """
    if int(grid[0]) != len(indices):
        raise ValueError("Video grid and source frame indices differ")
    tokens_per_frame = int(grid[1:].prod()) // merge_size ** 2
    blocks = "".join(
        f"Frame {index}\n<|vision_start|>"
        + "<|video_pad|>" * tokens_per_frame + "<|vision_end|>"
        for index in indices
    )
    content = content.replace("<video>", blocks, 1)
    system = tokenizer.encode(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n", add_special_tokens=False)
    user = tokenizer.encode(
        f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    return system + user


class Segmenter:
    """Load a merged RoboAtom-Anno model once and segment local videos.

    Args:
        model_path: Directory containing complete Hugging Face model weights.
        media_root: Root containing the relative video paths supplied in requests.
    """

    def __init__(self, model_path: Path, media_root: Path) -> None:
        self.media_root = media_root
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.eos_token = "<|im_end|>"
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
        ).eval()
        self.model.requires_grad_(False)
        self.generation = GenerationConfig(
            max_new_tokens=4096, do_sample=False, num_beams=1,
            repetition_penalty=1.0, length_penalty=1.0,
            eos_token_id=[self.tokenizer.eos_token_id], pad_token_id=self.tokenizer.pad_token_id,
        )

    @torch.inference_mode()
    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Generate frame-based action segments from benchmark input metadata.

        Args:
            row: Input row containing ID, task, length, FPS and camera metadata.

        Returns:
            Episode ID, frame metadata and generated skill/subtask segments.
        """
        video = camera_layout(row)
        tensors, indices = preprocess_video(video, self.media_root, self.processor)
        prompt = encode_prompt(
            self.tokenizer, user_prompt(row, video["layout"]), tensors["video_grid_thw"][0],
            indices, self.processor.video_processor.merge_size,
        )
        inputs = torch.tensor([prompt], device=self.model.device)
        token_types = torch.zeros_like(inputs)
        token_types[inputs == self.processor.image_token_id] = 1
        token_types[inputs == self.processor.video_token_id] = 2
        tensors = {
            key: value.to(device=self.model.device, dtype=self.model.dtype if value.is_floating_point() else value.dtype)
            for key, value in tensors.items()
        }
        generated = self.model.generate(
            inputs=inputs, attention_mask=torch.ones_like(inputs), mm_token_type_ids=token_types,
            generation_config=self.generation, **tensors,
        )
        text = self.tokenizer.decode(
            generated[0, len(prompt):], skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )
        segments = json.loads(text)["segments"]
        return {
            "episode_id": row["benchmark_id"], "num_frames": row["length"],
            "fps": row["fps"], "frame_interval": "closed", "segments": segments,
        }
