"""MiniMax TTS ComfyUI node.

Calls the MiniMax Text-to-Audio API (t2a_v2) and returns a ComfyUI AUDIO tensor.
API key is read from the MINIMAX_API_KEY environment variable or the api_key input.
"""

import io
import json
import logging
import os
import urllib.request

import numpy as np

logger = logging.getLogger("LongCatAudioDiT")

MINIMAX_TTS_MODELS = [
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
]

MINIMAX_TTS_VOICES = [
    "English_Graceful_Lady",
    "English_Insightful_Speaker",
    "English_radiant_girl",
    "English_Persuasive_Man",
    "English_Lucky_Robot",
    "English_expressive_narrator",
]

# PCM: 16-bit signed integers, little-endian
_PCM_DTYPE = np.int16
_PCM_MAX = 32768.0


def _pcm_bytes_to_numpy(pcm_bytes: bytes, sample_rate: int) -> tuple:
    """Convert raw signed 16-bit PCM bytes to a float32 numpy array."""
    audio_int16 = np.frombuffer(pcm_bytes, dtype=_PCM_DTYPE)
    audio_float32 = audio_int16.astype(np.float32) / _PCM_MAX
    return audio_float32, sample_rate


def _numpy_to_comfy(audio_np: np.ndarray, sample_rate: int) -> dict:
    """Convert a 1-D float32 numpy array to the ComfyUI AUDIO dict format."""
    import torch

    waveform = torch.from_numpy(audio_np[np.newaxis, np.newaxis, :]).float().contiguous()
    return {"waveform": waveform, "sample_rate": sample_rate}


def minimax_tts(
    text: str,
    api_key: str,
    model: str = "speech-2.8-hd",
    voice_id: str = "English_Graceful_Lady",
    speed: float = 1.0,
    sample_rate: int = 32000,
    base_url: str = "https://api.minimax.io",
) -> tuple:
    """Call MiniMax TTS API (streaming SSE with PCM) and return (pcm_bytes, sample_rate)."""
    url = f"{base_url.rstrip('/')}/v1/t2a_v2"
    payload = json.dumps(
        {
            "model": model,
            "text": text,
            "stream": True,
            "stream_options": {"exclude_aggregated_audio": True},
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": sample_rate,
                "format": "pcm",
                "channel": 1,
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    audio_chunks: list = []

    with urllib.request.urlopen(req, timeout=60) as resp:
        decoder = io.TextIOWrapper(resp, encoding="utf-8", errors="replace")
        for raw_line in decoder:
            line = raw_line.rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            json_str = line[5:].strip()
            if not json_str or json_str == "[DONE]":
                continue
            try:
                event = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            base_resp = event.get("base_resp", {})
            if base_resp.get("status_code", 0) != 0:
                raise RuntimeError(
                    f"MiniMax TTS error {base_resp.get('status_code')}: "
                    f"{base_resp.get('status_msg', 'unknown')}"
                )

            hex_audio = event.get("data", {}).get("audio", "")
            if hex_audio:
                audio_chunks.append(bytes.fromhex(hex_audio))

    if not audio_chunks:
        raise RuntimeError("MiniMax TTS returned no audio data.")

    return b"".join(audio_chunks), sample_rate


class MiniMaxTTS:
    """ComfyUI node: synthesize speech via the MiniMax TTS API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello! This is MiniMax TTS speaking.",
                        "tooltip": "Text to synthesize (max 10,000 characters).",
                    },
                ),
                "model": (
                    MINIMAX_TTS_MODELS,
                    {
                        "default": "speech-2.8-hd",
                        "tooltip": (
                            "MiniMax TTS model. "
                            "speech-2.8-hd: highest quality. "
                            "speech-2.8-turbo: faster."
                        ),
                    },
                ),
                "voice_id": (
                    MINIMAX_TTS_VOICES,
                    {
                        "default": "English_Graceful_Lady",
                        "tooltip": "Voice to use for synthesis.",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Speech speed. 1.0 = normal, 0.5 = half, 2.0 = double.",
                    },
                ),
            },
            "optional": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "MiniMax API key. "
                            "Leave empty to use the MINIMAX_API_KEY environment variable."
                        ),
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "https://api.minimax.io",
                        "tooltip": (
                            "MiniMax API base URL. "
                            "Overseas: https://api.minimax.io, "
                            "Domestic: https://api.minimaxi.com"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"
    CATEGORY = "LongCat-AudioDiT"
    DESCRIPTION = (
        "MiniMax Text-to-Speech node. "
        "Calls the MiniMax TTS API (t2a_v2) and returns synthesized audio. "
        "Requires MINIMAX_API_KEY environment variable or api_key input."
    )

    def synthesize(
        self,
        text: str,
        model: str,
        voice_id: str,
        speed: float,
        api_key: str = "",
        base_url: str = "https://api.minimax.io",
    ) -> tuple:
        if not text.strip():
            raise ValueError("Text cannot be empty.")

        resolved_key = api_key.strip() or os.environ.get("MINIMAX_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "MiniMax API key is required. "
                "Set the MINIMAX_API_KEY environment variable or provide it via the api_key input."
            )

        logger.info(
            f"MiniMax TTS: model={model}, voice={voice_id}, "
            f"text={text[:60]}{'...' if len(text) > 60 else ''}"
        )

        pcm_bytes, sample_rate = minimax_tts(
            text=text,
            api_key=resolved_key,
            model=model,
            voice_id=voice_id,
            speed=speed,
            base_url=base_url,
        )

        logger.info(f"MiniMax TTS: received {len(pcm_bytes):,} bytes of PCM audio.")

        audio_np, sr = _pcm_bytes_to_numpy(pcm_bytes, sample_rate)
        result = _numpy_to_comfy(audio_np, sr)

        logger.info(f"MiniMax TTS: generated {len(audio_np) / sr:.2f}s of audio at {sr}Hz.")
        return (result,)

