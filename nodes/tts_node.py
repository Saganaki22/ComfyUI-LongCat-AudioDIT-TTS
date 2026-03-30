import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .loader import (
    get_model_names,
    load_model,
    numpy_audio_to_comfy,
    normalize_text,
    approx_duration_from_text,
    resolve_device,
    _strip_auto_download_suffix,
)
from .model_cache import (
    cancel_event,
    get_cache_key,
    get_cached_model,
    is_offloaded,
    offload_model_to_cpu,
    resume_model_to_cuda,
    set_cached_model,
    unload_model,
)

try:
    from comfy.utils import ProgressBar

    _PBAR = True
except ImportError:
    _PBAR = False

try:
    import comfy.model_management as mm

    _MM = True
except ImportError:
    _MM = False

logger = logging.getLogger("LongCatAudioDiT")


class LongCatTTS:
    @classmethod
    def INPUT_TYPES(cls):
        model_names = get_model_names()
        return {
            "required": {
                "model_path": (
                    model_names,
                    {
                        "tooltip": (
                            "LongCat-AudioDiT model. "
                            "Models are stored in ComfyUI/models/audiodit/"
                        ),
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello! This is your Longcat Audio node speaking, everything is set-up and running smoothly!",
                        "tooltip": "Text to synthesize.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 16,
                        "min": 4,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Number of ODE Euler steps. More steps = better quality but slower.",
                    },
                ),
                "guidance_strength": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "CFG/APG guidance strength. Higher = more guidance.",
                    },
                ),
                "guidance_method": (
                    ["cfg", "apg"],
                    {
                        "default": "cfg",
                        "tooltip": "Guidance method. 'apg' often gives better results for voice cloning.",
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {
                        "default": "auto",
                        "tooltip": "Compute device. 'auto' picks CUDA > MPS > CPU.",
                    },
                ),
                "dtype": (
                    ["auto", "bf16", "fp16", "fp32"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Model dtype. 'auto' picks bf16 for CUDA, "
                            "fp16 for MPS, fp32 for CPU."
                        ),
                    },
                ),
                "attention": (
                    ["auto", "sdpa", "sage_attention", "flash_attention"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Attention implementation. "
                            "'auto' uses model default (SDPA). "
                            "'sage_attention' requires sageattention package. "
                            "'flash_attention' forces FlashAttention via SDPBackend."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Random seed. 0 = random.",
                    },
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep model loaded between runs. "
                            "Model is automatically offloaded to CPU after generation to free VRAM, "
                            "then resumed to GPU on the next run."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "LongCat-AudioDiT"
    DESCRIPTION = "LongCat-AudioDiT Text-to-Speech. Zero-shot TTS with diffusion-based waveform generation."

    def generate(
        self,
        model_path: str,
        text: str,
        steps: int,
        guidance_strength: float,
        guidance_method: str,
        device: str,
        dtype: str,
        attention: str,
        seed: int,
        keep_model_loaded: bool,
    ) -> Tuple[dict]:
        cancel_event.clear()
        self._check_interrupt()

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        model, tokenizer = self._get_model(
            model_path, device, dtype, attention, keep_model_loaded
        )

        import soundfile as sf

        pbar = ProgressBar(3) if _PBAR else None

        text_norm = normalize_text(text)
        logger.info(f"TTS: {text_norm[:80]}{'...' if len(text_norm) > 80 else ''}")

        if pbar:
            pbar.update_absolute(1, 3)

        inputs = tokenizer([text_norm], padding="longest", return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        sr = model.config.sampling_rate
        full_hop = model.config.latent_hop
        max_duration = model.config.max_wav_duration

        dur_sec = approx_duration_from_text(text_norm, max_duration=max_duration)
        duration = int(dur_sec * sr // full_hop)
        logger.info(f"Estimated duration: {dur_sec:.2f}s ({duration} latent frames)")

        actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(actual_seed)

        self._check_interrupt()

        if pbar:
            pbar.update_absolute(2, 3)

        try:
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    duration=duration,
                    steps=steps,
                    cfg_strength=guidance_strength,
                    guidance_method=guidance_method,
                    seed=actual_seed,
                )

            wav = output.waveform.squeeze().detach().cpu().numpy()
            logger.info(f"Generated {len(wav) / sr:.2f}s of audio at {sr}Hz")

            result = numpy_audio_to_comfy(wav, sr)

            if pbar:
                pbar.update_absolute(3, 3)

        finally:
            if not keep_model_loaded:
                unload_model()
            else:
                offload_model_to_cpu()

        return (result,)

    def _get_model(
        self,
        model_path,
        device,
        dtype,
        attention,
        keep_loaded=False,
    ):
        model_name = _strip_auto_download_suffix(model_path)
        key = get_cache_key(model_path, device, dtype, attention)
        cached_model, cached_tokenizer, cached_key = get_cached_model()

        # Check if settings changed - force full unload if so
        if cached_model is not None and cached_key != key:
            logger.info(
                f"Settings changed (model/device/dtype/attention) — "
                f"unloading cached model. Old: {cached_key}, New: {key}"
            )
            unload_model()

        if cached_model is not None and cached_key == key:
            if is_offloaded():
                device_str, _ = resolve_device(device)
                logger.info(f"Resuming offloaded model to {device_str}...")
                resume_model_to_cuda(device_str)
            else:
                logger.info("Reusing cached LongCat-AudioDiT model.")
            return cached_model, cached_tokenizer

        model, tokenizer = load_model(model_path, device, dtype, attention)
        set_cached_model(model, tokenizer, key, keep_loaded=keep_loaded)
        return model, tokenizer

    def _check_interrupt(self):
        if _MM:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                cancel_event.set()
                raise
