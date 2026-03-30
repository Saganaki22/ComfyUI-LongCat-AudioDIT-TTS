import io
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


def _load_audio_tensor(audio_path: str, sr: int) -> torch.Tensor:
    import librosa

    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    return torch.from_numpy(audio).unsqueeze(0)


def comfy_audio_to_tensor(audio_dict: dict, target_sr: int) -> torch.Tensor:
    waveform = audio_dict["waveform"]
    source_sr = audio_dict["sample_rate"]

    wav = waveform[0].float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.shape[0] == 1:
        pass
    else:
        wav = wav.unsqueeze(0)

    wav = wav.squeeze(0).numpy()

    if source_sr != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=target_sr)

    return torch.from_numpy(wav).unsqueeze(0)


class LongCatVoiceCloneTTS:
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
                        "default": "The sun glows warmly in a cloudless blue sky, a soft breeze drifts through the air, and birds fill the world with gentle, cheerful songs. Everything feels alive with beauty, just waiting to be discovered.",
                        "tooltip": "Text to synthesize in the cloned voice.",
                    },
                ),
                "prompt_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Reference audio to clone the voice from. "
                            "3-15 seconds gives the best results."
                        ),
                    },
                ),
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Transcript of the prompt audio. "
                            "Required for voice cloning. Improves quality significantly."
                        ),
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 16,
                        "min": 4,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Number of ODE Euler steps.",
                    },
                ),
                "guidance_strength": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "CFG/APG guidance strength.",
                    },
                ),
                "guidance_method": (
                    ["cfg", "apg"],
                    {
                        "default": "apg",
                        "tooltip": "Guidance method. 'apg' recommended for voice cloning.",
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {
                        "default": "auto",
                        "tooltip": "Compute device.",
                    },
                ),
                "dtype": (
                    ["auto", "bf16", "fp16", "fp32"],
                    {
                        "default": "auto",
                        "tooltip": "Model dtype.",
                    },
                ),
                "attention": (
                    ["auto", "sdpa", "sage_attention", "flash_attention"],
                    {
                        "default": "auto",
                        "tooltip": "Attention implementation.",
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
    DESCRIPTION = "LongCat-AudioDiT Voice Clone TTS. Clones voice from reference audio using diffusion-based generation."

    def generate(
        self,
        model_path: str,
        text: str,
        prompt_audio: dict,
        prompt_text: str,
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
        if not prompt_text.strip():
            logger.warning(
                "No prompt text provided. Voice cloning quality may be reduced. "
                "Providing the transcript of the reference audio is recommended."
            )

        model, tokenizer = self._get_model(
            model_path, device, dtype, attention, keep_model_loaded
        )

        pbar = ProgressBar(4) if _PBAR else None

        sr = model.config.sampling_rate
        full_hop = model.config.latent_hop
        max_duration = model.config.max_wav_duration

        logger.info("Encoding prompt audio...")
        prompt_wav = comfy_audio_to_tensor(prompt_audio, sr).to(model.device)

        if pbar:
            pbar.update_absolute(1, 4)

        text_norm = normalize_text(text)
        prompt_text_norm = normalize_text(prompt_text) if prompt_text.strip() else ""

        full_text = f"{prompt_text_norm} {text_norm}" if prompt_text_norm else text_norm
        logger.info(
            f"Voice Clone TTS: {text_norm[:80]}{'...' if len(text_norm) > 80 else ''}"
        )

        inputs = tokenizer([full_text], padding="longest", return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        self._check_interrupt()

        prompt_audio_tensor = prompt_wav.unsqueeze(0)

        off = 3
        pw = prompt_wav.clone()
        if pw.shape[-1] % full_hop != 0:
            pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
        pw = F.pad(pw, (0, full_hop * off))
        with torch.no_grad():
            plt = model.vae.encode(pw.unsqueeze(0))
        if off:
            plt = plt[..., :-off]
        prompt_dur = plt.shape[-1]

        prompt_time = prompt_dur * full_hop / sr
        dur_sec = approx_duration_from_text(
            text_norm, max_duration=max_duration - prompt_time
        )
        if prompt_text_norm:
            approx_pd = approx_duration_from_text(
                prompt_text_norm, max_duration=max_duration
            )
            ratio = np.clip(prompt_time / max(approx_pd, 0.1), 1.0, 1.5)
            dur_sec = dur_sec * ratio
        logger.info(f"Estimated duration: {dur_sec:.2f}s (prompt: {prompt_time:.2f}s)")
        duration = int(dur_sec * sr // full_hop)
        duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))

        if pbar:
            pbar.update_absolute(2, 4)

        actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(actual_seed)

        self._check_interrupt()

        try:
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_audio=prompt_audio_tensor,
                    duration=duration,
                    steps=steps,
                    cfg_strength=guidance_strength,
                    guidance_method=guidance_method,
                    seed=actual_seed,
                )

            if pbar:
                pbar.update_absolute(3, 4)

            wav = output.waveform.squeeze().detach().cpu().numpy()
            logger.info(f"Generated {len(wav) / sr:.2f}s of audio at {sr}Hz")

            result = numpy_audio_to_comfy(wav, sr)

            if pbar:
                pbar.update_absolute(4, 4)

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
