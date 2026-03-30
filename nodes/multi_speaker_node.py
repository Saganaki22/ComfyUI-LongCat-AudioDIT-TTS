"""LongCat-AudioDiT Multi-Speaker TTS node.

Uses the ComfyUI v3 IO API (IO.ComfyNode + DynamicCombo) so that the
speaker_N_audio / speaker_N_ref_text inputs appear and disappear as the
user changes num_speakers.
"""

import logging
import re
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

try:
    from comfy_api.latest import IO
    _V3 = True
except ImportError:
    _V3 = False

logger = logging.getLogger("LongCatAudioDiT")

MAX_SPEAKERS = 10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _speaker_inputs(count: int) -> list:
    """Return IO input descriptors for `count` speakers (1-indexed for UI)."""
    inputs = []
    for i in range(1, count + 1):
        inputs.append(
            IO.Audio.Input(
                f"speaker_{i}_audio",
                optional=True,
                tooltip=(
                    f"Reference audio for speaker {i} (3-15 seconds). "
                    f"Use [speaker_{i}]: in your text for this voice."
                ),
            )
        )
        inputs.append(
            IO.String.Input(
                f"speaker_{i}_ref_text",
                multiline=False,
                default="",
                optional=True,
                tooltip=(
                    f"Transcript of speaker {i}'s reference audio. "
                    "Improves clone quality significantly."
                ),
            )
        )
    return inputs


def _parse_dialogue_lines(text: str):
    """
    Parse multi-speaker text into a list of (speaker_idx_0based, line_text) tuples.

    Recognizes the form:
        [speaker_1]: Hello world
        [speaker_2]: Hi there

    Lines that do not start with a speaker tag are silently dropped.

    Returns: list of (int, str) — (0-based speaker index, text for that turn)
    """
    tag_re = re.compile(r'^\s*\[speaker_(\d+)\]:\s*(.*)$')

    lines = text.splitlines()
    turns = []
    current_speaker = None
    current_parts = []

    for raw in lines:
        m = tag_re.match(raw)
        if m:
            # Flush previous turn
            if current_speaker is not None and current_parts:
                turns.append((current_speaker, " ".join(current_parts).strip()))
            # Start new turn
            current_speaker = int(m.group(1)) - 1  # convert to 0-based
            current_parts = [m.group(2)] if m.group(2).strip() else []
        else:
            stripped = raw.strip()
            if stripped and current_speaker is not None:
                current_parts.append(stripped)

    # Flush last turn
    if current_speaker is not None and current_parts:
        turns.append((current_speaker, " ".join(current_parts).strip()))

    return turns


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


# ---------------------------------------------------------------------------
# V3 node (DynamicCombo — inputs update when num_speakers changes)
# ---------------------------------------------------------------------------

if _V3:
    class LongCatMultiSpeakerTTS(IO.ComfyNode):
        """
        LongCat-AudioDiT Multi-Speaker TTS.
        Synthesizes a conversation with multiple cloned voices.
        Change num_speakers to show/hide speaker reference audio inputs.
        Use [speaker_1]:, [speaker_2]:, ... in the text.
        """

        @classmethod
        def define_schema(cls) -> IO.Schema:
            model_names = get_model_names()

            speaker_options = [
                IO.DynamicCombo.Option(
                    key=str(n),
                    inputs=_speaker_inputs(n),
                )
                for n in range(2, MAX_SPEAKERS + 1)
            ]

            return IO.Schema(
                node_id="LongCatMultiSpeakerTTS",
                display_name="LongCat AudioDiT Multi-Speaker TTS",
                category="LongCat-AudioDiT",
                description=(
                    "LongCat-AudioDiT Multi-Speaker TTS. Synthesizes a "
                    "conversation between multiple cloned voices. "
                    "Connect reference audio clips and use [speaker_N]: tags in text."
                ),
                inputs=[
                    IO.Combo.Input(
                        "model_path",
                        options=model_names,
                        tooltip=(
                            "LongCat-AudioDiT model. "
                            "Models are stored in ComfyUI/models/audiodit/"
                        ),
                    ),
                    IO.String.Input(
                        "text",
                        multiline=True,
                        default=(
                            "[speaker_1]: Hello, I'm speaker one.\n"
                            "[speaker_2]: And I'm speaker two!"
                        ),
                        tooltip=(
                            "Multi-speaker text. Use [speaker_1]:, "
                            "[speaker_2]:, ... to assign lines to each speaker."
                        ),
                    ),
                    IO.Int.Input(
                        "steps",
                        default=16, min=4, max=64, step=1,
                        tooltip="Number of ODE Euler steps. More = better quality but slower.",
                    ),
                    IO.Float.Input(
                        "guidance_strength",
                        default=4.0, min=0.0, max=10.0, step=0.5,
                        tooltip="CFG/APG guidance strength.",
                    ),
                    IO.Combo.Input(
                        "guidance_method",
                        options=["cfg", "apg"],
                        default="apg",
                        tooltip="Guidance method. 'apg' recommended for voice cloning.",
                    ),
                    IO.Combo.Input(
                        "device",
                        options=["auto", "cuda", "cpu", "mps"],
                        default="auto",
                        tooltip="Compute device. 'auto' picks CUDA > MPS > CPU.",
                    ),
                    IO.Combo.Input(
                        "dtype",
                        options=["auto", "bf16", "fp16", "fp32"],
                        default="auto",
                        tooltip="Model dtype. 'auto' picks bf16 for CUDA.",
                    ),
                    IO.Combo.Input(
                        "attention",
                        options=["auto", "sdpa", "sage_attention", "flash_attention"],
                        default="auto",
                        tooltip="Attention implementation.",
                    ),
                    IO.Int.Input(
                        "seed",
                        default=0, min=0, max=2**31 - 1,
                        tooltip="Random seed. 0 = random.",
                    ),
                    IO.Boolean.Input(
                        "keep_model_loaded",
                        default=True,
                        tooltip=(
                            "Keep model loaded between runs. "
                            "Model is automatically offloaded to CPU after generation."
                        ),
                    ),
                    IO.Float.Input(
                        "pause_after_speaker",
                        default=0.4, min=0.0, max=2.0, step=0.1,
                        tooltip="Seconds of silence to add after each speaker turn.",
                    ),
                    IO.DynamicCombo.Input(
                        "num_speakers",
                        options=speaker_options,
                        display_name="Number of Speakers",
                        tooltip=(
                            f"How many speakers (2-{MAX_SPEAKERS}). "
                            "Changing this shows/hides speaker audio inputs."
                        ),
                    ),
                ],
                outputs=[
                    IO.Audio.Output(display_name="audio"),
                ],
            )

        @classmethod
        def execute(
            cls,
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
            pause_after_speaker: float,
            num_speakers: dict,
        ) -> "IO.NodeOutput":
            cancel_event.clear()
            _check_interrupt()

            if not text.strip():
                raise ValueError("Text cannot be empty.")

            model, tokenizer = _get_model(model_path, device, dtype, attention, keep_model_loaded)

            # num_speakers is a dict from DynamicCombo:
            # {"num_speakers": "3", "speaker_1_audio": ..., "speaker_1_ref_text": ..., ...}
            n = int(num_speakers["num_speakers"])

            sr = model.config.sampling_rate
            full_hop = model.config.latent_hop
            max_duration = model.config.max_wav_duration

            # Build per-speaker reference audio tensors
            references = {}  # {0-based idx: (audio_tensor, ref_text)}
            missing = []
            for i in range(1, n + 1):
                speaker_audio = num_speakers.get(f"speaker_{i}_audio")
                speaker_ref_text = num_speakers.get(f"speaker_{i}_ref_text") or ""

                if speaker_audio is None:
                    missing.append(i)
                else:
                    logger.info(f"Loading reference audio for speaker {i}...")
                    ref_tensor = comfy_audio_to_tensor(speaker_audio, sr).to(model.device)
                    references[i - 1] = (ref_tensor, speaker_ref_text.strip())

            if missing:
                missing_str = ", ".join(f"speaker_{i}_audio" for i in missing)
                raise ValueError(
                    f"Reference audio required for all speakers. "
                    f"Missing: {missing_str}. "
                    f"Please connect reference audio clips to each speaker input."
                )

            _check_interrupt()

            # Parse dialogue into individual (speaker_0based, line_text) turns
            dialogue_lines = _parse_dialogue_lines(text)
            if not dialogue_lines:
                raise ValueError(
                    "No speaker lines found. Use [speaker_1]: text format."
                )

            logger.info(
                f"Multi-speaker TTS ({n} speakers): {len(dialogue_lines)} lines — "
                f"generating each line independently then concatenating."
            )

            audio_turns = []
            total_steps = len(dialogue_lines)
            pbar = ProgressBar(total_steps) if _PBAR else None

            try:
                for line_idx, (speaker_idx, line_text) in enumerate(dialogue_lines):
                    _check_interrupt()

                    if speaker_idx not in references:
                        raise ValueError(
                            f"Line {line_idx + 1} uses speaker index {speaker_idx + 1} "
                            f"but no reference audio was provided for that speaker."
                        )

                    ref_audio, ref_text = references[speaker_idx]

                    logger.info(
                        f"  Line {line_idx + 1}/{len(dialogue_lines)} "
                        f"[speaker_{speaker_idx + 1}]: {line_text[:60]}"
                        f"{'...' if len(line_text) > 60 else ''}"
                    )

                    # Prepare text with reference text prefix for voice cloning
                    line_norm = normalize_text(line_text)
                    ref_norm = normalize_text(ref_text) if ref_text else ""
                    full_text = f"{ref_norm} {line_norm}" if ref_norm else line_norm

                    inputs = tokenizer([full_text], padding="longest", return_tensors="pt")
                    input_ids = inputs.input_ids.to(model.device)
                    attention_mask = inputs.attention_mask.to(model.device)

                    # Estimate duration for this line
                    dur_sec = approx_duration_from_text(line_norm, max_duration=max_duration)
                    duration = int(dur_sec * sr // full_hop)

                    # Encode prompt audio
                    off = 3
                    pw = ref_audio.clone()
                    if pw.shape[-1] % full_hop != 0:
                        pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
                    pw = F.pad(pw, (0, full_hop * off))
                    with torch.no_grad():
                        plt = model.vae.encode(pw.unsqueeze(0))
                    if off:
                        plt = plt[..., :-off]
                    prompt_dur = plt.shape[-1]

                    total_dur = min(duration + prompt_dur, int(max_duration * sr // full_hop))

                    # Seed for this line
                    actual_seed = seed + line_idx if seed != 0 else torch.randint(0, 2**31, (1,)).item() + line_idx
                    torch.manual_seed(actual_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(actual_seed)

                    with torch.no_grad():
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            prompt_audio=ref_audio.unsqueeze(0),
                            duration=total_dur,
                            steps=steps,
                            cfg_strength=guidance_strength,
                            guidance_method=guidance_method,
                            seed=actual_seed,
                        )

                    wav = output.waveform.squeeze().detach().cpu().numpy()
                    audio_turns.append(wav)

                    if pbar:
                        pbar.update_absolute(line_idx + 1, total_steps)

                # Concatenate all turns with optional silence between them
                if pause_after_speaker > 0:
                    silence_samples = int(pause_after_speaker * sr)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    audio_out = audio_turns[0]
                    for turn in audio_turns[1:]:
                        audio_out = np.concatenate([audio_out, silence, turn], axis=0)
                else:
                    audio_out = np.concatenate(audio_turns, axis=0)

                logger.info(f"Generated {len(audio_out) / sr:.2f}s of audio at {sr}Hz")
                result = numpy_audio_to_comfy(audio_out, sr)

            finally:
                if not keep_model_loaded:
                    unload_model()
                else:
                    offload_model_to_cpu()

            return IO.NodeOutput(result)


# ---------------------------------------------------------------------------
# V2 fallback (old INPUT_TYPES API)
# ---------------------------------------------------------------------------

else:
    class LongCatMultiSpeakerTTS:  # type: ignore[no-redef]
        """
        LongCat-AudioDiT Multi-Speaker TTS (legacy fallback — upgrade ComfyUI
        to 0.8.1+ for dynamic speaker inputs).
        """

        @classmethod
        def INPUT_TYPES(cls):
            model_names = get_model_names()
            optional_inputs = {}
            for i in range(1, MAX_SPEAKERS + 1):
                optional_inputs[f"speaker_{i}_audio"] = ("AUDIO", {
                    "tooltip": (
                        f"Reference audio for speaker {i}. "
                        f"Use [speaker_{i}]: in text."
                    ),
                })
                optional_inputs[f"speaker_{i}_ref_text"] = ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": f"Transcript of speaker {i}'s reference audio.",
                })

            return {
                "required": {
                    "model_path": (model_names, {
                        "tooltip": "LongCat-AudioDiT model.",
                    }),
                    "text": ("STRING", {
                        "multiline": True,
                        "default": (
                            "[speaker_1]: Hello, I'm speaker one.\n"
                            "[speaker_2]: And I'm speaker two!"
                        ),
                    }),
                    "num_speakers": ("INT", {
                        "default": 2, "min": 2, "max": MAX_SPEAKERS, "step": 1,
                        "tooltip": f"Number of active speakers (2-{MAX_SPEAKERS}).",
                    }),
                    "steps": ("INT", {
                        "default": 16, "min": 4, "max": 64, "step": 1,
                        "tooltip": "Number of ODE Euler steps.",
                    }),
                    "guidance_strength": ("FLOAT", {
                        "default": 4.0, "min": 0.0, "max": 10.0, "step": 0.5,
                        "tooltip": "CFG/APG guidance strength.",
                    }),
                    "guidance_method": (["cfg", "apg"], {
                        "default": "apg",
                        "tooltip": "Guidance method. 'apg' recommended for voice cloning.",
                    }),
                    "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                    "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                    "attention": (["auto", "sdpa", "sage_attention", "flash_attention"], {
                        "default": "auto",
                    }),
                    "seed": ("INT", {
                        "default": 0, "min": 0, "max": 2**31 - 1,
                        "tooltip": "Random seed. 0 = random.",
                    }),
                    "pause_after_speaker": ("FLOAT", {
                        "default": 0.4, "min": 0.0, "max": 5.0, "step": 0.1,
                        "tooltip": "Seconds of silence after each speaker's turn.",
                    }),
                    "keep_model_loaded": ("BOOLEAN", {"default": True}),
                },
                "optional": optional_inputs,
            }

        RETURN_TYPES = ("AUDIO",)
        RETURN_NAMES = ("audio",)
        FUNCTION = "generate"
        CATEGORY = "LongCat-AudioDiT"
        DESCRIPTION = "LongCat-AudioDiT Multi-Speaker TTS."

        def generate(
            self,
            model_path,
            text,
            num_speakers,
            steps,
            guidance_strength,
            guidance_method,
            device,
            dtype,
            attention,
            seed,
            pause_after_speaker,
            keep_model_loaded,
            **kwargs,
        ):
            cancel_event.clear()
            _check_interrupt()
            if not text.strip():
                raise ValueError("Text cannot be empty.")

            model, tokenizer = _get_model(model_path, device, dtype, attention, keep_model_loaded)

            sr = model.config.sampling_rate
            full_hop = model.config.latent_hop
            max_duration = model.config.max_wav_duration

            references = {}
            missing = []
            for i in range(1, num_speakers + 1):
                speaker_audio = kwargs.get(f"speaker_{i}_audio")
                speaker_ref_text = kwargs.get(f"speaker_{i}_ref_text") or ""
                if speaker_audio is None:
                    missing.append(i)
                else:
                    logger.info(f"Loading reference audio for speaker {i}...")
                    ref_tensor = comfy_audio_to_tensor(speaker_audio, sr).to(model.device)
                    references[i - 1] = (ref_tensor, speaker_ref_text.strip())

            if missing:
                missing_str = ", ".join(f"speaker_{i}_audio" for i in missing)
                raise ValueError(f"Missing reference audio: {missing_str}")

            _check_interrupt()

            dialogue_lines = _parse_dialogue_lines(text)
            if not dialogue_lines:
                raise ValueError("No speaker lines found. Use [speaker_1]: text format.")

            logger.info(
                f"Multi-speaker TTS ({num_speakers} speakers): "
                f"{len(dialogue_lines)} lines — generating each line independently."
            )

            audio_turns = []
            total_steps = len(dialogue_lines)
            pbar = ProgressBar(total_steps) if _PBAR else None

            try:
                for line_idx, (speaker_idx, line_text) in enumerate(dialogue_lines):
                    _check_interrupt()

                    if speaker_idx not in references:
                        raise ValueError(
                            f"Line {line_idx + 1} uses speaker {speaker_idx + 1} "
                            f"but no reference audio provided."
                        )

                    ref_audio, ref_text = references[speaker_idx]

                    logger.info(
                        f"  Line {line_idx + 1}/{len(dialogue_lines)} "
                        f"[speaker_{speaker_idx + 1}]: {line_text[:60]}"
                        f"{'...' if len(line_text) > 60 else ''}"
                    )

                    line_norm = normalize_text(line_text)
                    ref_norm = normalize_text(ref_text) if ref_text else ""
                    full_text = f"{ref_norm} {line_norm}" if ref_norm else line_norm

                    inputs = tokenizer([full_text], padding="longest", return_tensors="pt")
                    input_ids = inputs.input_ids.to(model.device)
                    attention_mask = inputs.attention_mask.to(model.device)

                    dur_sec = approx_duration_from_text(line_norm, max_duration=max_duration)
                    duration = int(dur_sec * sr // full_hop)

                    off = 3
                    pw = ref_audio.clone()
                    if pw.shape[-1] % full_hop != 0:
                        pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
                    pw = F.pad(pw, (0, full_hop * off))
                    with torch.no_grad():
                        plt = model.vae.encode(pw.unsqueeze(0))
                    if off:
                        plt = plt[..., :-off]
                    prompt_dur = plt.shape[-1]

                    total_dur = min(duration + prompt_dur, int(max_duration * sr // full_hop))

                    actual_seed = seed + line_idx if seed != 0 else torch.randint(0, 2**31, (1,)).item() + line_idx
                    torch.manual_seed(actual_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(actual_seed)

                    with torch.no_grad():
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            prompt_audio=ref_audio.unsqueeze(0),
                            duration=total_dur,
                            steps=steps,
                            cfg_strength=guidance_strength,
                            guidance_method=guidance_method,
                            seed=actual_seed,
                        )

                    wav = output.waveform.squeeze().detach().cpu().numpy()
                    audio_turns.append(wav)

                    if pbar:
                        pbar.update_absolute(line_idx + 1, total_steps)

                if pause_after_speaker > 0:
                    silence_samples = int(pause_after_speaker * sr)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    audio_out = audio_turns[0]
                    for turn in audio_turns[1:]:
                        audio_out = np.concatenate([audio_out, silence, turn], axis=0)
                else:
                    audio_out = np.concatenate(audio_turns, axis=0)

                logger.info(f"Generated {len(audio_out) / sr:.2f}s of audio at {sr}Hz")
                result = numpy_audio_to_comfy(audio_out, sr)

            finally:
                if not keep_model_loaded:
                    unload_model()
                else:
                    offload_model_to_cpu()

            return (result,)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_model(model_path, device, dtype, attention, keep_loaded=False):
    # Voice clone requires bf16 minimum - fp16 causes NaN in latent conditioning path
    if dtype == "fp16":
        logger.warning(
            "FP16 is not supported for voice cloning - the latent conditioning path "
            "causes numerical instability. Automatically upgrading to BF16."
        )
        dtype = "bf16"

    model_name = _strip_auto_download_suffix(model_path)
    key = get_cache_key(model_path, device, dtype, attention)
    cached_model, cached_tokenizer, cached_key = get_cached_model()

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


def _check_interrupt():
    if _MM:
        try:
            mm.throw_exception_if_processing_interrupted()
        except Exception:
            cancel_event.set()
            raise
