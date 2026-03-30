<div align="center">
  <h1>ComfyUI-LongCat-AudioDIT-TTS</h1>

  <p>
    ComfyUI custom nodes for
    <b><em>LongCat-AudioDiT — Diffusion-based Zero-Shot Text-to-Speech</em></b>
  </p>
  <p>
    <a href="https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_(FP32)-blue' alt="HF Model FP32"></a>
    <a href="https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-bf16"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_(BF16)-orange' alt="HF Model BF16"></a>
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/version-0.1.1-blue" alt="Version">
  </p>
</div>


<img width="1813" height="1166" alt="Screenshot 2026-03-30 210100" src="https://github.com/user-attachments/assets/1ba808d2-97c1-4a58-bf67-a72353c58a7a" />

---

## Overview

**LongCat-AudioDiT** is a diffusion-based text-to-speech model by Meituan that generates high-quality speech audio using a DiT (Diffusion Transformer) architecture with an ODE Euler solver. It supports zero-shot voice cloning from reference audio without any fine-tuning.

This ComfyUI wrapper provides native node-based integration with:
- **Zero-shot TTS** from text input
- **Voice cloning** from reference audio (3-15 seconds recommended)
- **Multi-speaker conversation synthesis** with multiple cloned voices
- **24kHz output** at broadcast quality

---

## Features

- **Zero-Shot Voice Cloning** — Clone any voice from a short reference audio clip
- **Multi-Speaker TTS** — Generate conversations with multiple cloned voices using `[speaker_N]:` tags
- **Diffusion-Based Generation** — DiT transformer with ODE Euler solver for high-quality audio
- **FP8 / FP16 / BF16 / FP32 Support** — FP8 models are auto-dequantized to BF16; FP16 runs the transformer in fp16 with fp32 ODE accumulation
- **Native ComfyUI Integration** — AUDIO noodle inputs, progress bars, interruption support
- **Smart Auto-Download** — Model weights auto-downloaded from HuggingFace on first use
- **Smart Caching** — Optional model caching with CPU offload between runs
- **Optimized Attention** — Support for SDPA, SageAttention backends
- **Auto-Install Dependencies** — Missing packages are installed automatically on startup

---

## Requirements

- **GPU:** NVIDIA GPU with **8GB+ VRAM** for bf16 model, **16GB+ VRAM** for fp32 model
- **CPU:** Supported but slow
- **MPS:** Experimental
- **Python:** 3.10+
- **CUDA:** 11.8+ (for GPU inference)

---

## Models

| Model | VRAM | Description |
|-------|------|-------------|
| **LongCat-AudioDiT-3.5B-bf16** | ~7GB | BF16 quantized (recommended) — best balance of quality and VRAM |
| **LongCat-AudioDiT-3.5B-fp8** | ~4GB | FP8 quantized — smallest download, dequantized to BF16 at load time |
| **LongCat-AudioDiT-3.5B** | ~14GB | FP32 original — highest quality, requires more VRAM |

Models are auto-downloaded from HuggingFace on first use:
- [meituan-longcat/LongCat-AudioDiT-3.5B](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B) — original FP32 model
- [drbaph/LongCat-AudioDiT-3.5B-bf16](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-bf16) — BF16 quantized
- [drbaph/LongCat-AudioDiT-3.5B-fp8](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-fp8) — FP8 quantized

---

## Installation

<details>
<summary><b>Click to expand installation methods</b></summary>

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "LongCat-AudioDiT"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/saganaki22/ComfyUI-LongCat-AudioDIT-TTS.git
cd ComfyUI-LongCat-AudioDIT-TTS
pip install -r requirements.txt
```

</details>

---

## Quick Start

### Node Overview

| Node | Description |
|------|-------------|
| **LongCat AudioDiT TTS** | Text-to-speech synthesis |
| **LongCat AudioDiT Voice Clone TTS** | Voice cloning from reference audio |
| **LongCat AudioDiT Multi-Speaker TTS** | Multi-speaker conversation synthesis |

### Basic Workflow

1. **Download Model**
   - Models are auto-downloaded from HuggingFace on first use
   - Or manually place in `ComfyUI/models/audiodit/`

2. **Text-to-Speech**
   - Add `LongCat AudioDiT TTS` node
   - Enter your text
   - Run!

3. **Voice Cloning**
   - Add `LongCat AudioDiT Voice Clone TTS` node
   - Connect reference audio (3-15 seconds recommended)
   - Optionally provide transcript of the reference audio
   - Enter text to synthesize in the cloned voice
   - Run!

4. **Multi-Speaker**
   - Add `LongCat AudioDiT Multi-Speaker TTS` node
   - Set number of speakers
   - Connect reference audio for each speaker
   - Use `[speaker_1]:`, `[speaker_2]:` tags in text
   - Run!

---

## Node Reference

### LongCat AudioDiT TTS

Basic text-to-speech synthesis.

**Inputs:**
- `model_path`: Model selection (auto-downloads on first use)
- `text`: Text to synthesize
- `steps`: Number of ODE Euler steps (4-64, default 16)
- `guidance_strength`: CFG/APG guidance strength (0-10, default 4.0)
- `guidance_method`: Guidance method (`cfg` or `apg`)
- `device`: Compute device (`auto`, `cuda`, `cpu`, `mps`)
- `dtype`: Model precision (`auto`, `bf16`, `fp16`, `fp32`)
- `attention`: Attention implementation (`auto`, `sdpa`, `sage_attention`, `flash_attention`)
- `seed`: Random seed (0 = random)
- `keep_model_loaded`: Keep model in memory between runs

**Outputs:**
- `audio`: Generated speech (AUDIO)

---

### LongCat AudioDiT Voice Clone TTS

Voice cloning from reference audio.

**Inputs:**
- All inputs from TTS, plus:
- `prompt_audio`: Reference audio to clone (AUDIO)
- `prompt_text`: Transcript of reference audio (improves quality)

**Outputs:**
- `audio`: Generated speech in cloned voice (AUDIO)

---

### LongCat AudioDiT Multi-Speaker TTS

Multi-speaker conversation synthesis with dynamic speaker inputs (ComfyUI v3) or fixed slots (v2 fallback).

**Inputs:**
- All inputs from Voice Clone TTS, plus:
- `num_speakers`: Number of speakers (2-10)
- `speaker_N_audio`: Reference audio for each speaker
- `speaker_N_ref_text`: Transcript for each speaker's reference audio
- `pause_after_speaker`: Seconds of silence between speakers (default 0.4)

**Text Format:**
```
[speaker_1]: Hello, I'm speaker one.
[speaker_2]: And I'm speaker two!
```

**Outputs:**
- `audio`: Generated multi-speaker conversation (AUDIO)

---

## Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **steps** | ODE Euler solver steps | `16` (balanced), `32` (higher quality) |
| **guidance_strength** | Guidance scale | `4.0` (balanced), higher = more guidance |
| **guidance_method** | Guidance algorithm | `cfg` (TTS), `apg` (voice clone) |
| **dtype** | Model precision | `auto` or `bf16` (recommended) |
| **attention** | Attention backend | `auto` (default), `sage_attention` (fastest) |
| **keep_model_loaded** | Cache model between runs | `True` for repeated use |

> **Note on precision:** FP16 runs the transformer in float16 while keeping the text encoder in BF16 (UMT5 layer_norm overflows in fp16) and accumulating ODE steps in fp32. FP8 models are dequantized to BF16 during loading. For best results, use `auto` or `bf16`.

---

## File Structure

```
ComfyUI/
├── models/
│   └── audiodit/
│       ├── LongCat-AudioDiT-3.5B/          # FP32 model (auto-downloaded)
│       ├── LongCat-AudioDiT-3.5B-bf16/     # BF16 model (auto-downloaded)
│       └── LongCat-AudioDiT-3.5B-fp8/      # FP8 model (auto-downloaded)
└── custom_nodes/
    └── ComfyUI-LongCat-AudioDIT-TTS/
        ├── __init__.py                      # Node registration + auto-install
        ├── pyproject.toml
        ├── requirements.txt
        ├── nodes/
        │   ├── tts_node.py                  # TTS node
        │   ├── voice_clone_node.py          # Voice clone node
        │   ├── multi_speaker_node.py         # Multi-speaker node
        │   ├── loader.py                    # Model loading + attention patching
        │   └── model_cache.py               # Caching + offload logic
        └── audiodit/
            ├── __init__.py                  # AutoConfig/AutoModel registration
            ├── modeling_audiodit.py         # AudioDiT model implementation
            ├── configuration_audiodit.py    # Config class
            ├── fp8_linear.py                # FP8 layer (unused, kept for reference)
            └── utils.py
```

---

## Troubleshooting

<details>
<summary><b>Click to expand troubleshooting guide</b></summary>

### Models Not Downloading?

Manually download from HuggingFace:
```bash
pip install -U huggingface_hub

# BF16 model (recommended)
huggingface-cli download drbaph/LongCat-AudioDiT-3.5B-bf16 --local-dir ComfyUI/models/audiodit/LongCat-AudioDiT-3.5B-bf16

# FP32 model
huggingface-cli download meituan-longcat/LongCat-AudioDiT-3.5B --local-dir ComfyUI/models/audiodit/LongCat-AudioDiT-3.5B

# FP8 model
huggingface-cli download drbaph/LongCat-AudioDiT-3.5B-fp8 --local-dir ComfyUI/models/audiodit/LongCat-AudioDiT-3.5B-fp8
```

### Nodes Not Loading / Missing Dependencies?

All required packages are auto-installed on first startup. If the node fails to load, **restart ComfyUI once** — the installer runs before nodes register. If it still fails after a restart, install manually:

```bash
pip install -r requirements.txt
```

### FP16 / FP8 Produces Silent Output?

FP8 models are dequantized to BF16 during loading — the 598 FP8 tensors are converted using per-tensor scales from `fp8_scales.json`. FP16 dtype runs the transformer in float16 with fp32 ODE accumulation. Both should produce clean audio. If you hear buzzing, ensure you're using the latest version of this node pack.

### Out of Memory?

- Use the **bf16 model** (~7GB VRAM)
- Set `keep_model_loaded=False`
- Enable `offload_to_cpu` — model moves to CPU between runs

### Model Not Unloading?

Clicking "Free model and node cache" in ComfyUI always fully unloads the model regardless of `keep_model_loaded`. That setting only controls auto-offload behavior between generation runs.

### Buzzing / Hissing Audio?

- Ensure `dtype` is set to `auto` or `bf16` for best quality
- The VAE audio decoder is always kept in BF16 on CUDA for audio quality
- FP16 and FP8 are supported but `bf16` or `auto` gives the most consistent results

</details>

---

## Links

### HuggingFace
- **Original Model (FP32):** [meituan-longcat/LongCat-AudioDiT-3.5B](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B)
- **BF16 Model:** [drbaph/LongCat-AudioDiT-3.5B-bf16](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-bf16)
- **FP8 Model:** [drbaph/LongCat-AudioDiT-3.5B-fp8](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-fp8)

### Source
- **Original Repository:** [meituan-longcat/LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
