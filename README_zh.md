<div align="center">
  <h1>ComfyUI-LongCat-AudioDIT-TTS</h1>

  <p>
    ComfyUI 自定义节点
    <b><em>LongCat-AudioDiT — 基于扩散模型的零样本文本转语音</em></b>
  </p>
  <p>
    <a href="https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_(FP32)-blue' alt="HF Model FP32"></a>
    <a href="https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-bf16"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_(BF16)-orange' alt="HF Model BF16"></a>
    <a href="https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-fp8"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model_(FP8)-purple' alt="HF Model FP8"></a>
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/version-0.1.7-blue" alt="Version">
  </p>
</div>


<img width="1813" height="1166" alt="Screenshot 2026-03-30 210100" src="https://github.com/user-attachments/assets/1ba808d2-97c1-4a58-bf67-a72353c58a7a" />

---

## 概述

**LongCat-AudioDiT** 是美团推出的基于扩散模型的文本转语音模型，采用 DiT（扩散 Transformer）架构和 ODE Euler 求解器生成高质量语音。支持从参考音频进行零样本声音克隆，无需任何微调。

本 ComfyUI 封装提供原生节点化集成：
- **零样本 TTS** — 从文本输入生成语音
- **声音克隆** — 从参考音频克隆声音（建议 3-15 秒）
- **多说话人对话合成** — 使用多个克隆声音生成对话
- **24kHz 输出** — 广播级音质

---

## 特性

- **零样本声音克隆** — 从短参考音频克隆任意声音
- **多说话人 TTS** — 使用 `[speaker_N]:` 标签生成多声音对话
- **基于扩散的生成** — DiT Transformer 配合 ODE Euler 求解器生成高质量音频
- **FP8 / FP16 / BF16 / FP32 支持** — FP8 模型自动反量化为 BF16；FP16 仅支持 TTS（声音克隆需要 BF16）
- **原生 ComfyUI 集成** — AUDIO 连线输入、进度条、中断支持
- **智能自动下载** — 首次使用时自动从 HuggingFace 下载模型权重
- **智能缓存** — 可选模型缓存，运行间隙自动卸载到 CPU
- **优化的注意力机制** — 支持 SDPA、SageAttention 后端
- **自动安装依赖** — 启动时自动安装缺失的包

---

## 系统要求

- **GPU:** NVIDIA GPU，bf16 模型需要 **8GB+ 显存**，fp32 模型需要 **16GB+ 显存**
- **CPU:** 支持但速度较慢
- **MPS:** 实验性支持
- **Python:** 3.10+
- **CUDA:** 11.8+（GPU 推理需要）

---

## 模型

| 模型 | 显存 | 描述 |
|-------|------|-------------|
| **LongCat-AudioDiT-1B** | ~6-8GB | 10亿参数 (FP32) — 最小模型，显存需求低 |
| **LongCat-AudioDiT-3.5B-bf16** | ~10-14GB | BF16 量化（推荐）— 质量与显存的最佳平衡 |
| **LongCat-AudioDiT-3.5B-fp8** | ~8-12GB | FP8 量化 — 最小下载体积，加载时反量化为 BF16 |
| **LongCat-AudioDiT-3.5B** | ~20GB | FP32 原版 — 最高质量，需要更多显存 |

首次使用时自动从 HuggingFace 下载：
- [meituan-longcat/LongCat-AudioDiT-1B](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B) — 10亿参数模型
- [meituan-longcat/LongCat-AudioDiT-3.5B](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B) — 原版 FP32 模型
- [drbaph/LongCat-AudioDiT-3.5B-bf16](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-bf16) — BF16 量化版
- [drbaph/LongCat-AudioDiT-3.5B-fp8](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-fp8) — FP8 量化版

---

## 安装

<details>
<summary><b>点击展开安装方法</b></summary>

### 方法 1：ComfyUI Manager（推荐）

1. 打开 ComfyUI Manager
2. 搜索 "LongCat-AudioDiT"
3. 点击安装
4. 重启 ComfyUI

### 方法 2：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/saganaki22/ComfyUI-LongCat-AudioDIT-TTS.git
cd ComfyUI-LongCat-AudioDIT-TTS
pip install -r requirements.txt
```

</details>

---

## 快速开始

### 节点概览

| 节点 | 描述 |
|------|-------------|
| **LongCat AudioDiT TTS** | 文本转语音合成 |
| **LongCat AudioDiT Voice Clone TTS** | 从参考音频克隆声音 |
| **LongCat AudioDiT Multi-Speaker TTS** | 多说话人对话合成 |

### 基本工作流程

1. **下载模型**
   - 首次使用时自动从 HuggingFace 下载
   - 或手动放置在 `ComfyUI/models/audiodit/`

2. **文本转语音**
   - 添加 `LongCat AudioDiT TTS` 节点
   - 输入文本
   - 运行！

3. **声音克隆**
   - 添加 `LongCat AudioDiT Voice Clone TTS` 节点
   - 连接参考音频（建议 3-15 秒）
   - 可选：提供参考音频的文本转录
   - 输入要用克隆声音合成的文本
   - 运行！

4. **多说话人**
   - 添加 `LongCat AudioDiT Multi-Speaker TTS` 节点
   - 设置说话人数量
   - 为每个说话人连接参考音频
   - 在文本中使用 `[speaker_1]:`、`[speaker_2]:` 标签
   - 运行！

---

## 节点参考

### LongCat AudioDiT TTS

基础文本转语音合成。

**输入：**
- `model_path`: 模型选择（首次使用自动下载）
- `text`: 要合成的文本
- `steps`: ODE Euler 步数（4-64，默认 16）
- `guidance_strength`: CFG/APG 引导强度（0-10，默认 4.0）
- `guidance_method`: 引导方法（`cfg` 或 `apg`）
- `device`: 计算设备（`auto`、`cuda`、`cpu`、`mps`）
- `dtype`: 模型精度（`auto`、`bf16`、`fp16`、`fp32`）
- `attention`: 注意力实现（`auto`、`sdpa`、`sage_attention`、`flash_attention`）
- `seed`: 随机种子（0 = 随机）
- `keep_model_loaded`: 在运行间隙保持模型加载

**输出：**
- `audio`: 生成的语音（AUDIO）

---

### LongCat AudioDiT Voice Clone TTS

从参考音频克隆声音。

**输入：**
- 所有 TTS 的输入，加上：
- `prompt_audio`: 用于克隆的参考音频（AUDIO）
- `prompt_text`: 参考音频的文本转录（提高质量）

**输出：**
- `audio`: 克隆声音生成的语音（AUDIO）

---

### LongCat AudioDiT Multi-Speaker TTS

多说话人对话合成，支持动态说话人输入（ComfyUI v3）或固定槽位（v2 兼容）。

**输入：**
- 所有 Voice Clone TTS 的输入，加上：
- `num_speakers`: 说话人数量（2-10）
- `speaker_N_audio`: 每个说话人的参考音频
- `speaker_N_ref_text`: 每个说话人参考音频的文本转录
- `pause_after_speaker`: 说话人之间的静音时长（默认 0.4 秒）

**文本格式：**
```
[speaker_1]: 你好，我是说话人一。
[speaker_2]: 我是说话人二！
```

**输出：**
- `audio`: 生成的多说话人对话（AUDIO）

---

## 参数说明

| 参数 | 描述 | 推荐值 |
|-----------|-------------|-------------|
| **steps** | ODE Euler 求解器步数 | `16`（平衡），`32`（更高质量） |
| **guidance_strength** | 引导强度 | `4.0`（平衡），更高 = 更多引导 |
| **guidance_method** | 引导算法 | `cfg`（TTS），`apg`（声音克隆） |
| **dtype** | 模型精度 | `auto` 或 `bf16`（推荐） |
| **attention** | 注意力后端 | `auto`（默认），`sage_attention`（最快） |
| **keep_model_loaded** | 在运行间隙缓存模型 | `True` 用于重复使用 |

> **精度说明：** FP16 在 float16 中运行 Transformer，同时保持文本编码器在 BF16（UMT5 layer_norm 在 fp16 中会溢出），ODE 步数在 fp32 中累积。FP8 模型在加载时反量化为 BF16。为获得最佳效果，请使用 `auto` 或 `bf16`。

> ⚠️ **FP16 不支持声音克隆：** Voice Clone TTS 和 Multi-Speaker TTS 节点会自动将 FP16 升级为 BF16。这是必需的，因为潜在条件路径（编码参考音频）在 FP16 中会导致数值溢出，产生 NaN 值并在 ODE 求解器中级联传播，最终产生静音输出。基础 TTS 可以使用 FP16，因为它使用零潜在向量作为条件。如果您为声音克隆选择 FP16，您会看到警告，节点将自动使用 BF16。

---

## 文件结构

```
ComfyUI/
├── models/
│   └── audiodit/
│       ├── LongCat-AudioDiT-3.5B/          # FP32 模型（自动下载）
│       ├── LongCat-AudioDiT-3.5B-bf16/     # BF16 模型（自动下载）
│       └── LongCat-AudioDiT-3.5B-fp8/      # FP8 模型（自动下载）
└── custom_nodes/
    └── ComfyUI-LongCat-AudioDIT-TTS/
        ├── __init__.py                      # 节点注册 + 自动安装
        ├── pyproject.toml
        ├── requirements.txt
        ├── nodes/
        │   ├── tts_node.py                  # TTS 节点
        │   ├── voice_clone_node.py          # 声音克隆节点
        │   ├── multi_speaker_node.py         # 多说话人节点
        │   ├── loader.py                    # 模型加载 + 注意力补丁
        │   └── model_cache.py               # 缓存 + 卸载逻辑
        └── audiodit/
            ├── __init__.py                  # AutoConfig/AutoModel 注册
            ├── modeling_audiodit.py         # AudioDiT 模型实现
            ├── configuration_audiodit.py    # 配置类
            ├── fp8_linear.py                # FP8 层（未使用，保留供参考）
            └── utils.py
```

---

## 故障排除

<details>
<summary><b>点击展开故障排除指南</b></summary>

### 模型没有下载？

手动从 HuggingFace 下载：
```bash
pip install -U huggingface_hub

# 1B 模型（最小）
huggingface-cli download meituan-longcat/LongCat-AudioDiT-1B --local-dir ComfyUI/models/audiodit/LongCat-AudioDiT-1B

# BF16 模型（推荐）
huggingface-cli download drbaph/LongCat-AudioDiT-3.5B-bf16 --local-dir ComfyUI/models/audiodit/LongCat-AudioDiT-3.5B-bf16

# FP32 模型
huggingface-cli download meituan-longcat/LongCat-AudioDiT-3.5B --local-dir ComfyUI/models/audiodit/LongCat-AudioDiT-3.5B

# FP8 模型
huggingface-cli download drbaph/LongCat-AudioDiT-3.5B-fp8 --local-dir ComfyUI/models/audiodit/LongCat-AudioDiT-3.5B-fp8
```

### 节点没有加载 / 缺少依赖？

所有必需的包在首次启动时自动安装。如果节点加载失败，**重启 ComfyUI 一次** — 安装程序在节点注册之前运行。如果重启后仍然失败，手动安装：

```bash
pip install -r requirements.txt
```

### FP16 / FP8 产生静音输出？

FP8 模型在加载时反量化为 BF16 — 598 个 FP8 张量使用 `fp8_scales.json` 中的逐张量缩放因子转换。FP16 在 float16 中运行 Transformer，ODE 累积在 fp32 中进行。两者都应产生清晰的音频。

**重要：** FP16 **不支持** Voice Clone TTS 和 Multi-Speaker TTS。这些节点会自动将 FP16 升级为 BF16，因为来自参考音频的潜在条件在 FP16 中会导致数值溢出，进而导致 NaN 传播和静音输出。如果您为这些节点选择 FP16，您会看到警告消息。

### 显存不足？

- 使用 **bf16 模型**（~7GB 显存）
- 设置 `keep_model_loaded=False`
- 启用 `offload_to_cpu` — 模型在运行间隙移动到 CPU

### 模型没有卸载？

在 ComfyUI 中点击"释放模型和节点缓存"总是会完全卸载模型，无论 `keep_model_loaded` 设置如何。该设置仅控制生成运行之间的自动卸载行为。

### 音频有噪音 / 嘶嘶声？

- 确保 `dtype` 设置为 `auto` 或 `bf16` 以获得最佳质量
- VAE 音频解码器在 CUDA 上始终保持在 BF16 以保证音质
- FP16 和 FP8 虽然支持，但 `bf16` 或 `auto` 能提供最一致的结果

### 声音克隆质量差？

如果您的参考音频太响（削波）或音量不一致，声音克隆质量会受影响。在使用前将参考音频归一化到 -3dB 到 -6dB 峰值：

```python
# 快速归一化脚本
import librosa
import soundfile as sf

audio, sr = librosa.load("reference.wav", sr=24000, mono=True)
peak = audio.max()
target_peak = 0.5  # -6dB
audio = audio * (target_peak / peak)
sf.write("reference_normalized.wav", audio, sr)
```

</details>

---

## 链接

### HuggingFace
- **1B 模型:** [meituan-longcat/LongCat-AudioDiT-1B](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B)
- **原版模型 (FP32):** [meituan-longcat/LongCat-AudioDiT-3.5B](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B)
- **BF16 模型:** [drbaph/LongCat-AudioDiT-3.5B-bf16](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-bf16)
- **FP8 模型:** [drbaph/LongCat-AudioDiT-3.5B-fp8](https://huggingface.co/drbaph/LongCat-AudioDiT-3.5B-fp8)

### 源代码
- **原始仓库:** [meituan-longcat/LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT)

---

## 许可证

MIT 许可证。详见 [LICENSE](LICENSE)。
