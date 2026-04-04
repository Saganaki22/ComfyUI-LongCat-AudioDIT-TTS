"""ComfyUI custom nodes for LongCat-AudioDiT TTS.

Provides three nodes:
  - LongCatTTS            -- text -> speech, zero-shot diffusion TTS
  - LongCatVoiceCloneTTS  -- reference audio + text -> voice-cloned speech
  - LongCatMultiSpeakerTTS -- multi-speaker conversation with cloned voices

Required pip packages are auto-installed on startup.
Model weights are auto-downloaded from HuggingFace on first inference.
"""

__version__ = "0.1.9"

import importlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).parent.resolve()

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logger = logging.getLogger("LongCatAudioDiT")
logger.propagate = False

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[LongCatAudioDiT] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _find_pip() -> list[str]:
    return [sys.executable, "-m", "pip"]


def _pip_install(spec: str) -> bool:
    cmd = _find_pip() + ["install"] + spec.split()
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"Successfully installed: {spec}")
            importlib.invalidate_caches()
            return True
        logger.error(f"pip install failed for '{spec}':\n{result.stderr.strip()}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"pip install timed out for: {spec}")
        return False
    except Exception as e:
        logger.error(f"pip install error for '{spec}': {e}")
        return False


def _restore_torch() -> None:
    try:
        import torch

        version = torch.__version__
        if "+cu" in version:
            logger.info(f"torch {version} is a CUDA build -- no restore needed.")
            return
        logger.warning(f"torch {version} is NOT a CUDA build. Restoring CUDA torch...")
    except ImportError:
        return

    cuda_tag = "cu128"
    try:
        import subprocess as sp

        sp.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        cuda_tag = "cu128"
    except Exception:
        pass

    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    logger.info(f"Restoring torch with: --index-url {index_url}")
    _pip_install(f"torch torchaudio --index-url {index_url}")


_REQUIRED = [
    ("numpy", "numpy"),
    ("soundfile", "soundfile"),
    ("transformers", "transformers>=4.45.2"),
    ("einops", "einops>=0.7.0"),
    ("librosa", "librosa>=0.10.1"),
    ("safetensors", "safetensors>=0.4.0"),
    ("huggingface_hub", "huggingface-hub"),
]


def _ensure_dependencies() -> bool:
    all_ok = True
    any_installed = False
    failed_specs: list[str] = []

    for import_name, pip_spec in _REQUIRED:
        try:
            __import__(import_name)
        except ImportError as e:
            logger.warning(f"'{import_name}' not found -- auto-installing: {pip_spec}")
            if _pip_install(pip_spec):
                any_installed = True
                try:
                    __import__(import_name)
                except ImportError:
                    logger.error(
                        f"Installed '{pip_spec}' but still can't import '{import_name}'. Restart ComfyUI."
                    )
                    failed_specs.append(pip_spec)
                    all_ok = False
            else:
                failed_specs.append(pip_spec)
                all_ok = False

    if any_installed:
        _restore_torch()

    if not all_ok:
        install_cmds = "\n".join(
            f"  {sys.executable} -m pip install {s}" for s in failed_specs
        )
        logger.error(
            "Auto-install failed for some packages. Install manually:\n" + install_cmds
        )
    return all_ok


NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

if _ensure_dependencies():
    try:
        from .nodes.loader import _register_folder

        _register_folder()

        from .nodes.tts_node import LongCatTTS
        from .nodes.voice_clone_node import LongCatVoiceCloneTTS
        from .nodes.multi_speaker_node import LongCatMultiSpeakerTTS

        NODE_CLASS_MAPPINGS = {
            "LongCatTTS": LongCatTTS,
            "LongCatVoiceCloneTTS": LongCatVoiceCloneTTS,
            "LongCatMultiSpeakerTTS": LongCatMultiSpeakerTTS,
        }

        NODE_DISPLAY_NAME_MAPPINGS = {
            "LongCatTTS": "LongCat AudioDiT TTS",
            "LongCatVoiceCloneTTS": "LongCat AudioDiT Voice Clone TTS",
            "LongCatMultiSpeakerTTS": "LongCat AudioDiT Multi-Speaker TTS",
        }

        logger.info(
            f"Registered {len(NODE_CLASS_MAPPINGS)} nodes "
            f"(v{__version__}): {', '.join(NODE_DISPLAY_NAME_MAPPINGS.values())}"
        )

    except Exception as e:
        logger.error(f"Failed to register nodes: {e}", exc_info=True)
else:
    logger.warning(
        "LongCat-AudioDiT nodes not registered -- fix dependency errors and restart."
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
