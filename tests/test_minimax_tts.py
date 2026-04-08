"""Unit tests for the MiniMax TTS node."""

import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy dependencies so tests run without GPU / full ComfyUI stack
# ---------------------------------------------------------------------------

# torch stub — must be registered before any module that imports torch
_torch_stub = types.ModuleType("torch")
_torch_stub.__spec__ = types.SimpleNamespace(name="torch")

def _from_numpy(arr):
    m = MagicMock()
    m.float.return_value = m
    m.contiguous.return_value = m
    return m

_torch_stub.from_numpy = _from_numpy
sys.modules["torch"] = _torch_stub

# numpy is real
import numpy as np  # noqa: E402


def _make_fake_sse_response(chunks_hex: list, status_code: int = 0) -> bytes:
    """Build a fake SSE byte stream for the MiniMax TTS API."""
    lines = []
    for chunk_hex in chunks_hex:
        event = {
            "data": {"audio": chunk_hex, "status": 1},
            "base_resp": {"status_code": status_code, "status_msg": "success"},
        }
        lines.append(f"data:{json.dumps(event)}\n\n")
    lines.append("data:[DONE]\n\n")
    return "".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Import the module under test directly (bypasses package __init__.py)
# ---------------------------------------------------------------------------
import importlib.util as _ilu
from pathlib import Path as _Path

_node_path = _Path(__file__).parent.parent / "nodes" / "minimax_tts_node.py"
_spec = _ilu.spec_from_file_location("nodes.minimax_tts_node", _node_path)
_mod = _ilu.module_from_spec(_spec)
sys.modules["nodes.minimax_tts_node"] = _mod
_spec.loader.exec_module(_mod)

MiniMaxTTS = _mod.MiniMaxTTS
MINIMAX_TTS_MODELS = _mod.MINIMAX_TTS_MODELS
MINIMAX_TTS_VOICES = _mod.MINIMAX_TTS_VOICES
minimax_tts = _mod.minimax_tts
_pcm_bytes_to_numpy = _mod._pcm_bytes_to_numpy
_numpy_to_comfy = _mod._numpy_to_comfy


class TestConstants(unittest.TestCase):
    def test_models_list(self):
        self.assertIn("speech-2.8-hd", MINIMAX_TTS_MODELS)
        self.assertIn("speech-2.8-turbo", MINIMAX_TTS_MODELS)

    def test_voices_list(self):
        self.assertIn("English_Graceful_Lady", MINIMAX_TTS_VOICES)
        self.assertIn("English_Insightful_Speaker", MINIMAX_TTS_VOICES)
        self.assertEqual(len(MINIMAX_TTS_VOICES), 6)


class TestInputTypes(unittest.TestCase):
    def test_has_required_inputs(self):
        spec = MiniMaxTTS.INPUT_TYPES()
        required = spec["required"]
        self.assertIn("text", required)
        self.assertIn("model", required)
        self.assertIn("voice_id", required)
        self.assertIn("speed", required)

    def test_has_optional_inputs(self):
        spec = MiniMaxTTS.INPUT_TYPES()
        optional = spec.get("optional", {})
        self.assertIn("api_key", optional)
        self.assertIn("base_url", optional)

    def test_default_model(self):
        spec = MiniMaxTTS.INPUT_TYPES()
        _type, meta = spec["required"]["model"]
        self.assertEqual(meta["default"], "speech-2.8-hd")

    def test_default_voice(self):
        spec = MiniMaxTTS.INPUT_TYPES()
        _type, meta = spec["required"]["voice_id"]
        self.assertEqual(meta["default"], "English_Graceful_Lady")

    def test_return_types(self):
        self.assertEqual(MiniMaxTTS.RETURN_TYPES, ("AUDIO",))
        self.assertEqual(MiniMaxTTS.FUNCTION, "synthesize")


class TestMiniMaxTTSFunction(unittest.TestCase):
    """Test the minimax_tts() helper that calls the API."""

    @patch("urllib.request.urlopen")
    def test_collects_hex_audio_chunks(self, mock_urlopen):
        chunk1 = bytes([0x41, 0x42, 0x43]).hex()
        chunk2 = bytes([0x44, 0x45, 0x46]).hex()
        sse_bytes = _make_fake_sse_response([chunk1, chunk2])
        mock_urlopen.return_value.__enter__ = lambda s: io.BytesIO(sse_bytes)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        result_bytes, sr = minimax_tts("hello", "test-key")

        self.assertEqual(result_bytes, bytes([0x41, 0x42, 0x43, 0x44, 0x45, 0x46]))
        self.assertEqual(sr, 32000)

    @patch("urllib.request.urlopen")
    def test_raises_on_api_error(self, mock_urlopen):
        event = {
            "data": {"audio": "", "status": 0},
            "base_resp": {"status_code": 1004, "status_msg": "auth failed"},
        }
        sse_bytes = f"data:{json.dumps(event)}\n\n".encode()
        mock_urlopen.return_value.__enter__ = lambda s: io.BytesIO(sse_bytes)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        with self.assertRaises(RuntimeError) as ctx:
            minimax_tts("hello", "bad-key")
        self.assertIn("1004", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_raises_on_empty_audio(self, mock_urlopen):
        sse_bytes = b"data:[DONE]\n\n"
        mock_urlopen.return_value.__enter__ = lambda s: io.BytesIO(sse_bytes)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        with self.assertRaises(RuntimeError) as ctx:
            minimax_tts("hello", "test-key")
        self.assertIn("no audio", str(ctx.exception).lower())

    @patch("urllib.request.urlopen")
    def test_uses_correct_url(self, mock_urlopen):
        chunk_hex = bytes([0x00]).hex()
        sse_bytes = _make_fake_sse_response([chunk_hex])
        mock_urlopen.return_value.__enter__ = lambda s: io.BytesIO(sse_bytes)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        minimax_tts("hello", "key", base_url="https://api.minimax.io")

        req = mock_urlopen.call_args[0][0]
        self.assertIn("api.minimax.io/v1/t2a_v2", req.full_url)

    @patch("urllib.request.urlopen")
    def test_sends_correct_auth_header(self, mock_urlopen):
        chunk_hex = bytes([0x00]).hex()
        sse_bytes = _make_fake_sse_response([chunk_hex])
        mock_urlopen.return_value.__enter__ = lambda s: io.BytesIO(sse_bytes)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        minimax_tts("hello", "my-secret-key")

        req = mock_urlopen.call_args[0][0]
        self.assertIn("my-secret-key", req.get_header("Authorization"))


class TestSynthesizeMethod(unittest.TestCase):
    """Test the MiniMaxTTS.synthesize() method."""

    @patch.object(_mod, "minimax_tts")
    @patch.object(_mod, "_pcm_bytes_to_numpy")
    @patch.object(_mod, "_numpy_to_comfy")
    def test_synthesize_returns_audio_tuple(
        self, mock_to_comfy, mock_to_numpy, mock_tts
    ):
        mock_tts.return_value = (b"\x00" * 100, 32000)
        mock_to_numpy.return_value = (np.zeros(3200), 32000)
        mock_to_comfy.return_value = {"waveform": MagicMock(), "sample_rate": 32000}

        node = MiniMaxTTS()
        result = node.synthesize(
            text="Hello",
            model="speech-2.8-hd",
            voice_id="English_Graceful_Lady",
            speed=1.0,
            api_key="test-key",
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        mock_tts.assert_called_once()

    @patch.object(_mod, "minimax_tts")
    def test_synthesize_raises_on_empty_text(self, mock_tts):
        node = MiniMaxTTS()
        with self.assertRaises(ValueError) as ctx:
            node.synthesize(
                text="  ",
                model="speech-2.8-hd",
                voice_id="English_Graceful_Lady",
                speed=1.0,
                api_key="test-key",
            )
        self.assertIn("empty", str(ctx.exception).lower())
        mock_tts.assert_not_called()

    @patch.object(_mod, "minimax_tts")
    def test_synthesize_raises_without_api_key(self, mock_tts):
        import os

        node = MiniMaxTTS()
        env_backup = os.environ.pop("MINIMAX_API_KEY", None)
        try:
            with self.assertRaises(ValueError) as ctx:
                node.synthesize(
                    text="Hello",
                    model="speech-2.8-hd",
                    voice_id="English_Graceful_Lady",
                    speed=1.0,
                    api_key="",
                )
            self.assertIn("api key", str(ctx.exception).lower())
        finally:
            if env_backup is not None:
                os.environ["MINIMAX_API_KEY"] = env_backup

    @patch.object(_mod, "minimax_tts")
    @patch.object(_mod, "_pcm_bytes_to_numpy")
    @patch.object(_mod, "_numpy_to_comfy")
    def test_synthesize_uses_env_api_key(self, mock_to_comfy, mock_to_numpy, mock_tts):
        import os

        mock_tts.return_value = (b"\x00" * 10, 32000)
        mock_to_numpy.return_value = (np.zeros(100), 32000)
        mock_to_comfy.return_value = {"waveform": MagicMock(), "sample_rate": 32000}

        node = MiniMaxTTS()
        os.environ["MINIMAX_API_KEY"] = "env-key-123"
        try:
            node.synthesize(
                text="Hello",
                model="speech-2.8-hd",
                voice_id="English_Graceful_Lady",
                speed=1.0,
                api_key="",
            )
        finally:
            del os.environ["MINIMAX_API_KEY"]

        call_kwargs = mock_tts.call_args[1]
        self.assertEqual(call_kwargs.get("api_key"), "env-key-123")


if __name__ == "__main__":
    unittest.main()

