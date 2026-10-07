import json
import subprocess
from pathlib import Path

import pytest

from meika_vox.media import probe_media


def test_probe_media_parses_ffprobe_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "bit_rate": "128000",
            }
        ],
        "format": {
            "duration": "12.345",
            "bit_rate": "130000",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        },
    }

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, json.dumps(payload), "")

    monkeypatch.setattr("meika_vox.media.subprocess.run", fake_run)
    result = probe_media(Path("fixture.m4a"))

    assert result.duration_ms == 12_345
    assert result.codec == "aac"
    assert result.sample_rate == 48_000
    assert result.channels == 2
    assert result.bitrate == 128_000
    assert result.format_name == "mov,mp4,m4a,3gp,3g2,mj2"
