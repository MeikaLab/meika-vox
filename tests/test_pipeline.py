from pathlib import Path
from meika_vox.hashing import sha256_file
from meika_vox.ids import make_audio_asset_id, make_segment_id
from meika_vox.pipeline import ingest_local

def test_ingest_is_stable_for_same_file(tmp_path: Path):
    audio=tmp_path/"fixture.wav"
    audio.write_bytes(b"synthetic-not-real-audio")
    first=ingest_local(audio,"demo")
    second=ingest_local(audio,"demo")
    assert first.audio_asset_id==second.audio_asset_id
    assert first.audio_asset_id==make_audio_asset_id("demo",sha256_file(audio))

def test_segment_id_is_zero_padded():
    assert make_segment_id("DEMO-AUD-ABC",41)=="DEMO-AUD-ABC-SEG-000041"
