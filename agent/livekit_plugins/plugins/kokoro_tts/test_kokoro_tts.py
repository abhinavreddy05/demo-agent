import pytest
from livekit_plugins.plugins.kokoro_tts import TTS as KokoroTTS

@pytest.mark.asyncio
async def test_kokoro_tts_basic():
    tts = KokoroTTS()
    text = "Hello, this is a test of Kokoro TTS."
    stream = tts.synthesize(text)
    results = []
    async for audio in stream:
        results.append(audio)
    assert results, "No audio frames returned"
    for audio in results:
        assert hasattr(audio, 'frame'), "SynthesizedAudio missing frame"
        assert audio.frame.data, "Audio frame data is empty"
        assert audio.frame.sample_rate == tts.sample_rate
        assert audio.frame.num_channels == tts.num_channels 