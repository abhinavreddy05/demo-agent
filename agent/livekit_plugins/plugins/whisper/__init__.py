import whisper
import tempfile
import os
import wave
import logging

try:
    from livekit.agents.stt.stt import STT as BaseSTT, STTCapabilities, SpeechEvent, SpeechEventType, SpeechData, RecognitionUsage
    from livekit.agents.utils.audio import AudioBuffer
except ImportError:
    from livekit_agents.stt.stt import STT as BaseSTT, STTCapabilities, SpeechEvent, SpeechEventType, SpeechData, RecognitionUsage
    from livekit_agents.utils.audio import AudioBuffer

logger = logging.getLogger("whisper_plugin")

class STT(BaseSTT):
    def __init__(self, model="base", language="en", sample_rate=16000, num_channels=1):
        super().__init__(capabilities=STTCapabilities(
            streaming=False,
            interim_results=False,
        ))
        self.model_name = model
        self.language = language
        self.model = whisper.load_model(model)
        self.sample_rate = sample_rate
        self.num_channels = num_channels

    @property
    def label(self):
        return f"whisper.STT({self.model_name})"

    async def _recognize_impl(self, buffer: AudioBuffer, *, language=None, conn_options=None):
        # Diagnostic logging
        frame_count = len(buffer) if isinstance(buffer, list) else 1
        logger.info(f"Whisper STT: Received {frame_count} frame(s)")
        audio_bytes = b''
        first_frame = buffer[0] if isinstance(buffer, list) else buffer
        logger.info(f"First frame sample_rate: {getattr(first_frame, 'sample_rate', 'unknown')}, num_channels: {getattr(first_frame, 'num_channels', 'unknown')}, samples_per_channel: {getattr(first_frame, 'samples_per_channel', 'unknown')}")
        if isinstance(buffer, list):
            for frame in buffer:
                audio_bytes += frame.data
        else:
            audio_bytes = buffer.data
        logger.info(f"Total audio bytes: {len(audio_bytes)}")
        logger.info(f"First 32 bytes: {audio_bytes[:32].hex()}")
        if not audio_bytes or len(audio_bytes) < 320:  # less than 10ms at 16kHz mono
            logger.warning("Audio buffer is empty or too short for transcription.")
        # Write a valid WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp, 'wb') as wf:
                wf.setnchannels(self.num_channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_bytes)
            tmp_path = tmp.name
        try:
            result = self.model.transcribe(tmp_path, language=self.language)
            text = result["text"]
            speech_data = SpeechData(
                language=self.language,
                text=text,
                start_time=0.0,
                end_time=0.0,
                confidence=1.0,
            )
            usage = RecognitionUsage(audio_duration=result.get("duration", 0.0))
            event = SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[speech_data],
                recognition_usage=usage,
            )
            return event
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            speech_data = SpeechData(
                language=self.language,
                text=f"[Whisper error: {e}]",
                start_time=0.0,
                end_time=0.0,
                confidence=0.0,
            )
            event = SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[speech_data],
                recognition_usage=None,
            )
            return event
        finally:
            os.remove(tmp_path) 