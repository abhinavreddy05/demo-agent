import numpy as np
import soundfile as sf
from kokoro import KPipeline
import torch
import uuid
import logging
from scipy.signal import resample_poly

try:
    from livekit.agents.tts.tts import TTS as BaseTTS, TTSCapabilities, SynthesizedAudio
    from livekit import rtc
    from livekit.agents.types import APIConnectOptions
except ImportError:
    from livekit_agents.tts.tts import TTS as BaseTTS, TTSCapabilities, SynthesizedAudio
    from livekit import rtc
    from livekit_agents.types import APIConnectOptions

logger = logging.getLogger("kokoro_tts_plugin")

class KokoroTTS(BaseTTS):
    """
    Kokoro TTS node for LiveKit Agent, using the open-weight Kokoro-82M model.
    Supports batch synthesis (not streaming).
    """
    def __init__(
        self,
        lang_code: str = "a",  # American English by default
        voice: str = "af_heart",
        speed: float = 1.0,
        sample_rate: int = 24000,
        num_channels: int = 1,
        **kwargs
    ):
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self.pipeline = KPipeline(lang_code=lang_code)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = None,
    ):
        """
        Synthesize speech from text using Kokoro TTS.
        Returns a ChunkedStream-like object yielding SynthesizedAudio.
        """
        return _KokoroChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    async def aclose(self):
        pass  # No persistent resources to close

class _KokoroChunkedStream:
    def __init__(self, *, tts: KokoroTTS, input_text: str, conn_options=None):
        self._tts = tts
        self._input_text = input_text
        self._conn_options = conn_options
        self._done = False
        self._exception = None
        self._audio_frames = None
        self._request_id = str(uuid.uuid4())

    @property
    def input_text(self):
        return self._input_text

    @property
    def done(self):
        return self._done

    @property
    def exception(self):
        return self._exception

    def __aiter__(self):
        return self._run()

    async def _run(self):
        try:
            # Synthesize audio using Kokoro
            generator = self._tts.pipeline(
                self._input_text,
                voice=self._tts.voice,
                speed=self._tts.speed,
                split_pattern=r'\n+'
            )
            for i, (gs, ps, audio) in enumerate(generator):
                # Ensure audio is a numpy array
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.cpu().numpy()
                else:
                    audio_np = audio
                # Log audio shape and range
                logger.info(f"Kokoro audio shape: {audio_np.shape}, dtype: {audio_np.dtype}, min: {audio_np.min()}, max: {audio_np.max()}")

                # Determine the expected output sample rate
                expected_rate = self._tts.sample_rate
                model_rate = 24000  # Kokoro default output

                # Resample if needed (do not mutate self._tts.sample_rate)
                if model_rate != expected_rate:
                    logger.warning(f"Resampling from {model_rate} Hz to {expected_rate} Hz.")
                    audio_np = resample_poly(audio_np, expected_rate, model_rate)

                # Convert to int16 PCM for agent compatibility
                audio_int16 = np.clip(audio_np, -1.0, 1.0)
                audio_int16 = (audio_int16 * 32767.0).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                samples_per_channel = len(audio_int16)
                frame = rtc.AudioFrame(
                    data=audio_bytes,
                    sample_rate=expected_rate,
                    num_channels=self._tts.num_channels,
                    samples_per_channel=samples_per_channel,
                )
                yield SynthesizedAudio(
                    frame=frame,
                    request_id=self._request_id,
                    is_final=True if i == 0 else False,
                    segment_id=str(i),
                    delta_text=gs,
                )
            self._done = True
        except Exception as e:
            self._exception = e
            raise 