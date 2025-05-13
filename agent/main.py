from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext
from livekit.plugins import (
    groq,
    silero,
    noise_cancellation,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from dataclasses import dataclass

load_dotenv()

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""
        You are Aditi,a spiritual partner by Ahoum.
        You are a helpful assistant that can help the user with their spiritual needs.
        You can help the user with their spiritual practices, beliefs, and goals.
        You can also help the user with their daily life.
                         
        INSTRUCTIONS:
        - Keep your responses concise and to the point.
        - Don't use emojis in your responses.
        - Don't use hashtags, markdown, bold, italic, underline, or strikethrough in your responses.
        """)

    async def on_enter(self) -> None:
        # userdata: UserInfo = self.session.userdata
        await self.session.generate_reply(
            instructions=f"Hello, I am Aditi, your spiritual partner by Ahoum."
        )

    async def on_exit(self):
        await self.session.generate_reply(
            instructions="Tell the user a friendly goodbye before you exit.",
        )


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        # To use local Whisper STT, uncomment the following and comment out groq.STT:
        # stt=whisper.STT(
        #     model="base",  # or "small", "medium", "large", etc.
        #     language="en",
        # ),
        stt=groq.STT(
            model="whisper-large-v3-turbo",
            language="en",
        ),
        llm=groq.LLM(model="llama3-8b-8192"),
        tts=groq.TTS(
            model="playai-tts",
            voice="Arista-PlayAI",
        ),
        # tts=KokoroTTS(lang_code="a", voice="af_heart", speed=1.0, sample_rate=24000),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))