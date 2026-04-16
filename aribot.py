#!/usr/bin/env python3
import json
import logging
import random
import re
import socket
import struct
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch
from dotenv import dotenv_values
from llama_cpp import Llama
from vosk import KaldiRecognizer, Model
from websocket import WebSocket

from ari import AsteriskARI

config = SimpleNamespace(**dotenv_values(".env"))


class ColorInjectedFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "color"):
            record.color = ""
        if not hasattr(record, "reset"):
            record.reset = ""
        return True


handler = logging.StreamHandler()
handler.addFilter(ColorInjectedFilter())
formatter = logging.Formatter("%(color)s[%(levelname)s] %(name)s %(message)s%(reset)s")
handler.setFormatter(formatter)
logger_root = logging.getLogger()
logger_root.addHandler(handler)
logger_root.setLevel(logging.INFO)


class ColoredLogger:
    def __init__(self, name, color=None):
        if color is None:
            color_code = random.choice([39, 45, 75, 112, 170, 208, 214])
        else:
            color_code = color

        self.color = f"\033[38;5;{color_code}m"
        self.reset = "\033[0m"

        base_logger = logging.getLogger(name)
        self._adapter = logging.LoggerAdapter(
            base_logger, {"color": self.color, "reset": self.reset}
        )

    def info(self, msg, *args, **kwargs):
        self._adapter.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._adapter.error(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._adapter.warning(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._adapter.debug(msg, *args, **kwargs)


logger = ColoredLogger("MAIN", color=7)

sessions = {}
ari = None
ws = None

logger.info("Loading STT model to listen")
stt_model = Model(config.MODEL_STT_PATH)
logger.info("-" * 50 + "\n")

logger.info("Loading LLM model to think")
llm = Llama(
    model_path=config.MODEL_LLM_PATH,
    n_ctx=4096,
    n_threads=8,
    n_threads_batch=8,
    verbose=False,
)
logger.info("-" * 50 + "\n")

logger.info("Loading TTS model to speak")
device = torch.device("cpu")  # type: ignore
torch.set_num_threads(4)  # type: ignore

tts_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language="ru",
    speaker="v4_ru",
    trust_repo="true",
)  # type: ignore

tts_model.to(device)
logger.info("-" * 50 + "\n")

logger.info(f"Models are loaded\nSTT: {stt_model}\nLLM: {llm}\nTTS: {tts_model}")
logger.info("-" * 50 + "\n")


class Aribot:
    def __init__(
        self,
        ip: str,
        port: int,
        stt_model: Model = stt_model,
        tts_model=tts_model,
        llm: Llama = llm,
        voice: str = "baya",
        asterisk_host: str = config.ASTERHOST,
        asterisk_send_port: int = 0,
    ):
        self.ip = ip
        self.port = port
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.llm = llm
        self.voice = voice
        self.asterisk_host = asterisk_host
        self.asterisk_send_port = asterisk_send_port

        self.rec = KaldiRecognizer(self.stt_model, 16000)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.ip, self.port))

        self.history = config.PROMPT
        self.seq_num: int = 0
        self.timestamp: int = 0
        self.ssrc = random.getrandbits(32)
        self.payload_type: int = 11
        self.bridge_id: str | None = None
        self.ext_channel_id: str | None = None
        self.logger = ColoredLogger("Aribot")

    def think(self, text: str) -> str:
        """
        Processes the input text and generates a thoughtful response.

        Args:
            text: The input string to analyze or respond to.

        Returns:
            A string containing the processed output.
        """
        self.history += f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        output = self.llm(
            self.history,
            max_tokens=128,
            echo=False,
        )
        llm_text = output["choices"][0]["text"].strip()  # type: ignore

        self.history += f"{llm_text}<|im_end|>\n"
        self.logger.warning(f"USER: {text}")
        self.logger.warning(f"I: {llm_text}")
        llm_text = re.sub(r"[^а-яА-ЯёЁ0-9\s.,!?-]", "", llm_text)

        if not llm_text:
            llm_text = config.NOANSWER
        return llm_text

    def speak(self, text: str) -> None:
        """
        Converts the provided text into audio chunks and transmits them to the server.

        The method encapsulates the text data into network packets and sends them
        via UDP to the configured Asterisk host and port.

        Args:
            text: The string content to be processed and transmitted.
        """
        if not self.asterisk_send_port:
            self.logger.warning("No Asterisk port set")
            return

        audio_tensor = self.tts_model.apply_tts(
            text=text, speaker=self.voice, sample_rate=8000
        )

        # 16-bit integers in Big-Endian format (standard for RTP L16) ">i2"
        audio_bytes = (audio_tensor * 32767).numpy().astype(">i2").tobytes()

        try:
            # Standard RTP packet size: 20ms of audio  8000Hz: 160 samples 16000Hz: 320 samples * 2 bytes per sample
            # chunk_size = 320
            # samples_per_chunk = 160
            chunk_size = 640
            samples_per_chunk = 320
            start_time = time.time()

            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]

                if len(chunk) < chunk_size:
                    chunk = chunk.ljust(chunk_size, b"\x00")

                # Prepare RTP header (RFC 3550) !BBHII: !=network(BE), B=1byte, H=2bytes, I=4bytes
                marker = 0x80 if i == 0 else 0x00
                header = struct.pack(
                    "!BBHII",
                    0x80,
                    self.payload_type | marker,
                    self.seq_num,
                    self.timestamp,
                    self.ssrc,
                )
                if self.sock:
                    self.sock.sendto(
                        header + chunk, (self.asterisk_host, self.asterisk_send_port)
                    )
                    self.seq_num = (self.seq_num + 1) & 0xFFFF
                    self.timestamp = (self.timestamp + samples_per_chunk) & 0xFFFFFFFF
                    expected_time = start_time + (i / chunk_size + 1) * 0.02
                    sleep_time = expected_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    self.logger.warning("No socket to speak")

        except Exception as e:
            self.logger.error(f"Error in speak method: {e}")

    def listen(self) -> None:
        """
        Runs the interactive cycle by listening to the stream and generating a response.

        The main interaction loop: it captures the incoming RTP stream, transcribes
        speech to text, and executes the 'listen-think-speak' workflow
        """

        self.logger.info(
            f"Starting ARIBot {self.ip}:{self.port} <=> {self.asterisk_host}:{self.asterisk_send_port}"
        )
        self.rec = KaldiRecognizer(self.stt_model, 16000)
        if self.sock:
            try:
                while True:
                    data, addr = self.sock.recvfrom(2048)
                    payload = data[12:]
                    le_payload = (
                        np.frombuffer(payload, dtype=">i2").astype("<i2").tobytes()
                    )

                    if self.rec.AcceptWaveform(le_payload):
                        result = json.loads(self.rec.Result())
                        if result["text"]:
                            self.logger.info(f"Recognized: {result['text']}")
                            thought = self.think(result["text"])
                            self.speak(thought)
                    # else:
                    #     logger.debug(
                    #         f"Partial result: {json.loads(self.rec.PartialResult())}"
                    #     )

            except KeyboardInterrupt:
                final_result = json.loads(self.rec.FinalResult())
                self.logger.debug(f"Recognized: {final_result['text']}")
            finally:
                self.sock.close()
        else:
            self.logger.warning("No socket to listen")

    def stop(self):
        logger.info("Stopping bot...")
        self.running = False
        time.sleep(0.2)
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
                self.sock = None
            except Exception as e:
                logger.error(f"Error closing socket: {e}")


class AsteriskBotManager:
    def __init__(
        self,
        ari_app: str,
        asterisk_host: str,
        asterisk_user: str,
        asterisk_password: str,
        audio_host: str,
        audio_start_port: int,
    ):
        self.ari_config = (ari_app, asterisk_host, asterisk_user, asterisk_password)
        self.ari: AsteriskARI | None = None
        self.ws: WebSocket | None = None
        self.audio_host = audio_host
        self.start_port = audio_start_port
        self.sessions = {}  # dict[int, Aribot]
        self.port_counter: int = 0
        self.logger = ColoredLogger("AsteriskBotManager", color=15)

    def _get_next_port(self):
        port = self.start_port + self.port_counter
        self.port_counter = (self.port_counter + 1) % 10_000
        return port

    def handle_stasis_start(self, event: dict):
        """
        Handles the entry of a new channel into the Stasis application.

        1. Initializes a new Aribot session for the incoming call.
        2. Spawns a dedicated thread for real-time audio processing (listen loop).
        3. Establishes an external media channel (Unicast RTP) for two-way audio exchange.
        4. Bridges the caller with the bot's audio stream to enable interaction.

        Args:
            event: The ARI event message.
        """
        self.logger.debug(f"ARI event START: {event}")
        ch_id = event["channel"]["id"]
        ch_name = event["channel"]["name"]

        if ch_name.startswith("UnicastRTP"):
            return

        dynamic_port = self._get_next_port()
        if self.ari:
            self.ari.answer_call(ch_id)
            bridge = self.ari.bridge_create(f"bridge-{ch_id}")

            ext_channel = self.ari.external_media_channel_create(
                host=self.audio_host,
                port=dynamic_port,
                format="slin16",
                direction="both",
            )
            self.logger.debug(f"External medeia channel: {ext_channel}")

            target_port = ext_channel.get("channelvars", {}).get(
                "UNICASTRTP_LOCAL_PORT"
            )

            self.ari.bridge_add_channel(bridge["id"], ch_id)
            self.ari.bridge_add_channel(bridge["id"], ext_channel["id"])

            if target_port:
                bot = Aribot(
                    self.audio_host,
                    port=dynamic_port,
                    asterisk_send_port=int(target_port),
                )
                self.sessions[ch_id] = bot
                bot.bridge_id = bridge["id"]
                bot.ext_channel_id = ext_channel["id"]
                threading.Thread(target=bot.listen, daemon=True).start()

            else:
                self.logger.warning(f"No port to send audio: {ext_channel}")

            # self.ari.bridge_play_sound(bridge["id"], "hello")
            # self.ari.bridge_play_sound(bridge["id"], "something-terribly-wrong")

        else:
            self.logger.error("ARI client is not connected!")
            return

    def handle_stasis_end(self, event: dict):
        """
        Handles the teardown of a session when a channel leaves the Stasis application.

        1. Identifies the active session associated with the terminating channel.
        2. Gracefully stops the bot's processing loop.
        3. Releases the external media channel and destroys the associated bridge.
        4. Removes the session from the active registry to free up resources.

        Args:
            event: The ARI event message.
        """
        self.logger.debug(f"ARI event END: {event}")
        event_id = event["channel"]["id"]
        target_main_id = None

        for main_id, bot in self.sessions.items():
            if event_id == main_id or event_id == getattr(bot, "ext_channel_id", None):
                target_main_id = main_id
                break

        if target_main_id:
            bot = self.sessions.pop(target_main_id)
            bot.stop()
            try:
                if self.ari:
                    self.ari.channel_hangup(bot.ext_channel_id)
                    self.ari.bridge_destroy(bot.bridge_id)
                else:
                    self.logger.error("ARI client is not connected!")
                    return
            except Exception as e:
                self.logger.warning(f"Cleanup error: {e}")

        self.logger.info(f"Sessions: {self.sessions}")

    def handle_channel_entered_bridge(self, event: dict):
        """
        Handles the connection process once the audio channel enters the bridge, ensuring the
        audio path is established before delivering an initial greeting (HELLO).

        Args:
            event: The ARI event message.
        """
        ch_id = event["channel"]["id"]
        bridge_id = event["bridge"]["id"]
        if self.ari:
            bridge_info = self.ari.bridge_get(bridge_id)
            self.logger.debug(f"Bridge {bridge_info}")
            channels = bridge_info.get("channels", [])
            if len(channels) >= 2:
                for bot in self.sessions.values():
                    if getattr(bot, "ext_channel_id", None) == ch_id:
                        self.logger.info(
                            f"Bot channel {ch_id} joined bridge. Audio path is ready."
                        )
                        time.sleep(0.2)
                        bot.speak(config.HELLO)
                        break
        else:
            self.logger.error("ARI client is not connected!")
            return

    def connect(self):
        """
        Establishes a persistent connection to the Asterisk ARI and its WebSocket stream.
        """
        while True:
            try:
                self.ari = AsteriskARI(*self.ari_config)
                self.ws = self.ari.create_websocket()
                logging.info("WebSocket connected successfully")
                return
            except Exception as e:
                logging.error(f"Connection failed: {e}. Retrying in 10s...")
                time.sleep(10)

    def run(self):
        """
        Starts the main event loop to monitor and maintains a persistent
        WebSocket connection, listens for incoming ARI events and dispatches
        them to their respective handlers.
        """
        if not self.ws:
            self.connect()

        self.logger.info("Bot Manager is running...")
        while True:
            try:
                if self.ws:
                    raw_event = self.ws.recv()
                    if not raw_event:
                        break
                    event = json.loads(raw_event)

                    if event["type"] == "StasisStart":
                        self.handle_stasis_start(event)
                    elif event["type"] == "StasisEnd":
                        self.handle_stasis_end(event)
                    elif event["type"] == "ChannelEnteredBridge":
                        self.handle_channel_entered_bridge(event)
                else:
                    logger.error("Websocket is not connected!")
                    return

            except KeyboardInterrupt:
                logger.info("Stopping Manager...")
                if self.ws:
                    self.ws.close()
                break
            except Exception as e:
                logger.error(f"Runtime error: {e}")
                self.connect()


if __name__ == "__main__":
    ari_bot_manger = AsteriskBotManager(
        config.ARIAPP,
        config.ASTERHOST,
        config.USER,
        config.PASSWORD,
        config.AUDIOHOST,
        int(config.AUDIOPORT),
    )
    ari_bot_manger.run()
