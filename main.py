import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import av
import numpy as np

from av import AudioFrame, VideoFrame

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("pyav_media_studio")


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    ""
).strip()

PORT = int(
    os.getenv(
        "PORT",
        "8000"
    )
)

BASE_DIR = Path(
    os.getenv(
        "MEDIA_DIR",
        "/tmp/pyav_media"
    )
)

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GLOBAL BOT
# ============================================================

telegram_application: Optional[Application] = None


# ============================================================
# USER STATE
# ============================================================

USER_FILES = {}
USER_SETTINGS = {}


def default_settings():
    return {
        "operation": "convert",

        "start": None,
        "end": None,

        "volume": 1.0,

        "speed": 1.0,

        "width": None,
        "height": None,

        "fps": None,

        "sample_rate": None,
        "channels": None,

        "audio_bitrate": None,
        "video_bitrate": None,

        "remove_audio": False,
        "extract_audio": False,

        "output_format": None,
    }


def get_settings(user_id):

    if user_id not in USER_SETTINGS:
        USER_SETTINGS[user_id] = default_settings()

    return USER_SETTINGS[user_id]


# ============================================================
# FILE UTILITIES
# ============================================================

def safe_filename(filename):

    filename = Path(
        filename or "media"
    ).name

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "._-"
    )

    result = "".join(
        c if c in allowed else "_"
        for c in filename
    )

    return result or "media"


def unique_path(directory, filename):

    return directory / (
        uuid.uuid4().hex
        + "_"
        + filename
    )


def cleanup_file(path):

    if not path:
        return

    try:
        Path(path).unlink(
            missing_ok=True
        )
    except Exception:
        pass


# ============================================================
# BITRATE
# ============================================================

def parse_bitrate(value):

    if value is None:
        return None

    if isinstance(
        value,
        int
    ):
        return value

    text = str(
        value
    ).strip().lower()

    multiplier = 1

    if text.endswith("k"):

        multiplier = 1000
        text = text[:-1]

    elif text.endswith("m"):

        multiplier = 1000000
        text = text[:-1]

    try:

        return int(
            float(text)
            * multiplier
        )

    except Exception:

        return None


# ============================================================
# MEDIA INSPECTION
# ============================================================

def inspect_media(path):

    container = None

    result = {
        "filename": Path(path).name,
        "format": None,
        "duration": None,
        "bitrate": None,
        "streams": [],
    }

    try:

        container = av.open(
            str(path)
        )

        if container.format:

            result["format"] = (
                container.format.name
            )

        if container.duration is not None:

            result["duration"] = (
                float(
                    container.duration
                )
                / float(av.time_base)
            )

        if container.bit_rate:

            result["bitrate"] = (
                container.bit_rate
            )

        for stream in container.streams:

            item = {
                "index": stream.index,
                "type": stream.type,
                "codec": None,
            }

            if stream.codec_context:

                item["codec"] = (
                    stream.codec_context.name
                )

            if stream.type == "video":

                item.update({
                    "width": stream.width,
                    "height": stream.height,
                    "fps": (
                        float(stream.average_rate)
                        if stream.average_rate
                        else None
                    ),
                })

            elif stream.type == "audio":

                item.update({
                    "sample_rate": stream.rate,
                    "channels": stream.channels,
                    "layout": (
                        str(stream.layout)
                        if stream.layout
                        else None
                    ),
                })

            result["streams"].append(
                item
            )

    finally:

        if container:

            try:
                container.close()
            except Exception:
                pass

    return result


# ============================================================
# STREAM HELPERS
# ============================================================

def find_video_stream(container):

    for stream in container.streams:

        if stream.type == "video":
            return stream

    return None


def find_audio_stream(container):

    for stream in container.streams:

        if stream.type == "audio":
            return stream

    return None


# ============================================================
# CODEC SELECTION
# ============================================================

def choose_video_codec(format_name):

    fmt = (
        format_name or ""
    ).lower()

    if fmt in {
        "mp4",
        "mov",
        "m4v",
    }:

        return "h264"

    if fmt == "webm":

        return "vp8"

    if fmt in {
        "mkv",
        "matroska",
    }:

        return "h264"

    return "h264"


def choose_audio_codec(format_name):

    fmt = (
        format_name or ""
    ).lower()

    if fmt == "mp3":

        return "mp3"

    if fmt in {
        "ogg",
        "opus",
    }:

        return "opus"

    if fmt == "flac":

        return "flac"

    if fmt == "wav":

        return "pcm_s16le"

    if fmt in {
        "m4a",
        "mp4",
        "mov",
    }:

        return "aac"

    return "aac"


# ============================================================
# VIDEO PROCESSING
# ============================================================

def resize_video_frame(
    frame,
    width,
    height,
):

    if not width or not height:

        return frame

    return frame.reformat(
        width=int(width),
        height=int(height),
    )


# ============================================================
# AUDIO PROCESSING
# ============================================================

def process_audio_frame(
    frame,
    volume=1.0,
):

    if volume is None:
        volume = 1.0

    if float(volume) == 1.0:

        return frame

    try:

        array = frame.to_ndarray()

        array = (
            array.astype(
                np.float32
            )
            * float(volume)
        )

        if np.issubdtype(
            array.dtype,
            np.integer
        ):

            info = np.iinfo(
                array.dtype
            )

            array = np.clip(
                array,
                info.min,
                info.max,
            )

        else:

            array = np.clip(
                array,
                -1.0,
                1.0,
            )

        if array.dtype != frame.to_ndarray().dtype:

            original = frame.to_ndarray().dtype

            if np.issubdtype(
                original,
                np.integer
            ):

                info = np.iinfo(
                    original
                )

                array = array.astype(
                    original
                )

        new_frame = (
            AudioFrame.from_ndarray(
                array,
                layout=frame.layout.name,
            )
        )

        new_frame.sample_rate = (
            frame.sample_rate
        )

        if frame.pts is not None:

            new_frame.pts = frame.pts

        new_frame.time_base = (
            frame.time_base
        )

        return new_frame

    except Exception as e:

        logger.warning(
            "Audio volume processing failed: %s",
            e,
        )

        return frame


# ============================================================
# PYAV TRANSCODING ENGINE
# ============================================================

def process_media_pyav(
    input_path,
    output_path,
    settings,
):

    input_container = None
    output_container = None

    try:

        input_container = av.open(
            str(input_path)
        )

        input_format = (
            input_container.format.name
            if input_container.format
            else "mp4"
        )

        output_format = (
            settings.get(
                "output_format"
            )
            or input_format
        )

        output_format = (
            str(output_format)
            .lower()
            .replace(
                ".",
                ""
            )
        )

        output_container = av.open(
            str(output_path),
            mode="w",
            format=output_format,
        )

        video_stream = find_video_stream(
            input_container
        )

        audio_stream = find_audio_stream(
            input_container
        )

        output_video = None
        output_audio = None

        speed = float(
            settings.get(
                "speed",
                1.0
            )
            or 1.0
        )

        if speed <= 0:
            speed = 1.0

        # ====================================================
        # VIDEO OUTPUT
        # ====================================================

        if (
            video_stream
            and not settings.get(
                "extract_audio",
                False
            )
        ):

            codec = choose_video_codec(
                output_format
            )

            try:

                output_video = (
                    output_container.add_stream(
                        codec
                    )
                )

            except Exception as e:

                logger.warning(
                    "Primary video codec failed: %s",
                    e,
                )

                source_codec = (
                    video_stream.codec_context.name
                )

                output_video = (
                    output_container.add_stream(
                        source_codec
                    )
                )

            width = (
                settings.get("width")
                or video_stream.width
            )

            height = (
                settings.get("height")
                or video_stream.height
            )

            output_video.width = int(
                width
            )

            output_video.height = int(
                height
            )

            fps = settings.get(
                "fps"
            )

            if fps:

                output_video.average_rate = (
                    float(fps)
                )

            elif video_stream.average_rate:

                output_video.average_rate = (
                    video_stream.average_rate
                )

            if settings.get(
                "video_bitrate"
            ):

                bitrate = parse_bitrate(
                    settings[
                        "video_bitrate"
                    ]
                )

                if bitrate:

                    output_video.bit_rate = (
                        bitrate
                    )

        # ====================================================
        # AUDIO OUTPUT
        # ====================================================

        if (
            audio_stream
            and not settings.get(
                "remove_audio",
                False
            )
        ):

            if settings.get(
                "extract_audio",
                False
            ):

                audio_format = (
                    output_format
                    or "mp3"
                )

            else:

                audio_format = (
                    output_format
                )

            codec = choose_audio_codec(
                audio_format
            )

            source_rate = (
                audio_stream.rate
                or 44100
            )

            requested_rate = (
                settings.get(
                    "sample_rate"
                )
                or source_rate
            )

            target_rate = int(
                requested_rate
            )

            # Speed changes playback duration
            # by changing the output sample rate.
            if (
                speed != 1.0
                and not settings.get(
                    "sample_rate"
                )
            ):

                target_rate = max(
                    8000,
                    min(
                        192000,
                        int(
                            source_rate
                            * speed
                        )
                    )
                )

            try:

                output_audio = (
                    output_container.add_stream(
                        codec,
                        rate=target_rate,
                    )
                )

            except Exception as e:

                logger.warning(
                    "Primary audio codec failed: %s",
                    e,
                )

                output_audio = (
                    output_container.add_stream(
                        audio_stream.codec_context.name,
                        rate=target_rate,
                    )
                )

            if settings.get(
                "audio_bitrate"
            ):

                bitrate = parse_bitrate(
                    settings[
                        "audio_bitrate"
                    ]
                )

                if bitrate:

                    output_audio.bit_rate = (
                        bitrate
                    )

        # ====================================================
        # AUDIO RESAMPLER
        # ====================================================

        resampler = None

        if audio_stream and output_audio:

            try:

                target_rate = (
                    output_audio.rate
                )

                target_layout = (
                    str(
                        audio_stream.layout
                    )
                    if audio_stream.layout
                    else None
                )

                resampler = (
                    av.audio.resampler.AudioResampler(
                        format="fltp",
                        layout=target_layout,
                        rate=target_rate,
                    )
                )

            except Exception as e:

                logger.warning(
                    "Audio resampler unavailable: %s",
                    e,
                )

                resampler = None

        # ====================================================
        # DECODE
        # ====================================================

        streams = []

        if video_stream:
            streams.append(
                video_stream
            )

        if audio_stream:
            streams.append(
                audio_stream
            )

        start = settings.get(
            "start"
        )

        end = settings.get(
            "end"
        )

        for frame in input_container.decode(
            *streams
        ):

            # =================================================
            # VIDEO FRAME
            # =================================================

            if isinstance(
                frame,
                VideoFrame
            ):

                pts_seconds = None

                if (
                    frame.pts is not None
                    and frame.time_base is not None
                ):

                    try:

                        pts_seconds = float(
                            frame.pts
                            * frame.time_base
                        )

                    except Exception:
                        pass

                if (
                    start is not None
                    and pts_seconds is not None
                    and pts_seconds < float(start)
                ):

                    continue

                if (
                    end is not None
                    and pts_seconds is not None
                    and pts_seconds > float(end)
                ):

                    continue

                if output_video:

                    frame = resize_video_frame(
                        frame,
                        settings.get(
                            "width"
                        ),
                        settings.get(
                            "height"
                        ),
                    )

                    try:

                        frame = frame.reformat(
                            format="yuv420p"
                        )

                    except Exception:

                        pass

                    # Change PTS for playback speed.
                    if (
                        speed != 1.0
                        and frame.pts is not None
                    ):

                        frame.pts = int(
                            frame.pts
                            / speed
                        )

                    try:

                        for packet in (
                            output_video.encode(
                                frame
                            )
                        ):

                            output_container.mux(
                                packet
                            )

                    except Exception as e:

                        raise RuntimeError(
                            f"Video encoding failed: {e}"
                        )

            # =================================================
            # AUDIO FRAME
            # =================================================

            elif isinstance(
                frame,
                AudioFrame
            ):

                pts_seconds = None

                if (
                    frame.pts is not None
                    and frame.time_base is not None
                ):

                    try:

                        pts_seconds = float(
                            frame.pts
                            * frame.time_base
                        )

                    except Exception:
                        pass

                if (
                    start is not None
                    and pts_seconds is not None
                    and pts_seconds < float(start)
                ):

                    continue

                if (
                    end is not None
                    and pts_seconds is not None
                    and pts_seconds > float(end)
                ):

                    continue

                if output_audio:

                    frame = process_audio_frame(
                        frame,
                        settings.get(
                            "volume",
                            1.0
                        ),
                    )

                    # -----------------------------------------
                    # Resample audio
                    # -----------------------------------------

                    frames_to_encode = [frame]

                    if resampler:

                        try:

                            resampled = (
                                resampler.resample(
                                    frame
                                )
                            )

                            if resampled:

                                if isinstance(
                                    resampled,
                                    list
                                ):

                                    frames_to_encode = (
                                        resampled
                                    )

                                else:

                                    frames_to_encode = [
                                        resampled
                                    ]

                        except Exception as e:

                            logger.warning(
                                "Audio resampling failed: %s",
                                e,
                            )

                    for audio_frame in (
                        frames_to_encode
                    ):

                        if (
                            speed != 1.0
                            and audio_frame.pts is not None
                        ):

                            audio_frame.pts = int(
                                audio_frame.pts
                                / speed
                            )

                        try:

                            for packet in (
                                output_audio.encode(
                                    audio_frame
                                )
                            ):

                                output_container.mux(
                                    packet
                                )

                        except Exception as e:

                            raise RuntimeError(
                                f"Audio encoding failed: {e}"
                            )

        # ====================================================
        # FLUSH VIDEO
        # ====================================================

        if output_video:

            for packet in (
                output_video.encode()
            ):

                output_container.mux(
                    packet
                )

        # ====================================================
        # FLUSH AUDIO
        # ====================================================

        if output_audio:

            if resampler:

                try:

                    flushed = (
                        resampler.resample(
                            None
                        )
                    )

                    if flushed:

                        if not isinstance(
                            flushed,
                            list
                        ):

                            flushed = [
                                flushed
                            ]

                        for frame in flushed:

                            for packet in (
                                output_audio.encode(
                                    frame
                                )
                            ):

                                output_container.mux(
                                    packet
                                )

                except Exception as e:

                    logger.warning(
                        "Audio resampler flush failed: %s",
                        e,
                    )

            for packet in (
                output_audio.encode()
            ):

                output_container.mux(
                    packet
                )

        # ====================================================
        # CLOSE
        # ====================================================

        output_container.close()

        output_container = None

        return str(
            output_path
        )

    finally:

        if input_container:

            try:
                input_container.close()
            except Exception:
                pass

        if output_container:

            try:
                output_container.close()
            except Exception:
                pass


# ============================================================
# FASTAPI
# ============================================================

@asynccontextmanager
async def lifespan(application):

    global telegram_application

    logger.info(
        "FastAPI starting..."
    )

    if BOT_TOKEN:

        try:

            telegram_application = (
                Application.builder()
                .token(BOT_TOKEN)
                .build()
            )

            register_handlers(
                telegram_application
            )

            await telegram_application.initialize()

            await telegram_application.start()

            if telegram_application.updater:

                await telegram_application.updater.start_polling(
                    allowed_updates=Update.ALL_TYPES
                )

            logger.info(
                "Telegram bot started successfully."
            )

        except Exception:

            logger.exception(
                "Telegram bot failed to start."
            )

            telegram_application = None

    else:

        logger.warning(
            "BOT_TOKEN is not configured."
        )

    yield

    logger.info(
        "FastAPI shutting down..."
    )

    if telegram_application:

        try:

            if telegram_application.updater:

                await telegram_application.updater.stop()

            await telegram_application.stop()

            await telegram_application.shutdown()

        except Exception:

            logger.exception(
                "Error while shutting down Telegram bot."
            )


app = FastAPI(
    title="PyAV Professional Media Studio",
    version="3.0.0",
    description=(
        "Interactive Telegram media processing "
        "server powered by PyAV."
    ),
    lifespan=lifespan,
)


# ============================================================
# API ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "service": "PyAV Professional Media Studio",
        "engine": "PyAV",
        "external_ffmpeg": False,
        "telegram_bot": bool(
            BOT_TOKEN
        ),
        "version": "3.0.0",
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "pyav": True,
        "external_ffmpeg": False,
        "telegram_bot": bool(
            telegram_application
        ),
    }


# ============================================================
# API INFO
# ============================================================

@app.post("/info")
async def api_info(
    file: UploadFile = File(...)
):

    filename = safe_filename(
        file.filename
    )

    path = unique_path(
        INPUT_DIR,
        filename
    )

    try:

        with open(
            path,
            "wb"
        ) as output:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output.write(
                    chunk
                )

        return await asyncio.to_thread(
            inspect_media,
            str(path)
        )

    finally:

        cleanup_file(
            path
        )


# ============================================================
# API PROCESS
# ============================================================

@app.post("/process")
async def api_process(
    file: UploadFile = File(...),

    output_format: Optional[str] = None,

    start: Optional[float] = None,
    end: Optional[float] = None,

    volume: Optional[float] = 1.0,

    speed: Optional[float] = 1.0,

    width: Optional[int] = None,
    height: Optional[int] = None,

    fps: Optional[float] = None,

    sample_rate: Optional[int] = None,

    audio_bitrate: Optional[str] = None,
    video_bitrate: Optional[str] = None,

    remove_audio: bool = False,
    extract_audio: bool = False,
):

    original_name = safe_filename(
        file.filename
    )

    input_path = unique_path(
        INPUT_DIR,
        original_name
    )

    try:

        with open(
            input_path,
            "wb"
        ) as output:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output.write(
                    chunk
                )

        extension = (
            output_format
            or input_path.suffix.lstrip(".")
        )

        if extract_audio:

            extension = (
                output_format
                or "mp3"
            )

        extension = (
            extension
            .lower()
            .replace(
                ".",
                ""
            )
        )

        output_name = (
            f"{input_path.stem}"
            f"_processed."
            f"{extension}"
        )

        output_path = (
            OUTPUT_DIR
            / output_name
        )

        settings = default_settings()

        settings.update({

            "output_format": extension,

            "start": start,

            "end": end,

            "volume": volume,

            "speed": speed,

            "width": width,

            "height": height,

            "fps": fps,

            "sample_rate": sample_rate,

            "audio_bitrate": (
                audio_bitrate
            ),

            "video_bitrate": (
                video_bitrate
            ),

            "remove_audio": (
                remove_audio
            ),

            "extract_audio": (
                extract_audio
            ),
        })

        await asyncio.to_thread(
            process_media_pyav,
            input_path,
            output_path,
            settings,
        )

        if not output_path.exists():

            raise HTTPException(
                status_code=500,
                detail="Processing failed."
            )

        return FileResponse(
            path=output_path,
            filename=output_name,
            media_type=(
                "application/octet-stream"
            ),
        )

    except HTTPException:

        raise

    except Exception as e:

        logger.exception(
            "API processing error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        cleanup_file(
            input_path
        )


# ============================================================
# TELEGRAM UI
# ============================================================

def main_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "✂️ قص",
                callback_data="trim"
            ),
            InlineKeyboardButton(
                "🔊 الصوت",
                callback_data="volume"
            ),
        ],

        [
            InlineKeyboardButton(
                "⚡ السرعة",
                callback_data="speed"
            ),
            InlineKeyboardButton(
                "📐 الدقة",
                callback_data="resolution"
            ),
        ],

        [
            InlineKeyboardButton(
                "🎞 FPS",
                callback_data="fps"
            ),
            InlineKeyboardButton(
                "🔄 الصيغة",
                callback_data="format"
            ),
        ],

        [
            InlineKeyboardButton(
                "🎵 استخراج الصوت",
                callback_data="extract"
            ),
            InlineKeyboardButton(
                "🔇 كتم الصوت",
                callback_data="mute"
            ),
        ],

        [
            InlineKeyboardButton(
                "🎚 Audio Bitrate",
                callback_data="audio_bitrate"
            ),
            InlineKeyboardButton(
                "🎚 Video Bitrate",
                callback_data="video_bitrate"
            ),
        ],

        [
            InlineKeyboardButton(
                "🎛 Sample Rate",
                callback_data="sample"
            ),
            InlineKeyboardButton(
                "ℹ️ معلومات",
                callback_data="info"
            ),
        ],

        [
            InlineKeyboardButton(
                "⚙️ الإعدادات الحالية",
                callback_data="settings"
            ),
        ],

        [
            InlineKeyboardButton(
                "🚀 تنفيذ",
                callback_data="process"
            ),
            InlineKeyboardButton(
                "🔄 إعادة ضبط",
                callback_data="reset"
            ),
        ],

        [
            InlineKeyboardButton(
                "📖 شرح الاستخدام",
                callback_data="help"
            ),
        ],
    ])


def back_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "↩️ رجوع",
                callback_data="back"
            ),

            InlineKeyboardButton(
                "❌ إلغاء",
                callback_data="cancel"
            ),
        ]

    ])


# ============================================================
# START MESSAGE
# ============================================================

START_TEXT = """
🎬 PyAV Professional Media Studio

مرحبًا بك 👋

هذا البوت يحول السيرفر إلى استوديو
تفاعلي لمعالجة الفيديو والصوت.

🧠 محرك المعالجة:
PyAV

🚫 لا يتم تشغيل FFmpeg executable.

━━━━━━━━━━━━━━━━━━

📤 ابدأ بإرسال:

🎥 فيديو
🎵 ملف صوتي
📄 ملف Media

ثم ستظهر لك أدوات التحكم.

━━━━━━━━━━━━━━━━━━

✂️ قص
🔊 الصوت
⚡ السرعة
📐 الدقة
🎞 FPS
🔄 الصيغة
🎵 استخراج الصوت
🔇 كتم الصوت
🎚 Bitrate
🎛 Sample Rate

ثم اضغط:

🚀 تنفيذ

━━━━━━━━━━━━━━━━━━

📖 لمعرفة طريقة الاستخدام:
اضغط «شرح الاستخدام».
"""


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not update.message:
        return

    context.user_data.pop(
        "awaiting",
        None
    )

    await update.message.reply_text(
        START_TEXT,
        reply_markup=main_keyboard(),
    )


# ============================================================
# HELP
# ============================================================

HELP_TEXT = """
📖 شرح استخدام PyAV Studio

1️⃣ أرسل الفيديو أو الملف الصوتي.

2️⃣ بعد رفع الملف ستظهر لوحة التحكم.

3️⃣ اختر العملية المطلوبة.

4️⃣ إذا كانت العملية تحتاج قيمة،
سيطلب منك البوت إدخالها.

5️⃣ بعد الانتهاء من الإعدادات،
اضغط ⚙️ الإعدادات الحالية
للتأكد من كل شيء.

6️⃣ اضغط 🚀 تنفيذ.

7️⃣ انتظر حتى ينتهي PyAV من المعالجة.

8️⃣ سيُرسل الملف الناتج إليك.

━━━━━━━━━━━━━━━━━━

🎥 الفيديو

✂️ القص:
أدخل:
10 60

يعني من الثانية 10
إلى الثانية 60.

📐 الدقة:
اختر دقة جاهزة أو أدخل:
1920 1080

🎞 FPS:
مثال:
30
أو:
60

⚡ السرعة:
مثال:
0.5
1
1.5
2

🔄 الصيغة:
MP4 / MKV / WEBM
أو الصيغ الصوتية.

━━━━━━━━━━━━━━━━━━

🎵 الصوت

🔊 الصوت:
1.5 = زيادة 50%
0.5 = خفض الصوت للنصف.

🎵 استخراج الصوت:
يحاول إنشاء ملف صوتي مستقل.

🔇 كتم الصوت:
يزيل مسار الصوت من الناتج.

🎛 Sample Rate:
44100
48000

━━━━━━━━━━━━━━━━━━

⚠️ ملاحظات

المعالجة تعتمد على قدرات PyAV
والـ codecs الموجودة ضمن بيئة PyAV.

كما توجد حدود لحجم الملفات
تفرضها Telegram والسيرفر.
"""


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if update.message:

        await update.message.reply_text(
            HELP_TEXT,
            reply_markup=main_keyboard(),
        )


# ============================================================
# MEDIA UPLOAD
# ============================================================

async def handle_media(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not update.message:
        return

    message = update.message

    user_id = (
        message.from_user.id
    )

    telegram_file = None

    filename = "media"

    if message.document:

        telegram_file = (
            await message.document.get_file()
        )

        filename = (
            message.document.file_name
            or "media"
        )

    elif message.video:

        telegram_file = (
            await message.video.get_file()
        )

        filename = "video.mp4"

    elif message.audio:

        telegram_file = (
            await message.audio.get_file()
        )

        filename = "audio.mp3"

    elif message.voice:

        telegram_file = (
            await message.voice.get_file()
        )

        filename = "voice.ogg"

    if not telegram_file:

        return

    filename = safe_filename(
        filename
    )

    local_path = unique_path(
        INPUT_DIR,
        f"{user_id}_{filename}"
    )

    try:

        await telegram_file.download_to_drive(
            custom_path=str(
                local_path
            )
        )

        old_file = USER_FILES.get(
            user_id
        )

        if old_file:

            cleanup_file(
                old_file
            )

        USER_FILES[user_id] = str(
            local_path
        )

        USER_SETTINGS[user_id] = (
            default_settings()
        )

        info = await asyncio.to_thread(
            inspect_media,
            str(local_path)
        )

        duration = info.get(
            "duration"
        )

        if duration:

            duration_text = (
                f"{duration:.1f} ثانية"
            )

        else:

            duration_text = (
                "غير معروف"
            )

        await message.reply_text(
            (
                "✅ تم رفع الملف بنجاح.\n\n"
                f"📄 {filename}\n"
                f"⏱ المدة: {duration_text}\n\n"
                "اختر العملية المطلوبة من القائمة:"
            ),
            reply_markup=main_keyboard(),
        )

    except Exception as e:

        cleanup_file(
            local_path
        )

        logger.exception(
            "Telegram upload error"
        )

        await message.reply_text(
            (
                "❌ حدث خطأ أثناء رفع الملف:\n\n"
                f"{str(e)[:3000]}"
            )
        )


# ============================================================
# SETTINGS SUMMARY
# ============================================================

def settings_summary(
    user_id
):

    settings = get_settings(
        user_id
    )

    lines = []

    lines.append(
        "⚙️ الإعدادات الحالية"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"✂️ البداية: "
        f"{settings.get('start') or 'تلقائي'}"
    )

    lines.append(
        f"🏁 النهاية: "
        f"{settings.get('end') or 'تلقائي'}"
    )

    lines.append(
        f"🔊 الصوت: "
        f"{settings.get('volume', 1.0)}×"
    )

    lines.append(
        f"⚡ السرعة: "
        f"{settings.get('speed', 1.0)}×"
    )

    width = settings.get(
        "width"
    )

    height = settings.get(
        "height"
    )

    resolution = (
        f"{width}×{height}"
        if width and height
        else "الأصلية"
    )

    lines.append(
        f"📐 الدقة: {resolution}"
    )

    lines.append(
        f"🎞 FPS: "
        f"{settings.get('fps') or 'الأصلية'}"
    )

    lines.append(
        f"🔄 الصيغة: "
        f"{settings.get('output_format') or 'الأصلية'}"
    )

    lines.append(
        f"🎛 Sample Rate: "
        f"{settings.get('sample_rate') or 'الأصلية'}"
    )

    lines.append(
        f"🎚 Audio Bitrate: "
        f"{settings.get('audio_bitrate') or 'افتراضي'}"
    )

    lines.append(
        f"🎚 Video Bitrate: "
        f"{settings.get('video_bitrate') or 'افتراضي'}"
    )

    lines.append(
        f"🎵 استخراج الصوت: "
        f"{'نعم' if settings.get('extract_audio') else 'لا'}"
    )

    lines.append(
        f"🔇 كتم الصوت: "
        f"{'نعم' if settings.get('remove_audio') else 'لا'}"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    return "\n".join(
        lines
    )


# ============================================================
# FORMAT KEYBOARD
# ============================================================

def format_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "🎬 MP4",
                callback_data="setformat_mp4"
            ),
            InlineKeyboardButton(
                "🎬 MKV",
                callback_data="setformat_mkv"
            ),
        ],

        [
            InlineKeyboardButton(
                "🌐 WEBM",
                callback_data="setformat_webm"
            ),
            InlineKeyboardButton(
                "🎬 MOV",
                callback_data="setformat_mov"
            ),
        ],

        [
            InlineKeyboardButton(
                "🎵 MP3",
                callback_data="setformat_mp3"
            ),
            InlineKeyboardButton(
                "🎵 WAV",
                callback_data="setformat_wav"
            ),
        ],

        [
            InlineKeyboardButton(
                "🎵 FLAC",
                callback_data="setformat_flac"
            ),
            InlineKeyboardButton(
                "🎵 M4A",
                callback_data="setformat_m4a"
            ),
        ],

        [
            InlineKeyboardButton(
                "✏️ صيغة مخصصة",
                callback_data="custom_format"
            ),
        ],

        [
            InlineKeyboardButton(
                "↩️ رجوع",
                callback_data="back"
            ),
        ],
    ])


# ============================================================
# RESOLUTION KEYBOARD
# ============================================================

def resolution_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "2160p 4K",
                callback_data="setres_3840_2160"
            ),
            InlineKeyboardButton(
                "1440p",
                callback_data="setres_2560_1440"
            ),
        ],

        [
            InlineKeyboardButton(
                "1080p",
                callback_data="setres_1920_1080"
            ),
            InlineKeyboardButton(
                "720p",
                callback_data="setres_1280_720"
            ),
        ],

        [
            InlineKeyboardButton(
                "480p",
                callback_data="setres_854_480"
            ),
            InlineKeyboardButton(
                "360p",
                callback_data="setres_640_360"
            ),
        ],

        [
            InlineKeyboardButton(
                "✏️ مخصصة",
                callback_data="custom_resolution"
            ),
        ],

        [
            InlineKeyboardButton(
                "↩️ رجوع",
                callback_data="back"
            ),
        ],
    ])


# ============================================================
# SPEED KEYBOARD
# ============================================================

def speed_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "🐢 0.5×",
                callback_data="setspeed_0.5"
            ),
            InlineKeyboardButton(
                "0.75×",
                callback_data="setspeed_0.75"
            ),
        ],

        [
            InlineKeyboardButton(
                "▶️ 1×",
                callback_data="setspeed_1"
            ),
            InlineKeyboardButton(
                "⚡ 1.25×",
                callback_data="setspeed_1.25"
            ),
        ],

        [
            InlineKeyboardButton(
                "⚡ 1.5×",
                callback_data="setspeed_1.5"
            ),
            InlineKeyboardButton(
                "🚀 2×",
                callback_data="setspeed_2"
            ),
        ],

        [
            InlineKeyboardButton(
                "✏️ سرعة مخصصة",
                callback_data="custom_speed"
            ),
        ],

        [
            InlineKeyboardButton(
                "↩️ رجوع",
                callback_data="back"
            ),
        ],
    ])


# ============================================================
# FPS KEYBOARD
# ============================================================

def fps_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "24 FPS",
                callback_data="setfps_24"
            ),
            InlineKeyboardButton(
                "25 FPS",
                callback_data="setfps_25"
            ),
        ],

        [
            InlineKeyboardButton(
                "30 FPS",
                callback_data="setfps_30"
            ),
            InlineKeyboardButton(
                "50 FPS",
                callback_data="setfps_50"
            ),
        ],

        [
            InlineKeyboardButton(
                "60 FPS",
                callback_data="setfps_60"
            ),
            InlineKeyboardButton(
                "120 FPS",
                callback_data="setfps_120"
            ),
        ],

        [
            InlineKeyboardButton(
                "✏️ مخصص",
                callback_data="custom_fps"
            ),
        ],

        [
            InlineKeyboardButton(
                "↩️ رجوع",
                callback_data="back"
            ),
        ],
    ])


# ============================================================
# CALLBACK HANDLER
# ============================================================

async def callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    if not query:
        return

    await query.answer()

    user_id = (
        query.from_user.id
    )

    action = query.data

    settings = get_settings(
        user_id
    )

    # ========================================================
    # CANCEL
    # ========================================================

    if action == "cancel":

        context.user_data.pop(
            "awaiting",
            None
        )

        await query.message.reply_text(
            "❌ تم إلغاء العملية.",
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # BACK
    # ========================================================

    if action == "back":

        context.user_data.pop(
            "awaiting",
            None
        )

        await query.message.reply_text(
            "🎛 لوحة التحكم:",
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # HELP
    # ========================================================

    if action == "help":

        await query.message.reply_text(
            HELP_TEXT,
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # FILE REQUIRED
    # ========================================================

    if action not in {
        "help",
        "reset",
        "info",
        "settings",
    }:

        if user_id not in USER_FILES:

            await query.message.reply_text(
                "⚠️ أرسل ملف فيديو أو صوت أولًا.",
                reply_markup=main_keyboard(),
            )

            return

    # ========================================================
    # RESET
    # ========================================================

    if action == "reset":

        USER_SETTINGS[user_id] = (
            default_settings()
        )

        context.user_data.pop(
            "awaiting",
            None
        )

        await query.message.reply_text(
            "🔄 تمت إعادة ضبط جميع الإعدادات.",
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # INFO
    # ========================================================

    if action == "info":

        if user_id not in USER_FILES:

            await query.message.reply_text(
                "⚠️ لا يوجد ملف حالي.",
                reply_markup=main_keyboard(),
            )

            return

        try:

            info = await asyncio.to_thread(
                inspect_media,
                USER_FILES[user_id],
            )

            text = (
                "ℹ️ معلومات الملف\n\n"
                f"📄 {info['filename']}\n"
                f"📦 Format: {info['format']}\n"
                f"⏱ Duration: "
                f"{info['duration'] or 'غير معروف'} sec\n"
                f"💾 Bitrate: "
                f"{info['bitrate'] or 'غير معروف'}\n\n"
            )

            for stream in info["streams"]:

                if stream["type"] == "video":

                    text += (
                        "🎥 Video\n"
                        f"Codec: {stream['codec']}\n"
                        f"Resolution: "
                        f"{stream.get('width')}×"
                        f"{stream.get('height')}\n"
                        f"FPS: {stream.get('fps')}\n\n"
                    )

                elif stream["type"] == "audio":

                    text += (
                        "🎵 Audio\n"
                        f"Codec: {stream['codec']}\n"
                        f"Sample Rate: "
                        f"{stream.get('sample_rate')}\n"
                        f"Channels: "
                        f"{stream.get('channels')}\n\n"
                    )

            await query.message.reply_text(
                text,
                reply_markup=main_keyboard(),
            )

        except Exception as e:

            await query.message.reply_text(
                f"❌ تعذر قراءة الملف:\n{str(e)[:3000]}",
                reply_markup=main_keyboard(),
            )

        return

    # ========================================================
    # SETTINGS
    # ========================================================

    if action == "settings":

        await query.message.reply_text(
            settings_summary(
                user_id
            ),
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # TRIM
    # ========================================================

    if action == "trim":

        context.user_data[
            "awaiting"
        ] = "trim"

        await query.message.reply_text(
            (
                "✂️ قص الفيديو\n\n"
                "أرسل البداية والنهاية بالثواني.\n\n"
                "مثال:\n"
                "10 60\n\n"
                "يعني من الثانية 10\n"
                "حتى الثانية 60."
            ),
            reply_markup=back_keyboard(),
        )

        return

    # ========================================================
    # VOLUME
    # ========================================================

    if action == "volume":

        context.user_data[
            "awaiting"
        ] = "volume"

        await query.message.reply_text(
            (
                "🔊 التحكم في الصوت\n\n"
                "أرسل معامل الصوت.\n\n"
                "0.5 = نصف الصوت\n"
                "1.0 = الصوت الأصلي\n"
                "1.5 = زيادة 50%\n"
                "2.0 = مضاعفة"
            ),
            reply_markup=back_keyboard(),
        )

        return

    # ========================================================
    # SPEED
    # ========================================================

    if action == "speed":

        await query.message.reply_text(
            "⚡ اختر سرعة المعالجة:",
            reply_markup=speed_keyboard(),
        )

        return

    # ========================================================
    # RESOLUTION
    # ========================================================

    if action == "resolution":

        await query.message.reply_text(
            "📐 اختر الدقة:",
            reply_markup=resolution_keyboard(),
        )

        return

    # ========================================================
    # FPS
    # ========================================================

    if action == "fps":

        await query.message.reply_text(
            "🎞 اختر FPS:",
            reply_markup=fps_keyboard(),
        )

        return

    # ========================================================
    # FORMAT
    # ========================================================

    if action == "format":

        await query.message.reply_text(
            "🔄 اختر صيغة الإخراج:",
            reply_markup=format_keyboard(),
        )

        return

    # ========================================================
    # EXTRACT AUDIO
    # ========================================================

    if action == "extract":

        settings[
            "extract_audio"
        ] = True

        settings[
            "remove_audio"
        ] = False

        settings[
            "output_format"
        ] = "mp3"

        await query.message.reply_text(
            (
                "🎵 تم اختيار استخراج الصوت.\n\n"
                "📦 الصيغة الحالية: MP3"
            ),
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # MUTE
    # ========================================================

    if action == "mute":

        settings[
            "remove_audio"
        ] = True

        settings[
            "extract_audio"
        ] = False

        await query.message.reply_text(
            "🔇 سيتم حذف مسار الصوت من الفيديو.",
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # AUDIO BITRATE
    # ========================================================

    if action == "audio_bitrate":

        context.user_data[
            "awaiting"
        ] = "audio_bitrate"

        await query.message.reply_text(
            (
                "🎚 Audio Bitrate\n\n"
                "أرسل القيمة.\n\n"
                "مثال:\n"
                "96k\n"
                "128k\n"
                "192k\n"
                "320k"
            ),
            reply_markup=back_keyboard(),
        )

        return

    # ========================================================
    # VIDEO BITRATE
    # ========================================================

    if action == "video_bitrate":

        context.user_data[
            "awaiting"
        ] = "video_bitrate"

        await query.message.reply_text(
            (
                "🎚 Video Bitrate\n\n"
                "مثال:\n"
                "1M\n"
                "2M\n"
                "4M\n"
                "8M"
            ),
            reply_markup=back_keyboard(),
        )

        return

    # ========================================================
    # SAMPLE RATE
    # ========================================================

    if action == "sample":

        context.user_data[
            "awaiting"
        ] = "sample"

        await query.message.reply_text(
            (
                "🎛 Sample Rate\n\n"
                "أرسل القيمة.\n\n"
                "44100\n"
                "48000\n"
                "96000"
            ),
            reply_markup=back_keyboard(),
        )

        return

    # ========================================================
    # CUSTOM INPUTS
    # ========================================================

    if action == "custom_speed":

        context.user_data[
            "awaiting"
        ] = "speed"

        await query.message.reply_text(
            "⚡ أرسل السرعة، مثل:\n1.75",
            reply_markup=back_keyboard(),
        )

        return

    if action == "custom_resolution":

        context.user_data[
            "awaiting"
        ] = "resolution"

        await query.message.reply_text(
            "📐 أرسل العرض والارتفاع.\n\nمثال:\n1920 1080",
            reply_markup=back_keyboard(),
        )

        return

    if action == "custom_fps":

        context.user_data[
            "awaiting"
        ] = "fps"

        await query.message.reply_text(
            "🎞 أرسل FPS.\n\nمثال:\n30",
            reply_markup=back_keyboard(),
        )

        return

    if action == "custom_format":

        context.user_data[
            "awaiting"
        ] = "format"

        await query.message.reply_text(
            "🔄 أرسل الصيغة.\n\nمثال:\nmp4",
            reply_markup=back_keyboard(),
        )

        return

    # ========================================================
    # PRESET RESOLUTION
    # ========================================================

    if action.startswith(
        "setres_"
    ):

        value = action.replace(
            "setres_",
            ""
        )

        width, height = (
            value.split("_")
        )

        settings["width"] = int(
            width
        )

        settings["height"] = int(
            height
        )

        await query.message.reply_text(
            (
                f"✅ تم اختيار الدقة "
                f"{width}×{height}"
            ),
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # PRESET SPEED
    # ========================================================

    if action.startswith(
        "setspeed_"
    ):

        value = action.replace(
            "setspeed_",
            ""
        )

        settings["speed"] = float(
            value
        )

        await query.message.reply_text(
            f"✅ السرعة: {value}×",
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # PRESET FPS
    # ========================================================

    if action.startswith(
        "setfps_"
    ):

        value = action.replace(
            "setfps_",
            ""
        )

        settings["fps"] = float(
            value
        )

        await query.message.reply_text(
            f"✅ تم ضبط FPS على {value}",
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # PRESET FORMAT
    # ========================================================

    if action.startswith(
        "setformat_"
    ):

        value = action.replace(
            "setformat_",
            ""
        )

        settings[
            "output_format"
        ] = value

        if value in {
            "mp3",
            "wav",
            "flac",
            "m4a",
        }:

            settings[
                "extract_audio"
            ] = True

            settings[
                "remove_audio"
            ] = False

        else:

            settings[
                "extract_audio"
            ] = False

        await query.message.reply_text(
            (
                f"✅ صيغة الإخراج: "
                f"{value.upper()}"
            ),
            reply_markup=main_keyboard(),
        )

        return

    # ========================================================
    # PROCESS
    # ========================================================

    if action == "process":

        await process_telegram_file(
            query,
            user_id,
        )

        return


# ============================================================
# TEXT SETTINGS
# ============================================================

async def text_settings_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not update.message:
        return

    awaiting = context.user_data.get(
        "awaiting"
    )

    if not awaiting:

        await update.message.reply_text(
            (
                "ℹ️ استخدم الأزرار الموجودة في لوحة التحكم.\n\n"
                "أو أرسل /start للعودة."
            ),
            reply_markup=main_keyboard(),
        )

        return

    user_id = (
        update.message.from_user.id
    )

    settings = get_settings(
        user_id
    )

    text = (
        update.message.text
        or ""
    ).strip()

    try:

        # ====================================================
        # TRIM
        # ====================================================

        if awaiting == "trim":

            values = text.split()

            if len(values) != 2:

                raise ValueError(
                    "أدخل البداية والنهاية."
                )

            start = float(
                values[0]
            )

            end = float(
                values[1]
            )

            if start < 0:
                raise ValueError()

            if end <= start:
                raise ValueError()

            settings["start"] = start

            settings["end"] = end

        # ====================================================
        # VOLUME
        # ====================================================

        elif awaiting == "volume":

            value = float(
                text
            )

            if value < 0:
                raise ValueError()

            settings[
                "volume"
            ] = value

        # ====================================================
        # SPEED
        # ====================================================

        elif awaiting == "speed":

            value = float(
                text
            )

            if value <= 0:
                raise ValueError()

            if value > 10:
                raise ValueError()

            settings[
                "speed"
            ] = value

        # ====================================================
        # RESOLUTION
        # ====================================================

        elif awaiting == "resolution":

            values = text.split()

            if len(values) != 2:

                raise ValueError()

            width = int(
                values[0]
            )

            height = int(
                values[1]
            )

            if (
                width < 16
                or height < 16
            ):

                raise ValueError()

            settings[
                "width"
            ] = width

            settings[
                "height"
            ] = height

        # ====================================================
        # FPS
        # ====================================================

        elif awaiting == "fps":

            value = float(
                text
            )

            if (
                value <= 0
                or value > 240
            ):

                raise ValueError()

            settings[
                "fps"
            ] = value

        # ====================================================
        # FORMAT
        # ====================================================

        elif awaiting == "format":

            value = (
                text
                .lower()
                .replace(
                    ".",
                    ""
                )
            )

            allowed = {
                "mp4",
                "mkv",
                "webm",
                "mov",
                "mp3",
                "wav",
                "flac",
                "m4a",
            }

            if value not in allowed:

                raise ValueError()

            settings[
                "output_format"
            ] = value

            if value in {
                "mp3",
                "wav",
                "flac",
                "m4a",
            }:

                settings[
                    "extract_audio"
                ] = True

                settings[
                    "remove_audio"
                ] = False

            else:

                settings[
                    "extract_audio"
                ] = False

        # ====================================================
        # AUDIO BITRATE
        # ====================================================

        elif awaiting == "audio_bitrate":

            value = text

            if parse_bitrate(
                value
            ) is None:

                raise ValueError()

            settings[
                "audio_bitrate"
            ] = value

        # ====================================================
        # VIDEO BITRATE
        # ====================================================

        elif awaiting == "video_bitrate":

            value = text

            if parse_bitrate(
                value
            ) is None:

                raise ValueError()

            settings[
                "video_bitrate"
            ] = value

        # ====================================================
        # SAMPLE RATE
        # ====================================================

        elif awaiting == "sample":

            value = int(
                text
            )

            allowed_rates = {
                8000,
                16000,
                22050,
                24000,
                32000,
                44100,
                48000,
                88200,
                96000,
            }

            if value not in allowed_rates:

                raise ValueError()

            settings[
                "sample_rate"
            ] = value

        context.user_data.pop(
            "awaiting",
            None
        )

        await update.message.reply_text(
            (
                "✅ تم حفظ الإعداد.\n\n"
                "يمكنك الآن اختيار إعداد آخر "
                "أو الضغط على 🚀 تنفيذ."
            ),
            reply_markup=main_keyboard(),
        )

    except Exception:

        await update.message.reply_text(
            (
                "❌ القيمة غير صحيحة.\n\n"
                "أعد إرسالها بالقيمة المطلوبة."
            ),
            reply_markup=back_keyboard(),
        )


# ============================================================
# TELEGRAM PROCESS
# ============================================================

async def process_telegram_file(
    query,
    user_id,
):

    if user_id not in USER_FILES:

        await query.message.reply_text(
            "⚠️ أرسل ملفًا أولًا.",
            reply_markup=main_keyboard(),
        )

        return

    input_path = Path(
        USER_FILES[user_id]
    )

    settings = get_settings(
        user_id
    )

    if not input_path.exists():

        USER_FILES.pop(
            user_id,
            None
        )

        await query.message.reply_text(
            "❌ الملف لم يعد موجودًا على السيرفر.\n"
            "أرسل الملف مرة أخرى.",
            reply_markup=main_keyboard(),
        )

        return

    await query.message.reply_text(
        (
            "🚀 بدأت المعالجة...\n\n"
            "🧠 Engine: PyAV\n"
            "🚫 External FFmpeg: لا\n\n"
            "⚙️ الإعدادات:\n"
            f"{settings_summary(user_id)}\n\n"
            "⏳ انتظر حتى اكتمال العملية..."
        ),
    )

    extension = (
        settings.get(
            "output_format"
        )
        or input_path.suffix.lstrip(".")
        or "mp4"
    )

    if settings.get(
        "extract_audio"
    ):

        extension = (
            settings.get(
                "output_format"
            )
            or "mp3"
        )

    extension = (
        extension
        .lower()
        .replace(
            ".",
            ""
        )
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"{input_path.stem}"
            f"_result_"
            f"{uuid.uuid4().hex[:8]}"
            f".{extension}"
        )
    )

    try:

        await asyncio.to_thread(
            process_media_pyav,
            input_path,
            output_path,
            settings,
        )

        if not output_path.exists():

            raise RuntimeError(
                "لم يتم إنشاء الملف الناتج."
            )

        file_size = (
            output_path.stat().st_size
        )

        size_mb = (
            file_size
            / 1024
            / 1024
        )

        await query.message.reply_text(
            (
                "✅ اكتملت المعالجة بنجاح.\n\n"
                f"📦 الحجم: {size_mb:.2f} MB\n"
                "📤 جاري إرسال الملف..."
            )
        )

        with open(
            output_path,
            "rb"
        ) as file:

            await query.message.reply_document(
                document=file,
                filename=output_path.name,
                caption=(
                    "🎬 PyAV Media Studio\n"
                    "✅ تم إنشاء الملف بنجاح."
                ),
            )

    except Exception as e:

        logger.exception(
            "Telegram processing error"
        )

        await query.message.reply_text(
            (
                "❌ حدث خطأ أثناء المعالجة.\n\n"
                f"{str(e)[:4000]}"
            ),
            reply_markup=main_keyboard(),
        )

    finally:

        cleanup_file(
            output_path
        )


# ============================================================
# HANDLER REGISTRATION
# ============================================================

def register_handlers(
    application: Application
):

    application.add_handler(
        CommandHandler(
            "start",
            start_command
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            callback_handler
        )
    )

    application.add_handler(
        MessageHandler(
            (
                filters.Document.ALL
                | filters.VIDEO
                | filters.AUDIO
                | filters.VOICE
            ),
            handle_media,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_settings_handler,
        )
    )


# ============================================================
# OPTIONAL CLEANUP
# ============================================================

async def cleanup_old_files():

    while True:

        try:

            await asyncio.sleep(
                30 * 60
            )

            for directory in (
                INPUT_DIR,
                OUTPUT_DIR,
            ):

                for file in directory.iterdir():

                    try:

                        if not file.is_file():
                            continue

                        age = (
                            __import__(
                                "time"
                            ).time()
                            - file.stat().st_mtime
                        )

                        if age > (
                            2 * 60 * 60
                        ):

                            file.unlink(
                                missing_ok=True
                            )

                    except Exception:
                        pass

        except asyncio.CancelledError:

            break

        except Exception:

            logger.exception(
                "Cleanup worker error"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )
