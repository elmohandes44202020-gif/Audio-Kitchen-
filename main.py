import os
import uuid
import asyncio
import logging
import mimetypes
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import av
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

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
# CONFIGURATION
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

PORT = int(os.getenv("PORT", "8000"))

BASE_DIR = Path(os.getenv("PYAV_MEDIA_DIR", "/tmp/pyav_media"))

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("pyav-server")


# ============================================================
# TELEGRAM APPLICATION
# ============================================================

telegram_application: Optional[Application] = None


# ============================================================
# USER STORAGE
# ============================================================

USER_FILES = {}
USER_SETTINGS = {}


DEFAULT_SETTINGS = {
    "operation": "convert",

    "trim_start": None,
    "trim_end": None,

    "volume": 1.0,
    "speed": 1.0,

    "width": None,
    "height": None,

    "fps": None,

    "sample_rate": None,
    "channels": None,

    "video_bitrate": None,
    "audio_bitrate": None,

    "fade_in": 0,
    "fade_out": 0,

    "remove_audio": False,
    "extract_audio": False,

    "output_format": None,
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_user_settings(user_id: int):
    if user_id not in USER_SETTINGS:
        USER_SETTINGS[user_id] = DEFAULT_SETTINGS.copy()

    return USER_SETTINGS[user_id]


def reset_user_settings(user_id: int):
    USER_SETTINGS[user_id] = DEFAULT_SETTINGS.copy()


def safe_filename(name: str) -> str:
    if not name:
        return "media"

    name = Path(name).name

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "._- "
    )

    name = "".join(c if c in allowed else "_" for c in name)

    return name[:180] or "media"


def unique_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    original = Path(filename)

    stem = original.stem
    suffix = original.suffix

    path = directory / filename

    counter = 1

    while path.exists():
        path = directory / f"{stem}_{counter}{suffix}"
        counter += 1

    return path


def cleanup_file(path):
    try:
        if path and Path(path).exists():
            Path(path).unlink()
    except Exception:
        logger.exception("Could not delete file")


def parse_bitrate(value):
    if value is None:
        return None

    if isinstance(value, int):
        return value

    value = str(value).strip().lower()

    try:
        if value.endswith("k"):
            return int(float(value[:-1]) * 1000)

        if value.endswith("m"):
            return int(float(value[:-1]) * 1000000)

        return int(value)

    except Exception:
        raise ValueError("Invalid bitrate")


def parse_float(value, minimum=None, maximum=None):
    number = float(value)

    if minimum is not None and number < minimum:
        raise ValueError(f"Value must be >= {minimum}")

    if maximum is not None and number > maximum:
        raise ValueError(f"Value must be <= {maximum}")

    return number


def parse_int(value, minimum=None, maximum=None):
    number = int(value)

    if minimum is not None and number < minimum:
        raise ValueError(f"Value must be >= {minimum}")

    if maximum is not None and number > maximum:
        raise ValueError(f"Value must be <= {maximum}")

    return number


# ============================================================
# MEDIA INSPECTION
# ============================================================

def inspect_media(path: str):
    result = {
        "filename": Path(path).name,
        "format": None,
        "duration": None,
        "bitrate": None,
        "size": None,
        "video": [],
        "audio": [],
    }

    path_obj = Path(path)

    if path_obj.exists():
        result["size"] = path_obj.stat().st_size

    container = None

    try:
        container = av.open(path)

        result["format"] = (
            container.format.name
            if container.format
            else None
        )

        if container.duration is not None:
            result["duration"] = container.duration / av.time_base

        if container.bit_rate:
            result["bitrate"] = container.bit_rate

        for stream in container.streams:

            if stream.type == "video":
                result["video"].append(
                    {
                        "index": stream.index,
                        "codec": (
                            stream.codec_context.name
                            if stream.codec_context
                            else None
                        ),
                        "width": stream.codec_context.width,
                        "height": stream.codec_context.height,
                        "fps": (
                            float(stream.average_rate)
                            if stream.average_rate
                            else None
                        ),
                        "pix_fmt": (
                            stream.codec_context.pix_fmt
                            if stream.codec_context
                            else None
                        ),
                    }
                )

            elif stream.type == "audio":
                result["audio"].append(
                    {
                        "index": stream.index,
                        "codec": (
                            stream.codec_context.name
                            if stream.codec_context
                            else None
                        ),
                        "sample_rate": stream.codec_context.sample_rate,
                        "channels": stream.codec_context.channels,
                        "layout": (
                            stream.layout.name
                            if stream.layout
                            else None
                        ),
                    }
                )

    finally:
        if container:
            container.close()

    return result


# ============================================================
# CODEC SELECTION
# ============================================================

def select_video_codec(output_format: str):
    output_format = output_format.lower()

    if output_format in {"mp4", "m4v", "mov"}:
        return "libx264"

    if output_format == "webm":
        return "libvpx"

    if output_format in {"mkv", "matroska"}:
        return "libx264"

    return "libx264"


def select_audio_codec(output_format: str):
    output_format = output_format.lower()

    if output_format == "mp3":
        return "libmp3lame"

    if output_format in {"ogg", "opus"}:
        return "libopus"

    if output_format == "flac":
        return "flac"

    if output_format == "wav":
        return "pcm_s16le"

    if output_format in {"mp4", "m4a", "mov"}:
        return "aac"

    return "aac"


# ============================================================
# VIDEO FRAME PROCESSING
# ============================================================

def resize_video_frame(frame, width=None, height=None):

    if not width and not height:
        return frame

    target_width = width or frame.width
    target_height = height or frame.height

    return frame.reformat(
        width=target_width,
        height=target_height,
        format="yuv420p",
    )


# ============================================================
# AUDIO FRAME PROCESSING
# ============================================================

def process_audio_frame(frame, volume=1.0):

    if volume == 1.0:
        return frame

    try:
        array = frame.to_ndarray()

        array = array.astype(np.float32)

        array *= float(volume)

        array = np.clip(
            array,
            -32768,
            32767,
        )

        array = array.astype(np.int16)

        new_frame = av.AudioFrame.from_ndarray(
            array,
            layout=frame.layout.name,
        )

        new_frame.sample_rate = frame.sample_rate

        if frame.pts is not None:
            new_frame.pts = frame.pts

        new_frame.time_base = frame.time_base

        return new_frame

    except Exception:
        logger.exception(
            "Audio processing failed; returning original frame"
        )

        return frame


# ============================================================
# MAIN PYAV PROCESSOR
# ============================================================

def process_media_pyav(
    input_path: str,
    output_path: str,
    settings: dict,
):

    input_container = None
    output_container = None

    try:

        input_container = av.open(input_path)

        output_format = (
            settings.get("output_format")
            or Path(output_path).suffix.lstrip(".")
            or "mp4"
        )

        output_format = output_format.lower()

        # ----------------------------------------------------
        # OUTPUT CONTAINER
        # ----------------------------------------------------

        output_container = av.open(
            output_path,
            mode="w",
            format=output_format,
        )

        # ----------------------------------------------------
        # INPUT STREAMS
        # ----------------------------------------------------

        input_video = None
        input_audio = None

        for stream in input_container.streams:

            if stream.type == "video" and input_video is None:
                input_video = stream

            elif stream.type == "audio" and input_audio is None:
                input_audio = stream

        # ----------------------------------------------------
        # SETTINGS
        # ----------------------------------------------------

        remove_audio = bool(
            settings.get("remove_audio", False)
        )

        extract_audio = bool(
            settings.get("extract_audio", False)
        )

        width = settings.get("width")
        height = settings.get("height")

        fps = settings.get("fps")

        volume = float(
            settings.get("volume", 1.0)
        )

        speed = float(
            settings.get("speed", 1.0)
        )

        trim_start = settings.get("trim_start")
        trim_end = settings.get("trim_end")

        audio_sample_rate = settings.get(
            "sample_rate"
        )

        video_bitrate = parse_bitrate(
            settings.get("video_bitrate")
        )

        audio_bitrate = parse_bitrate(
            settings.get("audio_bitrate")
        )

        # ----------------------------------------------------
        # EXTRACT AUDIO MODE
        # ----------------------------------------------------

        if extract_audio:

            if input_audio is None:
                raise RuntimeError(
                    "No audio stream found."
                )

            audio_codec = select_audio_codec(
                output_format
            )

            audio_rate = (
                audio_sample_rate
                or input_audio.codec_context.sample_rate
                or 44100
            )

            audio_stream = output_container.add_stream(
                audio_codec,
                rate=audio_rate,
            )

            if audio_bitrate:
                audio_stream.bit_rate = audio_bitrate

        else:

            audio_stream = None

            if input_audio is not None and not remove_audio:

                audio_codec = select_audio_codec(
                    output_format
                )

                audio_rate = (
                    audio_sample_rate
                    or input_audio.codec_context.sample_rate
                    or 44100
                )

                audio_stream = output_container.add_stream(
                    audio_codec,
                    rate=audio_rate,
                )

                if audio_bitrate:
                    audio_stream.bit_rate = audio_bitrate

        # ----------------------------------------------------
        # VIDEO OUTPUT STREAM
        # ----------------------------------------------------

        video_stream = None

        if (
            input_video is not None
            and not extract_audio
        ):

            video_codec = select_video_codec(
                output_format
            )

            input_width = (
                input_video.codec_context.width
            )

            input_height = (
                input_video.codec_context.height
            )

            output_width = (
                width or input_width
            )

            output_height = (
                height or input_height
            )

            source_rate = (
                input_video.average_rate
            )

            output_rate = fps or source_rate

            if output_rate:
                video_stream = output_container.add_stream(
                    video_codec,
                    rate=output_rate,
                )
            else:
                video_stream = output_container.add_stream(
                    video_codec
                )

            video_stream.width = output_width
            video_stream.height = output_height

            if video_codec in {"libx264", "h264"}:
                video_stream.pix_fmt = "yuv420p"

            if video_bitrate:
                video_stream.bit_rate = video_bitrate

        # ----------------------------------------------------
        # DECODE + ENCODE
        # ----------------------------------------------------

        for packet in input_container.demux():

            if packet.stream.type not in {
                "video",
                "audio",
            }:
                continue

            try:
                frames = packet.decode()
            except Exception:
                logger.exception(
                    "Could not decode packet"
                )
                continue

            for frame in frames:

                # ==========================================
                # VIDEO
                # ==========================================

                if (
                    frame.type == "video"
                    and video_stream is not None
                ):

                    timestamp = None

                    if frame.pts is not None:
                        timestamp = float(
                            frame.pts
                            * frame.time_base
                        )

                    # ------------------------------
                    # TRIM
                    # ------------------------------

                    if (
                        timestamp is not None
                        and trim_start is not None
                        and timestamp < trim_start
                    ):
                        continue

                    if (
                        timestamp is not None
                        and trim_end is not None
                        and timestamp > trim_end
                    ):
                        continue

                    # ------------------------------
                    # RESIZE
                    # ------------------------------

                    frame = resize_video_frame(
                        frame,
                        width,
                        height,
                    )

                    # ------------------------------
                    # PIXEL FORMAT
                    # ------------------------------

                    if video_stream.codec_context.name in {
                        "libx264",
                        "h264",
                    }:

                        frame = frame.reformat(
                            format="yuv420p"
                        )

                    # ------------------------------
                    # SPEED
                    # ------------------------------

                    if speed != 1.0:
                        if frame.pts is not None:
                            frame.pts = int(
                                frame.pts / speed
                            )

                    try:

                        for encoded_packet in video_stream.encode(
                            frame
                        ):
                            output_container.mux(
                                encoded_packet
                            )

                    except Exception:
                        logger.exception(
                            "Video encoding error"
                        )

                # ==========================================
                # AUDIO
                # ==========================================

                elif (
                    frame.type == "audio"
                    and audio_stream is not None
                ):

                    timestamp = None

                    if frame.pts is not None:
                        timestamp = float(
                            frame.pts
                            * frame.time_base
                        )

                    # ------------------------------
                    # TRIM
                    # ------------------------------

                    if (
                        timestamp is not None
                        and trim_start is not None
                        and timestamp < trim_start
                    ):
                        continue

                    if (
                        timestamp is not None
                        and trim_end is not None
                        and timestamp > trim_end
                    ):
                        continue

                    # ------------------------------
                    # VOLUME
                    # ------------------------------

                    frame = process_audio_frame(
                        frame,
                        volume,
                    )

                    # ------------------------------
                    # SPEED
                    #
                    # We intentionally don't attempt
                    # dangerous audio time-stretching
                    # here. Volume and trimming remain
                    # fully PyAV based.
                    # ------------------------------

                    try:

                        for encoded_packet in audio_stream.encode(
                            frame
                        ):
                            output_container.mux(
                                encoded_packet
                            )

                    except Exception:
                        logger.exception(
                            "Audio encoding error"
                        )

        # ----------------------------------------------------
        # FLUSH VIDEO
        # ----------------------------------------------------

        if video_stream is not None:

            try:

                for encoded_packet in video_stream.encode():
                    output_container.mux(
                        encoded_packet
                    )

            except Exception:
                logger.exception(
                    "Video encoder flush error"
                )

        # ----------------------------------------------------
        # FLUSH AUDIO
        # ----------------------------------------------------

        if audio_stream is not None:

            try:

                for encoded_packet in audio_stream.encode():
                    output_container.mux(
                        encoded_packet
                    )

            except Exception:
                logger.exception(
                    "Audio encoder flush error"
                )

        # ----------------------------------------------------
        # CLOSE
        # ----------------------------------------------------

        output_container.close()
        output_container = None

        input_container.close()
        input_container = None

        if not Path(output_path).exists():
            raise RuntimeError(
                "Output file was not created."
            )

        if Path(output_path).stat().st_size <= 0:
            raise RuntimeError(
                "Output file is empty."
            )

        return output_path

    except Exception:

        logger.exception(
            "PyAV processing failed"
        )

        raise

    finally:

        try:
            if input_container:
                input_container.close()
        except Exception:
            pass

        try:
            if output_container:
                output_container.close()
        except Exception:
            pass


# ============================================================
# FASTAPI
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global telegram_application

    logger.info("========================================")
    logger.info("Starting PyAV Media Server")
    logger.info("========================================")

    logger.info(
        "External FFmpeg executable: DISABLED"
    )

    logger.info(
        "PyAV media engine: ENABLED"
    )

    if BOT_TOKEN:

        try:

            telegram_application = (
                Application.builder()
                .token(BOT_TOKEN)
                .build()
            )

            register_bot_handlers(
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
            "BOT_TOKEN is not configured. "
            "Telegram bot is disabled."
        )

    yield

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    if telegram_application:

        try:

            if telegram_application.updater:

                await telegram_application.updater.stop()

            await telegram_application.stop()

            await telegram_application.shutdown()

            logger.info(
                "Telegram bot stopped."
            )

        except Exception:

            logger.exception(
                "Telegram shutdown error"
            )


app = FastAPI(
    title="PyAV Professional Media Studio",
    version="3.0.0",
    description=(
        "Professional media processing API powered by PyAV. "
        "No external FFmpeg executable is called."
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
        "version": "3.0.0",
        "engine": "PyAV",
        "external_ffmpeg": False,
        "telegram_bot": bool(
            telegram_application
        ),
        "endpoints": {
            "health": "/health",
            "bot_status": "/bot-status",
            "media_info": "/info",
            "process": "/process",
        },
    }


# ============================================================
# HEALTH
# ============================================================

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
# BOT STATUS
# ============================================================

@app.get("/bot-status")
async def bot_status():

    return {
        "configured": bool(BOT_TOKEN),
        "running": bool(
            telegram_application
        ),
        "status": (
            "running"
            if telegram_application
            else (
                "token_missing"
                if not BOT_TOKEN
                else "not_running"
            )
        ),
    }


# ============================================================
# INFO API
# ============================================================

@app.post("/info")
async def media_info(
    file: UploadFile = File(...)
):

    suffix = (
        Path(file.filename or "media")
        .suffix
        or ".bin"
    )

    temp_path = (
        INPUT_DIR
        / f"{uuid.uuid4().hex}{suffix}"
    )

    try:

        with open(temp_path, "wb") as output:
            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output.write(chunk)

        result = inspect_media(
            str(temp_path)
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    finally:

        cleanup_file(temp_path)


# ============================================================
# PROCESS API
# ============================================================

@app.post("/process")
async def process_api(
    file: UploadFile = File(...),

    output_format: str = "mp4",

    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,

    volume: float = 1.0,
    speed: float = 1.0,

    width: Optional[int] = None,
    height: Optional[int] = None,

    fps: Optional[float] = None,

    sample_rate: Optional[int] = None,

    video_bitrate: Optional[str] = None,
    audio_bitrate: Optional[str] = None,

    remove_audio: bool = False,
    extract_audio: bool = False,
):

    if speed <= 0:
        raise HTTPException(
            status_code=400,
            detail="Speed must be greater than 0.",
        )

    if volume < 0:
        raise HTTPException(
            status_code=400,
            detail="Volume cannot be negative.",
        )

    allowed_formats = {
        "mp4",
        "mkv",
        "mov",
        "webm",
        "m4a",
        "mp3",
        "wav",
        "flac",
        "ogg",
    }

    output_format = output_format.lower().strip()

    if output_format not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported output format. "
                f"Allowed: {sorted(allowed_formats)}"
            ),
        )

    input_suffix = (
        Path(file.filename or "media")
        .suffix
        or ".bin"
    )

    input_path = (
        INPUT_DIR
        / f"{uuid.uuid4().hex}{input_suffix}"
    )

    original_name = safe_filename(
        file.filename or "media"
    )

    output_name = (
        Path(original_name).stem
        + "_processed."
        + output_format
    )

    output_path = unique_path(
        OUTPUT_DIR,
        output_name,
    )

    try:

        with open(input_path, "wb") as output:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output.write(chunk)

        settings = {
            "output_format": output_format,

            "trim_start": trim_start,
            "trim_end": trim_end,

            "volume": volume,
            "speed": speed,

            "width": width,
            "height": height,

            "fps": fps,

            "sample_rate": sample_rate,

            "video_bitrate": video_bitrate,
            "audio_bitrate": audio_bitrate,

            "remove_audio": remove_audio,
            "extract_audio": extract_audio,
        }

        await asyncio.to_thread(
            process_media_pyav,
            str(input_path),
            str(output_path),
            settings,
        )

        return FileResponse(
            path=str(output_path),
            filename=output_name,
            media_type=(
                mimetypes.guess_type(
                    output_name
                )[0]
                or "application/octet-stream"
            ),
        )

    except Exception as exc:

        cleanup_file(output_path)

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        cleanup_file(input_path)


# ============================================================
# TELEGRAM UI
# ============================================================

def main_keyboard():

    keyboard = [

        [
            InlineKeyboardButton(
                "✂️ قص الفيديو",
                callback_data="trim",
            ),
            InlineKeyboardButton(
                "🔊 الصوت",
                callback_data="volume",
            ),
        ],

        [
            InlineKeyboardButton(
                "⚡ السرعة",
                callback_data="speed",
            ),
            InlineKeyboardButton(
                "📐 الدقة",
                callback_data="resolution",
            ),
        ],

        [
            InlineKeyboardButton(
                "🎞 FPS",
                callback_data="fps",
            ),
            InlineKeyboardButton(
                "🔄 الصيغة",
                callback_data="format",
            ),
        ],

        [
            InlineKeyboardButton(
                "🎚 Bitrate",
                callback_data="bitrate",
            ),
            InlineKeyboardButton(
                "🎛 Sample Rate",
                callback_data="sample",
            ),
        ],

        [
            InlineKeyboardButton(
                "🎵 استخراج الصوت",
                callback_data="extract",
            ),
            InlineKeyboardButton(
                "🔇 كتم الصوت",
                callback_data="mute",
            ),
        ],

        [
            InlineKeyboardButton(
                "ℹ️ معلومات الملف",
                callback_data="info",
            ),
        ],

        [
            InlineKeyboardButton(
                "🚀 تنفيذ المعالجة",
                callback_data="process",
            ),
        ],

        [
            InlineKeyboardButton(
                "🔄 إعادة ضبط",
                callback_data="reset",
            ),
        ],

    ]

    return InlineKeyboardMarkup(keyboard)


# ============================================================
# TELEGRAM START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    user = update.effective_user

    if user:

        reset_user_settings(
            user.id
        )

    text = """
🎬 مرحبًا بك في PyAV Media Studio

محرك معالجة احترافي للفيديو والصوت يعمل باستخدام:

⚙️ PyAV
🚫 بدون تشغيل برنامج FFmpeg الخارجي

━━━━━━━━━━━━━━━━━━

📤 أرسل لي:

🎥 فيديو
🎵 ملف صوتي
📄 Video Document
📎 Audio Document

ثم ستظهر لك لوحة التحكم.

━━━━━━━━━━━━━━━━━━

🎛 الأدوات:

✂️ قص
🔊 التحكم في الصوت
⚡ السرعة
📐 تغيير الدقة
🎞 تغيير FPS
🔄 تغيير الصيغة
🎚 Bitrate
🎛 Sample Rate
🎵 استخراج الصوت
🔇 كتم الصوت
ℹ️ معلومات الملف

━━━━━━━━━━━━━━━━━━

بعد اختيار الإعدادات اضغط:

🚀 تنفيذ المعالجة

وسيتم إرسال الملف الناتج لك.
"""

    await update.message.reply_text(
        text,
        reply_markup=main_keyboard(),
    )


# ============================================================
# HELP
# ============================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    text = """
📚 شرح الاستخدام

1️⃣ أرسل الفيديو أو الملف الصوتي للبوت.

2️⃣ اختر الوظيفة المطلوبة من الأزرار.

3️⃣ أدخل القيمة عندما يطلبها البوت.

4️⃣ يمكنك تغيير أكثر من إعداد.

5️⃣ اضغط 🚀 تنفيذ المعالجة.

━━━━━━━━━━━━━━━━━━

مثال للقص:

✂️ قص الفيديو

ثم أرسل:

10 60

وهذا يعني:

⏱ البداية = 10 ثوانٍ
⏱ النهاية = 60 ثانية

━━━━━━━━━━━━━━━━━━

مثال للدقة:

📐 الدقة

ثم:

1280x720

━━━━━━━━━━━━━━━━━━

مثال للصوت:

🔊 الصوت

ثم:

1.5

أي رفع الصوت إلى 150%.

━━━━━━━━━━━━━━━━━━

مثال للسرعة:

⚡ السرعة

ثم:

1.25

━━━━━━━━━━━━━━━━━━

مثال للصيغة:

🔄 الصيغة

ثم:

mp4

أو:

mkv
webm
mov
mp3
wav
flac
ogg

━━━━━━━━━━━━━━━━━━

لإلغاء أي إدخال:

/cancel
"""

    await update.message.reply_text(
        text,
        reply_markup=main_keyboard(),
    )


# ============================================================
# CANCEL
# ============================================================

async def cancel_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    context.user_data.pop(
        "awaiting",
        None,
    )

    await update.message.reply_text(
        "❌ تم إلغاء العملية الحالية.",
        reply_markup=main_keyboard(),
    )


# ============================================================
# MEDIA RECEIVER
# ============================================================

async def handle_media(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    user = update.effective_user

    if not user:
        return

    message = update.message

    telegram_file = None
    original_name = "media"

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    if message.document:

        telegram_file = await message.document.get_file()

        original_name = (
            message.document.file_name
            or "media"
        )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    elif message.video:

        telegram_file = await message.video.get_file()

        original_name = "video.mp4"

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    elif message.audio:

        telegram_file = await message.audio.get_file()

        original_name = (
            message.audio.file_name
            or "audio"
        )

    # --------------------------------------------------------
    # VOICE
    # --------------------------------------------------------

    elif message.voice:

        telegram_file = await message.voice.get_file()

        original_name = "voice.ogg"

    else:
        return

    original_name = safe_filename(
        original_name
    )

    suffix = (
        Path(original_name).suffix
        or ".bin"
    )

    input_path = unique_path(
        INPUT_DIR,
        f"{uuid.uuid4().hex}{suffix}",
    )

    try:

        await telegram_file.download_to_drive(
            custom_path=str(input_path)
        )

        USER_FILES[user.id] = str(
            input_path
        )

        reset_user_settings(
            user.id
        )

        info = await asyncio.to_thread(
            inspect_media,
            str(input_path),
        )

        duration = info.get("duration")

        if duration is not None:
            duration_text = (
                f"{duration:.2f} ثانية"
            )
        else:
            duration_text = "غير معروف"

        video_count = len(
            info.get("video", [])
        )

        audio_count = len(
            info.get("audio", [])
        )

        text = f"""
✅ تم استلام الملف.

📄 الاسم:
{original_name}

⏱ المدة:
{duration_text}

🎥 Video Streams:
{video_count}

🎵 Audio Streams:
{audio_count}

━━━━━━━━━━━━━━━━━━

اختر الإعداد المطلوب:
"""

        await message.reply_text(
            text,
            reply_markup=main_keyboard(),
        )

    except Exception as exc:

        logger.exception(
            "Telegram file download error"
        )

        cleanup_file(input_path)

        await message.reply_text(
            f"❌ حدث خطأ أثناء قراءة الملف:\n\n{exc}"
        )


# ============================================================
# CALLBACK HANDLER
# ============================================================

async def callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    user = update.effective_user

    if not user:
        return

    user_id = user.id

    settings = get_user_settings(
        user_id
    )

    action = query.data

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if action == "reset":

        reset_user_settings(
            user_id
        )

        await query.edit_message_text(
            "🔄 تم إعادة ضبط جميع الإعدادات.",
            reply_markup=main_keyboard(),
        )

        return

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if action not in {
        "reset",
    }:

        if user_id not in USER_FILES:

            await query.message.reply_text(
                "📤 أرسل فيديو أو ملف صوتي أولًا."
            )

            return

    # --------------------------------------------------------
    # TRIM
    # --------------------------------------------------------

    if action == "trim":

        context.user_data[
            "awaiting"
        ] = "trim"

        await query.message.reply_text(
            """
✂️ قص الفيديو

أرسل:

البداية النهاية

مثال:

10 60

يعني من الثانية 10 إلى الثانية 60.
"""
        )

        return

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if action == "volume":

        context.user_data[
            "awaiting"
        ] = "volume"

        await query.message.reply_text(
            """
🔊 مستوى الصوت

أرسل قيمة مثل:

1.0 = الصوت الطبيعي
1.5 = رفع 50%
2.0 = ضعف الصوت
0.5 = خفض الصوت 50%

مثال:

1.5
"""
        )

        return

    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    if action == "speed":

        context.user_data[
            "awaiting"
        ] = "speed"

        await query.message.reply_text(
            """
⚡ سرعة الفيديو

أرسل مثل:

0.5 = نصف السرعة
1.0 = طبيعي
1.25 = أسرع 25%
1.5 = أسرع 50%
2.0 = ضعف السرعة

مثال:

1.25
"""
        )

        return

    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    if action == "resolution":

        context.user_data[
            "awaiting"
        ] = "resolution"

        await query.message.reply_text(
            """
📐 الدقة

أرسل:

العرضxالارتفاع

أمثلة:

1920x1080
1280x720
854x480
640x360
"""
        )

        return

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    if action == "fps":

        context.user_data[
            "awaiting"
        ] = "fps"

        await query.message.reply_text(
            """
🎞 معدل الإطارات FPS

أرسل مثل:

24
25
30
50
60
"""
        )

        return

    # --------------------------------------------------------
    # FORMAT
    # --------------------------------------------------------

    if action == "format":

        context.user_data[
            "awaiting"
        ] = "format"

        await query.message.reply_text(
            """
🔄 صيغة الإخراج

اختر أو اكتب:

mp4
mkv
mov
webm
mp3
wav
flac
ogg
"""
        )

        return

    # --------------------------------------------------------
    # BITRATE
    # --------------------------------------------------------

    if action == "bitrate":

        context.user_data[
            "awaiting"
        ] = "bitrate"

        await query.message.reply_text(
            """
🎚 Bitrate

أرسل:

VideoBitrate AudioBitrate

مثال:

2M 128k

أو:

4M 192k
"""
        )

        return

    # --------------------------------------------------------
    # SAMPLE RATE
    # --------------------------------------------------------

    if action == "sample":

        context.user_data[
            "awaiting"
        ] = "sample"

        await query.message.reply_text(
            """
🎛 Sample Rate

أرسل قيمة مثل:

8000
16000
22050
44100
48000
"""
        )

        return

    # --------------------------------------------------------
    # EXTRACT AUDIO
    # --------------------------------------------------------

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
            """
🎵 تم اختيار استخراج الصوت.

📦 الصيغة:
MP3

اضغط الآن:

🚀 تنفيذ المعالجة
""",
            reply_markup=main_keyboard(),
        )

        return

    # --------------------------------------------------------
    # MUTE
    # --------------------------------------------------------

    if action == "mute":

        settings[
            "remove_audio"
        ] = True

        settings[
            "extract_audio"
        ] = False

        await query.message.reply_text(
            """
🔇 تم تفعيل كتم الصوت.

الفيديو الناتج سيكون بدون Audio Stream.

اضغط:

🚀 تنفيذ المعالجة
""",
            reply_markup=main_keyboard(),
        )

        return

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    if action == "info":

        try:

            info = await asyncio.to_thread(
                inspect_media,
                USER_FILES[user_id],
            )

            text = format_media_info(
                info
            )

            await query.message.reply_text(
                text,
                reply_markup=main_keyboard(),
            )

        except Exception as exc:

            await query.message.reply_text(
                f"❌ تعذر قراءة معلومات الملف:\n{exc}"
            )

        return

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    if action == "process":

        await process_telegram_file(
            update,
            context,
        )

        return


# ============================================================
# MEDIA INFO FORMATTER
# ============================================================

def format_media_info(info):

    lines = []

    lines.append("ℹ️ معلومات الملف")
    lines.append("")
    lines.append(
        f"📄 {info.get('filename')}"
    )

    if info.get("format"):
        lines.append(
            f"📦 Format: {info['format']}"
        )

    if info.get("duration") is not None:
        lines.append(
            f"⏱ Duration: {info['duration']:.2f}s"
        )

    if info.get("bitrate"):
        lines.append(
            f"🎚 Bitrate: {info['bitrate']} bps"
        )

    if info.get("size"):
        size_mb = (
            info["size"]
            / 1024
            / 1024
        )

        lines.append(
            f"💾 Size: {size_mb:.2f} MB"
        )

    lines.append("")

    for video in info.get(
        "video",
        [],
    ):

        lines.append("🎥 Video")

        lines.append(
            f"Codec: {video.get('codec')}"
        )

        lines.append(
            f"Resolution: "
            f"{video.get('width')}x"
            f"{video.get('height')}"
        )

        if video.get("fps"):
            lines.append(
                f"FPS: {video['fps']:.2f}"
            )

        lines.append(
            f"Pixel Format: "
            f"{video.get('pix_fmt')}"
        )

        lines.append("")

    for audio in info.get(
        "audio",
        [],
    ):

        lines.append("🎵 Audio")

        lines.append(
            f"Codec: {audio.get('codec')}"
        )

        lines.append(
            f"Sample Rate: "
            f"{audio.get('sample_rate')} Hz"
        )

        lines.append(
            f"Channels: "
            f"{audio.get('channels')}"
        )

        lines.append(
            f"Layout: "
            f"{audio.get('layout')}"
        )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# TEXT SETTINGS HANDLER
# ============================================================

async def text_settings_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    user = update.effective_user

    if not user:
        return

    awaiting = context.user_data.get(
        "awaiting"
    )

    if not awaiting:
        return

    settings = get_user_settings(
        user.id
    )

    text = (
        update.message.text
        or ""
    ).strip()

    try:

        # ----------------------------------------------------
        # TRIM
        # ----------------------------------------------------

        if awaiting == "trim":

            parts = text.replace(
                ",",
                " ",
            ).split()

            if len(parts) != 2:
                raise ValueError(
                    "اكتب البداية والنهاية مثل:\n10 60"
                )

            start = float(parts[0])
            end = float(parts[1])

            if start < 0:
                raise ValueError(
                    "البداية لا يمكن أن تكون سالبة."
                )

            if end <= start:
                raise ValueError(
                    "النهاية يجب أن تكون أكبر من البداية."
                )

            settings[
                "trim_start"
            ] = start

            settings[
                "trim_end"
            ] = end

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        elif awaiting == "volume":

            value = parse_float(
                text,
                minimum=0,
                maximum=10,
            )

            settings[
                "volume"
            ] = value

        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        elif awaiting == "speed":

            value = parse_float(
                text,
                minimum=0.1,
                maximum=4.0,
            )

            settings[
                "speed"
            ] = value

        # ----------------------------------------------------
        # RESOLUTION
        # ----------------------------------------------------

        elif awaiting == "resolution":

            normalized = (
                text.lower()
                .replace("×", "x")
            )

            parts = normalized.split("x")

            if len(parts) != 2:
                raise ValueError(
                    "استخدم الشكل:\n1280x720"
                )

            width = int(parts[0])
            height = int(parts[1])

            if width <= 0 or height <= 0:
                raise ValueError(
                    "الدقة يجب أن تكون موجبة."
                )

            settings[
                "width"
            ] = width

            settings[
                "height"
            ] = height

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elif awaiting == "fps":

            value = parse_float(
                text,
                minimum=1,
                maximum=240,
            )

            settings[
                "fps"
            ] = value

        # ----------------------------------------------------
        # FORMAT
        # ----------------------------------------------------

        elif awaiting == "format":

            value = (
                text.lower()
                .replace(".", "")
                .strip()
            )

            allowed = {
                "mp4",
                "mkv",
                "mov",
                "webm",
                "mp3",
                "wav",
                "flac",
                "ogg",
            }

            if value not in allowed:
                raise ValueError(
                    "الصيغة غير مدعومة."
                )

            settings[
                "output_format"
            ] = value

        # ----------------------------------------------------
        # BITRATE
        # ----------------------------------------------------

        elif awaiting == "bitrate":

            parts = text.split()

            if len(parts) == 1:

                value = parts[0]

                settings[
                    "video_bitrate"
                ] = value

                settings[
                    "audio_bitrate"
                ] = "128k"

            elif len(parts) == 2:

                settings[
                    "video_bitrate"
                ] = parts[0]

                settings[
                    "audio_bitrate"
                ] = parts[1]

            else:

                raise ValueError(
                    "مثال:\n2M 128k"
                )

            parse_bitrate(
                settings[
                    "video_bitrate"
                ]
            )

            parse_bitrate(
                settings[
                    "audio_bitrate"
                ]
            )

        # ----------------------------------------------------
        # SAMPLE RATE
        # ----------------------------------------------------

        elif awaiting == "sample":

            value = parse_int(
                text,
                minimum=8000,
                maximum=192000,
            )

            settings[
                "sample_rate"
            ] = value

        context.user_data.pop(
            "awaiting",
            None,
        )

        await update.message.reply_text(
            "✅ تم حفظ الإعداد.\n\n"
            "يمكنك اختيار إعداد آخر أو الضغط على:\n"
            "🚀 تنفيذ المعالجة",
            reply_markup=main_keyboard(),
        )

    except Exception as exc:

        await update.message.reply_text(
            f"❌ قيمة غير صحيحة.\n\n{exc}\n\n"
            "حاول مرة أخرى."
        )


# ============================================================
# TELEGRAM PROCESSING
# ============================================================

async def process_telegram_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    user = update.effective_user

    if not user:
        return

    user_id = user.id

    input_path = USER_FILES.get(
        user_id
    )

    if not input_path:

        await update.effective_message.reply_text(
            "📤 أرسل ملفًا أولًا."
        )

        return

    if not Path(input_path).exists():

        USER_FILES.pop(
            user_id,
            None,
        )

        await update.effective_message.reply_text(
            "❌ الملف لم يعد موجودًا على السيرفر.\n"
            "📤 أرسله مرة أخرى."
        )

        return

    settings = get_user_settings(
        user_id
    ).copy()

    # --------------------------------------------------------
    # DETERMINE FORMAT
    # --------------------------------------------------------

    output_format = (
        settings.get(
            "output_format"
        )
        or Path(input_path)
        .suffix
        .lstrip(".")
        .lower()
        or "mp4"
    )

    if settings.get(
        "extract_audio"
    ):

        output_format = "mp3"

    settings[
        "output_format"
    ] = output_format

    input_name = safe_filename(
        Path(input_path).name
    )

    output_name = (
        Path(input_name).stem
        + "_processed."
        + output_format
    )

    output_path = unique_path(
        OUTPUT_DIR,
        output_name,
    )

    status_message = (
        await update.effective_message.reply_text(
            """
⏳ جاري معالجة الملف...

⚙️ المحرك: PyAV
🚫 FFmpeg executable: غير مستخدم

قد تستغرق العملية بعض الوقت حسب حجم الملف.
"""
        )
    )

    try:

        await asyncio.to_thread(
            process_media_pyav,
            input_path,
            str(output_path),
            settings,
        )

        # ----------------------------------------------------
        # SEND FILE
        # ----------------------------------------------------

        await update.effective_message.reply_document(
            document=str(output_path),
            filename=output_name,
            caption=(
                "✅ تمت المعالجة بنجاح!\n\n"
                f"📦 الصيغة: {output_format}\n"
                "⚙️ Engine: PyAV"
            ),
        )

        try:
            await status_message.delete()
        except Exception:
            pass

        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        cleanup_file(
            input_path
        )

        cleanup_file(
            output_path
        )

        USER_FILES.pop(
            user_id,
            None,
        )

        reset_user_settings(
            user_id
        )

        context.user_data.pop(
            "awaiting",
            None,
        )

    except Exception as exc:

        logger.exception(
            "Telegram processing failed"
        )

        cleanup_file(
            output_path
        )

        await update.effective_message.reply_text(
            "❌ حدث خطأ أثناء معالجة الملف.\n\n"
            f"التفاصيل:\n{exc}\n\n"
            "📤 يمكنك إرسال الملف مرة أخرى والمحاولة."
        )


# ============================================================
# UNKNOWN TEXT
# ============================================================

async def unknown_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if context.user_data.get(
        "awaiting"
    ):
        return

    await update.message.reply_text(
        """
🤖 لم أفهم الأمر.

📤 أرسل فيديو أو ملف صوتي،
أو استخدم:

/start
/help
/cancel
""",
        reply_markup=main_keyboard(),
    )


# ============================================================
# REGISTER BOT HANDLERS
# ============================================================

def register_bot_handlers(
    application: Application,
):

    application.add_handler(
        CommandHandler(
            "start",
            start_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "cancel",
            cancel_command,
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            callback_handler
        )
    )

    # --------------------------------------------------------
    # MEDIA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TEXT SETTINGS
    # --------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_settings_handler,
        )
    )


# ============================================================
# UVICORN ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )
