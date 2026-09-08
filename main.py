import os
import uuid
import asyncio
import threading
import shutil
from pathlib import Path
from typing import Optional

import av
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
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

PORT = int(os.getenv("PORT", "8000"))

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
# FASTAPI
# ============================================================

app = FastAPI(
    title="PyAV Professional Media Studio",
    version="2.0.0",
    description=(
        "Professional audio/video processing "
        "using PyAV without calling the FFmpeg executable."
    )
)


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

        "fade_in": None,
        "fade_out": None,

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
        uuid.uuid4().hex + "_" + filename
    )


def cleanup_file(path):

    try:
        if path:
            Path(path).unlink(
                missing_ok=True
            )
    except Exception:
        pass


# ============================================================
# PYAV MEDIA INFORMATION
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

        container = av.open(path)

        if container.format:
            result["format"] = (
                container.format.name
            )

        if container.duration is not None:
            result["duration"] = (
                float(container.duration)
                / av.time_base
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

            if stream.type == "audio":

                item.update({
                    "sample_rate": stream.rate,
                    "channels": stream.channels,
                    "layout": (
                        str(stream.layout)
                        if stream.layout
                        else None
                    ),
                })

            result["streams"].append(item)

    finally:

        if container:
            container.close()

    return result


# ============================================================
# CODEC HELPERS
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


def choose_video_codec(format_name):

    format_name = (
        format_name or ""
    ).lower()

    if format_name in {
        "mp4",
        "mov",
        "m4v",
    }:
        return "h264"

    if format_name in {
        "webm",
    }:
        return "vp8"

    if format_name in {
        "mkv",
        "matroska",
    }:
        return "h264"

    return "h264"


def choose_audio_codec(format_name):

    format_name = (
        format_name or ""
    ).lower()

    if format_name == "mp3":
        return "mp3"

    if format_name in {
        "ogg",
        "opus",
    }:
        return "opus"

    if format_name == "flac":
        return "flac"

    if format_name == "wav":
        return "pcm_s16le"

    if format_name in {
        "m4a",
        "mp4",
        "mov",
    }:
        return "aac"

    return "aac"


# ============================================================
# VIDEO RESIZE
# ============================================================

def resize_video_frame(
    frame,
    width,
    height,
):

    if not width or not height:
        return frame

    return frame.reformat(
        width=width,
        height=height,
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

    if volume == 1.0:
        return frame

    try:

        array = frame.to_ndarray()

        array = array * float(volume)

        # Prevent integer overflow.
        if array.dtype.kind in "iu":

            info = __import__(
                "numpy"
            ).iinfo(array.dtype)

            array = array.clip(
                info.min,
                info.max,
            )

        else:

            array = array.clip(
                -1.0,
                1.0,
            )

        new_frame = AudioFrame.from_ndarray(
            array,
            layout=frame.layout.name,
        )

        new_frame.sample_rate = (
            frame.sample_rate
        )

        return new_frame

    except Exception:

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
            settings.get("output_format")
            or input_format
        )

        output_format = (
            output_format
            .lower()
            .replace(".", "")
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

        # ----------------------------------------------------
        # VIDEO STREAM
        # ----------------------------------------------------

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

            except Exception:

                # Fallback to source codec
                output_video = (
                    output_container.add_stream(
                        video_stream.codec_context.name
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

            output_video.width = width
            output_video.height = height

            if settings.get("fps"):

                output_video.average_rate = (
                    settings["fps"]
                )

            elif video_stream.average_rate:

                output_video.average_rate = (
                    video_stream.average_rate
                )

            if settings.get(
                "video_bitrate"
            ):

                output_video.bit_rate = (
                    parse_bitrate(
                        settings[
                            "video_bitrate"
                        ]
                    )
                )

        # ----------------------------------------------------
        # AUDIO STREAM
        # ----------------------------------------------------

        if (
            audio_stream
            and not settings.get(
                "remove_audio",
                False
            )
            and not settings.get(
                "extract_audio",
                False
            )
        ):

            codec = choose_audio_codec(
                output_format
            )

            try:

                output_audio = (
                    output_container.add_stream(
                        codec,
                        rate=(
                            settings.get(
                                "sample_rate"
                            )
                            or audio_stream.rate
                        ),
                    )
                )

            except Exception:

                output_audio = (
                    output_container.add_stream(
                        audio_stream.codec_context.name,
                        rate=audio_stream.rate,
                    )
                )

            if settings.get(
                "audio_bitrate"
            ):

                output_audio.bit_rate = (
                    parse_bitrate(
                        settings[
                            "audio_bitrate"
                        ]
                    )
                )

        # ----------------------------------------------------
        # EXTRACT AUDIO
        # ----------------------------------------------------

        if (
            settings.get(
                "extract_audio",
                False
            )
            and audio_stream
        ):

            codec = choose_audio_codec(
                output_format
            )

            output_audio = (
                output_container.add_stream(
                    codec,
                    rate=(
                        settings.get(
                            "sample_rate"
                        )
                        or audio_stream.rate
                    ),
                )
            )

        # ----------------------------------------------------
        # PACKETS / FRAMES
        # ----------------------------------------------------

        streams_to_decode = []

        if video_stream:
            streams_to_decode.append(
                video_stream
            )

        if audio_stream:
            streams_to_decode.append(
                audio_stream
            )

        for frame in input_container.decode(
            *streams_to_decode
        ):

            # ----------------------------------------------
            # VIDEO
            # ----------------------------------------------

            if isinstance(
                frame,
                VideoFrame
            ):

                pts_seconds = None

                if frame.pts is not None:
                    try:

                        pts_seconds = float(
                            frame.pts
                            * frame.time_base
                        )

                    except Exception:
                        pass

                start = settings.get(
                    "start"
                )

                end = settings.get(
                    "end"
                )

                if (
                    start is not None
                    and pts_seconds is not None
                    and pts_seconds < start
                ):
                    continue

                if (
                    end is not None
                    and pts_seconds is not None
                    and pts_seconds > end
                ):
                    continue

                if output_video:

                    frame = resize_video_frame(
                        frame,
                        settings.get("width"),
                        settings.get("height"),
                    )

                    frame = frame.reformat(
                        format="yuv420p"
                    )

                    for packet in (
                        output_video.encode(
                            frame
                        )
                    ):

                        output_container.mux(
                            packet
                        )

            # ----------------------------------------------
            # AUDIO
            # ----------------------------------------------

            elif isinstance(
                frame,
                AudioFrame
            ):

                pts_seconds = None

                if frame.pts is not None:

                    try:

                        pts_seconds = float(
                            frame.pts
                            * frame.time_base
                        )

                    except Exception:
                        pass

                start = settings.get(
                    "start"
                )

                end = settings.get(
                    "end"
                )

                if (
                    start is not None
                    and pts_seconds is not None
                    and pts_seconds < start
                ):
                    continue

                if (
                    end is not None
                    and pts_seconds is not None
                    and pts_seconds > end
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

                    for packet in (
                        output_audio.encode(
                            frame
                        )
                    ):

                        output_container.mux(
                            packet
                        )

        # ----------------------------------------------------
        # FLUSH ENCODERS
        # ----------------------------------------------------

        if output_video:

            for packet in (
                output_video.encode()
            ):

                output_container.mux(
                    packet
                )

        if output_audio:

            for packet in (
                output_audio.encode()
            ):

                output_container.mux(
                    packet
                )

        # ----------------------------------------------------
        # CLOSE
        # ----------------------------------------------------

        output_container.close()
        output_container = None

        return str(output_path)

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
# API ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "engine": "PyAV",
        "external_ffmpeg": False,
        "telegram_bot": bool(BOT_TOKEN),
        "processing_limits": "none_in_application",
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "pyav": True,
        "external_ffmpeg": False,
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

                output.write(chunk)

        return await asyncio.to_thread(
            inspect_media,
            str(path)
        )

    finally:

        cleanup_file(path)


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

    width: Optional[int] = None,
    height: Optional[int] = None,

    fps: Optional[float] = None,

    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,

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

                output.write(chunk)

        extension = (
            output_format
            or input_path.suffix.lstrip(".")
        )

        if extract_audio:

            extension = (
                output_format
                or "mp3"
            )

        extension = extension.replace(
            ".",
            ""
        )

        output_name = (
            f"{input_path.stem}_processed."
            f"{extension}"
        )

        output_path = (
            OUTPUT_DIR / output_name
        )

        settings = default_settings()

        settings.update({
            "output_format": extension,
            "start": start,
            "end": end,
            "volume": volume,
            "width": width,
            "height": height,
            "fps": fps,
            "sample_rate": sample_rate,
            "channels": channels,
            "audio_bitrate": audio_bitrate,
            "video_bitrate": video_bitrate,
            "remove_audio": remove_audio,
            "extract_audio": extract_audio,
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
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        cleanup_file(input_path)


# ============================================================
# TELEGRAM KEYBOARD
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
                "🎚 Bitrate",
                callback_data="bitrate"
            ),
            InlineKeyboardButton(
                "🎛 Sample Rate",
                callback_data="sample"
            ),
        ],

        [
            InlineKeyboardButton(
                "ℹ️ معلومات",
                callback_data="info"
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
    ])


# ============================================================
# TELEGRAM START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    text = """
🎬 *PyAV Professional Media Studio*

مرحبًا بك 👋

أرسل فيديو أو ملفًا صوتيًا، ثم اختر أدوات المعالجة.

🎥 الفيديو:
• قص
• تغيير الدقة
• تغيير FPS
• تحويل الصيغة
• تغيير Bitrate
• إزالة الصوت

🎵 الصوت:
• رفع/خفض الصوت
• استخراج الصوت
• تغيير Sample Rate
• تغيير Bitrate

⚙️ المعالجة تتم مباشرة على السيرفر باستخدام PyAV.

📥 بعد الانتهاء يصلك الملف النهائي مباشرة.

استخدم /help لشرح الاستخدام.
"""

    await update.message.reply_text(
        text,
        parse_mode="Markdown",
        reply_markup=main_keyboard(),
    )


# ============================================================
# TELEGRAM HELP
# ============================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    text = """
📖 *طريقة الاستخدام*

1️⃣ أرسل فيديو أو صوت.

2️⃣ اختر الأداة.

3️⃣ أدخل القيمة المطلوبة.

4️⃣ اضغط 🚀 تنفيذ.

5️⃣ انتظر انتهاء المعالجة.

6️⃣ سيُرسل لك الملف الناتج.

💡 يمكنك رفع ملف جديد في أي وقت.

⚠️ لا يضع التطبيق حدًا اصطناعيًا لمدة الملف أو عدد العمليات.
لكن حدود Telegram والسيرفر نفسه تظل قائمة.
"""

    await update.message.reply_text(
        text,
        parse_mode="Markdown",
        reply_markup=main_keyboard(),
    )


# ============================================================
# TELEGRAM MEDIA UPLOAD
# ============================================================

async def handle_media(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

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

    await telegram_file.download_to_drive(
        custom_path=str(local_path)
    )

    USER_FILES[user_id] = str(
        local_path
    )

    USER_SETTINGS[user_id] = (
        default_settings()
    )

    await message.reply_text(
        "✅ تم رفع الملف بنجاح.\n\n"
        "اختر ما تريد فعله:",
        reply_markup=main_keyboard(),
    )


# ============================================================
# TELEGRAM CALLBACKS
# ============================================================

async def callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    user_id = (
        query.from_user.id
    )

    action = query.data

    settings = get_settings(
        user_id
    )

    if action == "reset":

        USER_SETTINGS[user_id] = (
            default_settings()
        )

        await query.message.reply_text(
            "🔄 تم إعادة ضبط الإعدادات.",
            reply_markup=main_keyboard(),
        )

        return

    if action != "info" and action != "process":

        if user_id not in USER_FILES:

            await query.message.reply_text(
                "⚠️ أرسل ملفًا أولًا."
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
            "✂️ أرسل البداية والنهاية بالثواني:\n\n"
            "مثال:\n"
            "`10 60`",
            parse_mode="Markdown",
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
            "🔊 أرسل معامل الصوت:\n\n"
            "`1.5` زيادة 50%\n"
            "`0.5` خفض للنصف",
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
            "⚡ أرسل سرعة الصوت.\n\n"
            "`1.5` أسرع\n"
            "`0.75` أبطأ",
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
            "📐 أرسل العرض والارتفاع:\n\n"
            "`1920 1080`",
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
            "🎞 أرسل FPS:\n\n"
            "`30`\n"
            "`60`",
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
            "🔄 أرسل الصيغة:\n\n"
            "`mp4`\n"
            "`mkv`\n"
            "`webm`\n"
            "`mp3`\n"
            "`wav`",
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
            "🎚 أرسل Bitrate.\n\n"
            "مثال:\n"
            "`128k`\n"
            "`192k`\n"
            "`2M`",
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
            "🎛 أرسل Sample Rate.\n\n"
            "مثال:\n"
            "`44100`\n"
            "`48000`",
        )

        return

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    if action == "extract":

        settings[
            "extract_audio"
        ] = True

        settings[
            "output_format"
        ] = "mp3"

        await query.message.reply_text(
            "🎵 تم اختيار استخراج الصوت إلى MP3.",
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

        await query.message.reply_text(
            "🔇 تم اختيار إزالة الصوت.",
            reply_markup=main_keyboard(),
        )

        return

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    if action == "info":

        if user_id not in USER_FILES:

            await query.message.reply_text(
                "⚠️ لا يوجد ملف."
            )

            return

        try:

            info = await asyncio.to_thread(
                inspect_media,
                USER_FILES[user_id],
            )

            text = (
                "ℹ️ *معلومات الملف*\n\n"
                f"📄 `{info['filename']}`\n"
                f"📦 Format: `{info['format']}`\n"
                f"⏱ Duration: "
                f"`{info['duration']}` sec\n"
                f"💾 Bitrate: "
                f"`{info['bitrate']}`\n\n"
            )

            for stream in info[
                "streams"
            ]:

                text += (
                    f"🎛 `{stream}`\n"
                )

            await query.message.reply_text(
                text,
                parse_mode="Markdown",
            )

        except Exception as e:

            await query.message.reply_text(
                f"❌ خطأ:\n{e}"
            )

        return

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    if action == "process":

        await process_telegram_file(
            query,
            user_id,
        )

        return


# ============================================================
# TELEGRAM TEXT SETTINGS
# ============================================================

async def text_settings_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    awaiting = context.user_data.get(
        "awaiting"
    )

    if not awaiting:
        return

    user_id = (
        update.message.from_user.id
    )

    settings = get_settings(
        user_id
    )

    text = update.message.text.strip()

    try:

        if awaiting == "trim":

            values = text.split()

            if len(values) < 2:
                raise ValueError()

            settings["start"] = float(
                values[0]
            )

            settings["end"] = float(
                values[1]
            )

        elif awaiting == "volume":

            settings["volume"] = float(
                text
            )

        elif awaiting == "speed":

            settings["speed"] = float(
                text
            )

        elif awaiting == "resolution":

            values = text.split()

            settings["width"] = int(
                values[0]
            )

            settings["height"] = int(
                values[1]
            )

        elif awaiting == "fps":

            settings["fps"] = float(
                text
            )

        elif awaiting == "format":

            settings[
                "output_format"
            ] = text.lower().replace(
                ".",
                ""
            )

        elif awaiting == "bitrate":

            settings[
                "audio_bitrate"
            ] = text

            settings[
                "video_bitrate"
            ] = text

        elif awaiting == "sample":

            settings[
                "sample_rate"
            ] = int(text)

        context.user_data.pop(
            "awaiting",
            None,
        )

        await update.message.reply_text(
            "✅ تم حفظ الإعداد.",
            reply_markup=main_keyboard(),
        )

    except Exception:

        await update.message.reply_text(
            "❌ القيمة غير صحيحة.\n"
            "أعد إرسالها بالصيغة المطلوبة."
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
            "⚠️ أرسل ملفًا أولًا."
        )

        return

    input_path = Path(
        USER_FILES[user_id]
    )

    settings = get_settings(
        user_id
    )

    await query.message.reply_text(
        "⏳ بدأت معالجة الملف...\n\n"
        "🧠 محرك المعالجة: PyAV\n"
        "🚫 لا يتم استدعاء FFmpeg executable\n\n"
        "انتظر حتى اكتمال العملية."
    )

    extension = (
        settings.get(
            "output_format"
        )
        or input_path.suffix.lstrip(".")
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

        await query.message.reply_text(
            "✅ اكتملت المعالجة.\n"
            "📤 جاري إرسال الملف..."
        )

        with open(
            output_path,
            "rb"
        ) as file:

            await query.message.reply_document(
                document=file,
                filename=output_path.name,
                caption=(
                    "🎬 تم إنشاء الملف بنجاح."
                ),
            )

    except Exception as e:

        await query.message.reply_text(
            "❌ حدث خطأ أثناء المعالجة:\n\n"
            f"{str(e)[:4000]}"
        )

    finally:

        cleanup_file(
            output_path
        )


# ============================================================
# TELEGRAM BOT
# ============================================================

def run_bot():

    if not BOT_TOKEN:

        print(
            "BOT_TOKEN not configured."
        )

        return

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

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

    print(
        "Telegram bot starting..."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event(
    "startup"
)
async def startup():

    if BOT_TOKEN:

        thread = threading.Thread(
            target=run_bot,
            daemon=True,
        )

        thread.start()

        print(
            "Telegram bot started."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
    )
