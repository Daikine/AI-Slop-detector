import base64
import binascii
import logging
import re
import time

from fastapi import FastAPI, HTTPException

from mime_walker import parse_email, EmailTooLarge
from models import ParseRequest, ParsedEmail

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parser")

app = FastAPI(title="parser-service", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "parser"}


@app.post("/parse", response_model=ParsedEmail)
def parse(req: ParseRequest):
    # validate=True после вычищения пробелов: переводы строк в base64
    # легальны, а вот мусорные символы должны дать 400
    cleaned = re.sub(r"\s+", "", req.raw_email)
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="raw_email is not valid base64")

    if not raw.strip():
        raise HTTPException(status_code=400, detail="empty payload")

    started = time.perf_counter()
    try:
        result = parse_email(
            raw,
            include_inline_content=req.include_inline_content,
            max_inline_bytes=req.max_inline_image_bytes,
        )
    except EmailTooLarge as e:
        raise HTTPException(status_code=413, detail=str(e))

    result.parse_metadata.parse_time_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info("parsed: images=%d links=%d attachments=%d warnings=%d (%.1fms)",
                len(result.images), len(result.links), len(result.attachments),
                len(result.parse_metadata.warnings), result.parse_metadata.parse_time_ms)
    return result