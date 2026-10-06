from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class AuthResults(BaseModel):
    spf: Optional[str] = None      # pass | fail | softfail | neutral | none | ...
    dkim: Optional[str] = None
    dmarc: Optional[str] = None


class HeaderInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: Optional[str] = Field(None, alias="from")   # "from" — зарезервировано в Python
    from_domain: Optional[str] = None                   # домен отправителя — фича для ML
    subject: Optional[str] = None
    reply_to: Optional[str] = None
    return_path: Optional[str] = None
    x_mailer: Optional[str] = None
    date: Optional[str] = None
    message_id: Optional[str] = None
    list_unsubscribe: Optional[str] = None
    authentication: Optional[AuthResults] = None


class ImageInfo(BaseModel):
    url: Optional[str] = None            # внешняя картинка (для image-analyzer)
    cid: Optional[str] = None            # инлайновая (Content-ID)
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None         # дедупликация между письмами
    content_b64: Optional[str] = None    # только если include_inline_content=True
    context: Optional[str] = None        # alt-текст
    source: str = "html"                 # "html" | "mime" | "html+mime"


class AttachmentInfo(BaseModel):
    filename: Optional[str] = None
    mime_type: str
    size: int
    sha256: Optional[str] = None
    inline: bool = False


class LinkInfo(BaseModel):
    url: str
    context: Optional[str] = None        # анкорный текст — фича для link-analyzer
    position: int                        # позиция в документе
    scheme: Optional[str] = None         # http | https | mailto | javascript | cid
    is_cid: bool = False                 # такие ссылки link-analyzer пропускает


class ParseMetadata(BaseModel):
    text_source: Optional[str] = None    # "mime_plain" | "derived_from_html"
    multipart_type: Optional[str] = None
    nesting_depth: int = 0
    parse_time_ms: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


class ParseRequest(BaseModel):
    raw_email: str                       # base64(raw MIME)
    include_inline_content: bool = False # класть ли байты инлайн-картинок в ответ
    max_inline_image_bytes: int = 2_000_000


class ParsedEmail(BaseModel):
    headers: HeaderInfo
    html: Optional[str] = None
    text: Optional[str] = None
    images: List[ImageInfo] = Field(default_factory=list)
    attachments: List[AttachmentInfo] = Field(default_factory=list)
    links: List[LinkInfo] = Field(default_factory=list)
    parse_metadata: ParseMetadata = Field(default_factory=ParseMetadata)