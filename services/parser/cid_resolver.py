from models import ImageInfo

def resolve_images(html_images: list[ImageInfo],
                   mime_images: list[ImageInfo]) -> tuple[list[ImageInfo], list[str]]:
    """HTML даёт позицию и alt, MIME даёт байты и хеш. Склеиваем по cid."""
    warnings = []

    by_cid: dict[str, ImageInfo] = {}
    for mi in mime_images:
        if mi.cid:
            by_cid[mi.cid] = mi
            by_cid.setdefault(mi.cid.lower(), mi)   # битые генераторы путают регистр

    merged: list[ImageInfo] = []
    matched_ids: set[int] = set()

    for hi in html_images:
        if hi.cid:
            mi = by_cid.get(hi.cid) or by_cid.get(hi.cid.lower())
            if mi is not None:
                matched_ids.add(id(mi))
                merged.append(ImageInfo(
                    cid=hi.cid,
                    content_type=mi.content_type,
                    size_bytes=mi.size_bytes,
                    sha256=mi.sha256,
                    content_b64=mi.content_b64,
                    context=hi.context or mi.context,
                    source="html+mime",
                ))
                continue
            warnings.append(f"cid not resolved: {hi.cid}")   # html ссылается, части нет
            merged.append(hi)
        else:
            merged.append(hi)                                 # внешняя картинка по URL

    # MIME-картинки, на которые HTML не ссылается — тоже отдаём!
    for mi in mime_images:
        if id(mi) not in matched_ids:
            merged.append(mi)
            warnings.append(f"inline image without HTML reference (cid={mi.cid})")

    return merged, warnings