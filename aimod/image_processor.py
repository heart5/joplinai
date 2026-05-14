"""Image resource fetching and preprocessing for Joplin notes.

Fetches image resources from Joplin API, converts to base64
for use with multimodal LLM APIs (DeepSeek V4 vision).
"""
import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pathmagic

with pathmagic.Context():
    try:
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")

__all__ = ["ImageProcessor"]

DEEPSEEK_SUPPORTED_MIMES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE = 8 * 1024 * 1024  # 8MB
MAX_IMAGE_WORKERS = 5


class ImageProcessor:
    """Fetch and encode image resources from Joplin API.

    Returns pure base64 strings (no data URI prefix) with MIME type,
    matching DeepSeek V4 API requirements.
    """

    def __init__(self, joplin_api):
        self.api = joplin_api

    def fetch_image_base64(self, resource_id: str) -> Optional[dict]:
        """Fetch a single image resource, return {'b64': ..., 'mime': ...}.

        Returns None if the resource is not a supported image or fetch fails.
        """
        try:
            content = self.api.get_resource_file(resource_id)
            if not content:
                log.debug(f"资源 {resource_id} 内容为空")
                return None
            mime = _guess_mime_from_bytes(content)
            if not mime:
                log.debug(f"资源 {resource_id} 无法识别图片格式，跳过")
                return None
            if mime not in DEEPSEEK_SUPPORTED_MIMES:
                log.debug(
                    f"资源 {resource_id} 格式 {mime} 不被DeepSeek支持，跳过"
                )
                return None
            if len(content) > MAX_IMAGE_SIZE:
                log.warning(
                    f"资源 {resource_id} 图片 {len(content)} 字节超过8MB限制，跳过"
                )
                return None
            b64 = base64.b64encode(content).decode("ascii")
            return {"b64": b64, "mime": mime}
        except Exception as e:
            log.warning(f"获取资源 {resource_id} 失败: {e}")
            return None

    def fetch_images_for_note(
        self, note_id: str, resource_ids: list[str]
    ) -> dict[str, dict]:
        """Fetch multiple image resources concurrently.

        Returns {resource_id: {'b64': str, 'mime': str}} dict.
        """
        if not resource_ids:
            return {}
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(MAX_IMAGE_WORKERS, len(resource_ids))) as ex:
            future_to_rid = {
                ex.submit(self.fetch_image_base64, rid): rid for rid in resource_ids
            }
            for future in as_completed(future_to_rid):
                rid = future_to_rid[future]
                try:
                    img_data = future.result()
                    if img_data:
                        results[rid] = img_data
                except Exception as e:
                    log.warning(f"并发获取资源 {rid} 异常: {e}")
        return results


def _guess_mime_from_bytes(data: bytes) -> str:
    """Guess image MIME type from magic bytes."""
    if len(data) < 4:
        return ""
    if data[:4] == b'\xff\xd8\xff':
        return "image/jpeg"
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    if data[:4] in (b'RIFF') and data[8:12] == b'WEBP':
        return "image/webp"
    if data[:2] == b'BM':
        return "image/bmp"
    return ""
