"""Text preprocessing utilities for embedding generation.

Extracted from EmbeddingGenerator to keep the main class focused on embedding logic.
"""
import re
from typing import Optional

__all__ = ["TextPreprocessor"]


class TextPreprocessor:
    """Text cleaning, normalization, and validation for embedding input."""

    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size

    RESOURCE_ID_PATTERN = re.compile(r"!?\[.*?\]\(:/([a-fA-F0-9]{32})\)")
    IMAGE_SYNTAX_PATTERN = re.compile(r"!?\[(.*?)\]\(:/([a-fA-F0-9]{32})\)")

    @staticmethod
    def extract_resource_ids(text: str) -> list[str]:
        """Extract Joplin resource IDs from image syntax in text.

        Matches ![alt_text](:/32-char-hex-resource-id) and returns
        the ordered list of resource IDs as they appear in the text.
        """
        if not text:
            return []
        return TextPreprocessor.RESOURCE_ID_PATTERN.findall(text)

    @staticmethod
    def remove_image_syntax(text: str, keep_alt: bool = False) -> str:
        """Remove Joplin image syntax ![alt](:/resource_id) from text.

        Args:
            text: The raw text with image references.
            keep_alt: If True, replace each image reference with its alt_text
                      (e.g. "工作日志"). If False, remove them entirely.
        """
        if not text:
            return ""
        if keep_alt:
            return TextPreprocessor.IMAGE_SYNTAX_PATTERN.sub(r"\1", text)
        return TextPreprocessor.IMAGE_SYNTAX_PATTERN.sub("", text)

    @staticmethod
    def replace_images_with_descriptions(text: str, descriptions: dict[str, str]) -> str:
        """将 ![alt](:/rid) / [text](:/rid) 替换为 【图片：{desc}】"""
        if not text or not descriptions:
            return text

        def _replacer(match):
            rid = match.group(2)
            desc = descriptions.get(rid, "")
            if desc:
                return f"【图片：{desc}】"
            alt = match.group(1).strip()
            return alt if alt else ""

        return TextPreprocessor.IMAGE_SYNTAX_PATTERN.sub(_replacer, text)

    @staticmethod
    def _looks_like_table_header(line: str) -> bool:
        """表头/数据行：以|开头结尾，内部至少含一个|"""
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            return False
        inner = line[1:-1]
        return "|" in inner

    @staticmethod
    def _looks_like_table_separator(line: str) -> bool:
        """分隔行：以|开头结尾，内部仅含| - : 空格"""
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            return False
        inner = line[1:-1]
        return bool(re.match(r"^[\|\-:\s]+$", inner))

    @staticmethod
    def _looks_like_table_data_row(line: str) -> bool:
        """数据行：同表头检测"""
        return TextPreprocessor._looks_like_table_header(line)

    @staticmethod
    def _split_table_row(row: str) -> list[str]:
        """切分管道分隔的行为单元格列表"""
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [cell.strip() for cell in row.split("|")]

    @staticmethod
    def _normalize_cell(cells: list[str]) -> list[str]:
        """规范列名，空列名补充为列1/列2..."""
        result = []
        for i, c in enumerate(cells):
            c = c.strip()
            if not c:
                c = f"列{i + 1}"
            result.append(c)
        return result

    @staticmethod
    def _truncate_cell(value: str, max_len: int = 50) -> str:
        """截断过长的单元格内容"""
        value = value.strip()
        if len(value) > max_len:
            return value[:max_len] + "…"
        return value

    @staticmethod
    def _convert_table_block(table_lines: list[str]) -> str:
        """将解析的表格块转换为自然语言描述

        【表格：包含N列（a、b、c），共M行数据。第1行：a为v1，b为v2。...】
        """
        if len(table_lines) < 2:
            return ""

        header_cells = TextPreprocessor._split_table_row(table_lines[0])
        num_cols = len(header_cells)
        col_names = TextPreprocessor._normalize_cell(header_cells)

        data_rows = []
        for line in table_lines[2:]:
            cells = TextPreprocessor._split_table_row(line)
            if len(cells) < num_cols:
                cells += [""] * (num_cols - len(cells))
            data_rows.append(cells[:num_cols])

        num_rows = len(data_rows)

        max_cols = 8
        max_rows = 20

        parts = []
        if num_cols <= max_cols:
            col_desc = "、".join(col_names)
            parts.append(f"包含{num_cols}列（{col_desc}）")
        else:
            col_desc = "、".join(col_names[:max_cols])
            parts.append(f"包含{num_cols}列（{col_desc}等）")

        if num_rows == 0:
            return "【表格：" + "，".join(parts) + "，暂无数据行。】"

        parts.append(f"共{num_rows}行数据")
        row_parts = []
        for row_idx, row in enumerate(data_rows[:max_rows], 1):
            if num_cols <= max_cols:
                cell_parts = []
                for col_idx, cell in enumerate(row):
                    cell_val = TextPreprocessor._truncate_cell(cell)
                    if cell_val == "":
                        cell_val = "空"
                    cell_parts.append(f"{col_names[col_idx]}为{cell_val}")
                row_parts.append(f"第{row_idx}行：" + "，".join(cell_parts))
            else:
                vals = [TextPreprocessor._truncate_cell(c) for c in row]
                row_parts.append(f"第{row_idx}行：" + "、".join(vals))

        row_text = "。".join(row_parts)
        if num_rows > max_rows:
            row_text += f"。后{num_rows - max_rows}行数据略"

        # 保护破折号：替换为–（en-dash U+2013），避免被clean_text误删
        result = "【表格：" + "，".join(parts) + "。" + row_text + "。】"
        result = result.replace("-", "–")
        return result

    @staticmethod
    def convert_tables_to_text(text: str) -> str:
        """检测文本中所有 Markdown 表格块，替换为自然语言描述

        逐行状态机：表头+分隔行→收集数据行→转换→继续扫描
        """
        if not text:
            return ""

        lines = text.split("\n")
        result_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]
            if (TextPreprocessor._looks_like_table_header(line)
                    and i + 1 < len(lines)
                    and TextPreprocessor._looks_like_table_separator(lines[i + 1])):
                table_lines = [line, lines[i + 1]]
                i += 2
                while (i < len(lines)
                       and TextPreprocessor._looks_like_table_data_row(lines[i])):
                    table_lines.append(lines[i])
                    i += 1
                nl_text = TextPreprocessor._convert_table_block(table_lines)
                if nl_text:
                    result_lines.append(nl_text)
            else:
                result_lines.append(line)
                i += 1

        return "\n".join(result_lines)

    def clean_text(self, text: str) -> str:
        """Remove images, formatting symbols, and excess whitespace from note text."""
        if not text:
            return ""

        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[#*`>~\-]", "", text)
        text = text.strip()

        from func.logme import log
        if len(text) < 10:
            log.warning(f"清理后文本过短，不到10个字符。清理前为: {text[:50]}...")
        return text

    def preprocess_for_embedding(self, text: str) -> str:
        """Preprocess text specifically for embedding model input."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)

        strong_interjections = [r"我操\s*", r"他妈\s*的", r"哈哈哈\s*"]
        for pattern in strong_interjections:
            text = re.sub(pattern, "", text)
        return text.strip()

    def convert_health_data_to_text(self, raw_content: str) -> str:
        """Convert structured health data lines to natural language."""
        lines = [line for line in raw_content.strip().split("\n") if line]
        converted_lines = []

        for line in lines:
            line = line.strip()
            if re.match(r"^\d+[，,]\s*\d+[:：]\d+([，,]\s*\d+)?$", line):
                parts = re.split(r"[，,]\s*", line)
                if len(parts) >= 2:
                    steps = parts[0]
                    desc = f"今日步数{steps}步，"
                    sleep_time = parts[1].replace("：", ":")
                    if ":" in sleep_time:
                        sleep_parts = sleep_time.split(":")
                        if len(sleep_parts) == 2:
                            desc += f"睡眠时长{sleep_parts[0]}小时{sleep_parts[1]}分钟"
                    if len(parts) >= 3 and parts[2].isdigit():
                        desc += f"，喝啤酒{parts[2]}瓶"
                    line = desc + "。"
            converted_lines.append(line)

        return "\n".join(converted_lines)

    def condense_dense_lists(self, text: str) -> str:
        """Condense dense name lists to reduce token count."""
        pattern = r"([一-龥]{2,4}、){3,}[一-龥]{2,4}"
        matches = re.findall(pattern, text)

        from func.logme import log
        for match in matches:
            original = match
            names = original.split("、")
            if len(names) > 4:
                condensed = f"{names[0]}、{names[1]}等{len(names)}人"
                text = text.replace(original, condensed)
                log.debug(f"浓缩密集列表: {original[:20]}... -> {condensed}")
        return text

    def aggressive_text_reduction(self, text: str) -> str:
        """Aggressive text reduction when model reports input too long."""
        safe_length = 400
        len_text = len(text)
        if len_text > safe_length:
            text = text[:safe_length]
            from func.logme import log
            log.debug(f"[激进缩减] 从{len_text}字符缩减至{safe_length}字符")

        from func.logme import log
        if "：" in text and "、" in text and len(text.splitlines()) < 5:
            log.debug(f"检测到密集人名列表，进行针对性清理。该文本块头为：{text[:200]} ……")
            text = text.replace("\n", "").replace(" ", "")
            text = text.replace("、", ",")
            if len(text) > safe_length:
                parts = text.split(",")
                if len(parts) > 10:
                    text = ",".join(parts[:10]) + "...（名单截断）"

        text = re.sub(r"(.)\1{2,}", r"\1", text)
        return text

    def reduce_text_length(self, text: str, max_chars: int = 400) -> str:
        """Smart text length reduction, preserving high-information-density content."""
        if len(text) <= max_chars:
            return text

        original_len = len(text)
        from func.logme import log
        log.warning(f"[智能缩减] 文本过长({original_len}字符)，启动智能缩减")

        lines = text.splitlines()
        filtered_lines = [
            line for line in lines
            if line.strip() and not re.match(r"^[\s\-*=•·●○◆◇■□▣▢▤▥▦▧▨▩▱▰]*$", line.strip())
        ]
        text = "\n".join(filtered_lines)
        text = self.condense_dense_lists(text)

        if len(text) > max_chars:
            sentences = re.split(r"(?<=[。！？；\n])", text)
            important_sentences = []
            for sent in sentences:
                if (
                    re.search(r"\d{4}年\d{1,2}月\d{1,2}日", sent)
                    or re.search(r"\b(总计|合计|主要|关键|总结|认为|记录|建议)\b", sent)
                    or re.search(r"[A-Za-z一-龥]{2,}：[^。]+", sent)
                ):
                    important_sentences.append(sent)
            if important_sentences:
                text = "".join(important_sentences)
                log.debug("通过关键句提取缩减文本。")

        if len(text) > max_chars:
            truncated = text[:max_chars]
            last_para_break = truncated.rfind("\n\n")
            if last_para_break > max_chars * 0.5:
                text = (
                    truncated[:last_para_break]
                    + f"\n\n【注：因长度限制，后续内容已省略。原始文本共{original_len}字符。】"
                )
            else:
                text = truncated + f"...【文本截断，原始长度{original_len}字符】"
            log.warning("文本经智能缩减后仍超长，已进行截断并添加标记。")

        log.info(f"[智能缩减完成] {original_len}字符 -> {len(text)}字符")
        return text

    @staticmethod
    def normalize_date_string(date_str: str) -> str:
        """Normalize date formats to unified 'YYYY年M月D日' format."""
        if not date_str or not isinstance(date_str, str):
            return ""

        patterns = [
            r'(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})[日号]',
            r'(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})',
            r'(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})',
        ]

        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                year = match.group('year')
                month = str(int(match.group('month')))
                day = str(int(match.group('day')))
                return f"{year}年{month}月{day}日"

        return date_str

    def normalize_single_date_unit(self, raw_text: str, captured_date: str) -> str:
        """Normalize a single date unit's text."""
        lines = raw_text.splitlines()
        if not lines:
            return ""

        normalized_captured_date = self.normalize_date_string(captured_date)
        normalized_header = f"### {normalized_captured_date}"

        body_lines = []
        for line in lines[1:]:
            if body_lines or line.strip():
                body_lines.append(line.rstrip())

        while body_lines and not body_lines[-1].strip():
            body_lines.pop()

        if not body_lines:
            return normalized_header
        else:
            normalized_body = "\n".join(body_lines)
            return f"{normalized_header}\n\n{normalized_body}"

    @staticmethod
    def is_valid_chunk(text: str, min_length: int = 10) -> bool:
        """Check if a text chunk is valid (non-empty, non-trivial, non-symbol-only)."""
        if not text or len(text.strip()) < min_length:
            return False
        if re.match(r"^[\s\\d\\W]+$", text):
            return False
        return True
