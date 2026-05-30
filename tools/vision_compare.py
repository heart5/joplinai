"""Vision model comparison: Qwen3-VL-32B vs Qwen3-VL-8B on 5 images from 3 notes.

Selects 3 notes from 烟物缭绕, picks 5 images total, runs both models,
creates a Joplin comparison note.
"""
import base64
import pathmagic
import re
import time

with pathmagic.Context():
    from func.jpfuncs import (
        createnote,
        get_notes_in_notebook_by_title,
        getinivaluefromcloud,
        jpapi,
        searchnotebook,
    )
    from func.logme import log
    from aimod.note_enhancer import describe_images
    from aimod.image_processor import ImageProcessor

# --- Config ---
NOTEBOOK = "烟物缭绕"
MODEL_32B = "Qwen/Qwen3-VL-32B-Instruct"
MODEL_8B = "Qwen/Qwen3-VL-8B-Instruct"

# 3 target notes (note_id -> max_images)
TARGET_NOTES = {
    "94690027c8944bddb05e4c0dd758646b": 2,   # 黄鹤楼细枝感恩
    "98174d03b0b44a9d937b49055ed2eda5": 2,   # 小米音箱
    "a59b0eeb2b104604908ed7cbc9d6d77c": 1,   # 台灯LP005
}

print("=" * 60)
print("Vision Model Comparison: 32B vs 8B")
print("=" * 60)

# 1. Get notes
notes = get_notes_in_notebook_by_title(notebook_title=NOTEBOOK)
target_notes = [n for n in notes if n.id in TARGET_NOTES]
print(f"\nFound {len(target_notes)} target notes")

# 2. Extract images per note
image_proc = ImageProcessor(jpapi)
nb_id = searchnotebook(NOTEBOOK)

all_comparisons = []  # [(note_title, resource_id, mime, b64, result_32b, result_8b, time_32b, time_8b)]

for note in target_notes:
    max_imgs = TARGET_NOTES[note.id]
    body = note.body if hasattr(note, 'body') else ''
    rids = list(set(re.findall(r':/([a-f0-9]{32,})', body)))[:max_imgs]
    print(f"\n--- {note.title} ({len(rids)} images) ---")

    images = image_proc.fetch_images_for_note(note.id, rids)
    if not images:
        print(f"  Failed to fetch images")
        continue

    for rid, img_data in images.items():
        print(f"\n  Image: {rid[:16]}... ({img_data['mime']}, {len(img_data['b64'])} chars b64)")

        # 32B
        t0 = time.time()
        r32 = describe_images(
            {rid: img_data},
            context=body[:2000],
            model=MODEL_32B,
        )
        t32 = time.time() - t0

        # 8B
        t0 = time.time()
        r8 = describe_images(
            {rid: img_data},
            context=body[:2000],
            model=MODEL_8B,
        )
        t8 = time.time() - t0

        all_comparisons.append({
            'note_title': note.title,
            'note_id': note.id,
            'resource_id': rid,
            'mime': img_data['mime'],
            'b64': img_data['b64'],
            'result_32b': r32,
            'result_8b': r8,
            'time_32b': t32,
            'time_8b': t8,
        })

        print(f"    32B: {t32:.1f}s, {len(r32 or '')} chars")
        print(f"    8B:  {t8:.1f}s, {len(r8 or '')} chars")

# 3. Build comparison note
print("\n" + "=" * 60)
print("Building comparison note...")

md_parts = [
    "# Vision模型对比：Qwen3-VL-32B vs Qwen3-VL-8B",
    "",
    f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"来源笔记本：{NOTEBOOK}",
    f"测试图片数：{len(all_comparisons)} 张（来自 {len(set(c['note_id'] for c in all_comparisons))} 条笔记）",
    "",
    "---",
    "",
]

for i, c in enumerate(all_comparisons):
    md_parts.append(f"## 图片 {i+1}：{c['note_title']}")
    md_parts.append("")
    md_parts.append(f"- 资源ID：`{c['resource_id']}`")
    md_parts.append(f"- 格式：{c['mime']}")
    md_parts.append("")

    # Embed image
    data_url = f"data:{c['mime']};base64,{c['b64']}"
    md_parts.append(f"![图片{i+1}]({data_url})")
    md_parts.append("")

    # 32B result
    md_parts.append(f"### Qwen3-VL-32B-Instruct（{c['time_32b']:.1f}秒）")
    md_parts.append("")
    if c['result_32b']:
        md_parts.append(c['result_32b'])
    else:
        md_parts.append("> 识别失败")
    md_parts.append("")

    # 8B result
    md_parts.append(f"### Qwen3-VL-8B-Instruct（{c['time_8b']:.1f}秒）")
    md_parts.append("")
    if c['result_8b']:
        md_parts.append(c['result_8b'])
    else:
        md_parts.append("> 识别失败")
    md_parts.append("")

    md_parts.append("---")
    md_parts.append("")

# Summary
avg_32b = sum(c['time_32b'] for c in all_comparisons) / len(all_comparisons)
avg_8b = sum(c['time_8b'] for c in all_comparisons) / len(all_comparisons)
len_32b = sum(len(c['result_32b'] or '') for c in all_comparisons)
len_8b = sum(len(c['result_8b'] or '') for c in all_comparisons)

md_parts.append("## 汇总")
md_parts.append("")
md_parts.append("| 指标 | 32B | 8B |")
md_parts.append("|------|-----|-----|")
md_parts.append(f"| 平均耗时 | {avg_32b:.1f}s | {avg_8b:.1f}s |")
md_parts.append(f"| 总描述字数 | {len_32b} | {len_8b} |")
md_parts.append(f"| 成功率 | {sum(1 for c in all_comparisons if c['result_32b'])}/{len(all_comparisons)} | {sum(1 for c in all_comparisons if c['result_8b'])}/{len(all_comparisons)} |")

note_body = "\n".join(md_parts)
note_title = f"Vision模型对比 32Bvs8B {time.strftime('%m%d %H:%M')}"

# 4. Create Joplin note
print(f"Creating note: {note_title}")
print(f"Body size: {len(note_body)} chars (with {len(all_comparisons)} embedded images)")

createnote(title=note_title, body=note_body, parent_id=nb_id)
print("Note created! Now run `joplin sync` to sync...")

# 5. Sync Joplin (local - will push to cloud)
from func.sysfunc import execcmd
# execcmd("joplin sync")  # Let user sync manually
print("\nDone! Sync the note manually or via joplin sync.")
