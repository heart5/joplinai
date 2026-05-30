"""Vision 简版测试：8B + 50字限制，从指定笔记本选3张图。"""
import re, time

import pathmagic
with pathmagic.Context():
    from func.jpfuncs import (
        createnote,
        get_notes_in_notebook_by_title,
    )
    from func.logme import log
    from aimod.note_enhancer import describe_images
    from aimod.image_processor import ImageProcessor
    from func.jpfuncs import jpapi as _jpapi

NOTEBOOK = "美食消费单"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"

print(f"笔记本: {NOTEBOOK}  模型: {MODEL}")
notes = get_notes_in_notebook_by_title(notebook_title=NOTEBOOK)

img_notes = []
for n in notes:
    body = n.body if hasattr(n, 'body') else ''
    rids = list(set(re.findall(r':/([a-f0-9]{32,})', body)))
    if rids:
        img_notes.append((n, rids))

print(f"共{len(notes)}条笔记, {len(img_notes)}条有图")

image_proc = ImageProcessor(_jpapi)
results = []

for n, rids in img_notes[:3]:
    rid = rids[0]
    imgs = image_proc.fetch_images_for_note(n.id, [rid])
    if not imgs:
        print(f"  {n.title}: 图片获取失败")
        continue
    img_data = imgs[rid]
    t0 = time.time()
    desc = describe_images({rid: img_data}, context=n.body or '', model=MODEL)
    elapsed = time.time() - t0
    size_kb = len(img_data['b64']) * 3 // 4 // 1024
    print(f"  {n.title}: {elapsed:.1f}s | {len(desc or '')}字 | {size_kb}KB")
    if desc:
        print(f"    {desc[:100]}...")
    results.append({
        'title': n.title, 'rid': rid,
        'mime': img_data['mime'], 'b64': img_data['b64'],
        'desc': desc, 'time': elapsed,
    })

if not results:
    print("无有效结果"); exit()

md = [
    f"# Vision简版测试 — {MODEL.split('/')[-1]}（50字限制）",
    "",
    f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"笔记本：{NOTEBOOK}",
    f"图片数：{len(results)}",
    "",
    "---",
]
for i, r in enumerate(results):
    md.append(f"## 图片{i+1}：{r['title']}")
    md.append(f"- 耗时 {r['time']:.1f}s | {len(r['desc'] or '')}字")
    md.append(f"![图](data:{r['mime']};base64,{r['b64']})")
    md.append(f"> {r['desc']}")
    md.append("---")
md.append("## 汇总")
md.append("| 图片 | 耗时 | 字数 |")
md.append("|------|------|------|")
for r in results:
    md.append(f"| {r['title']} | {r['time']:.1f}s | {len(r['desc'] or '')} |")

title = f"Vision简版测试 8B {time.strftime('%m%d %H:%M')}"
createnote(title, "\n".join(md), NOTEBOOK)
print(f"\nNote: {title}")
