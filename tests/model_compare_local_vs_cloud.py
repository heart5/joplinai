#!/usr/bin/env python3
"""
本地模型 (qwen2.5:1.5b) vs DeepSeek API 摘要/标签对比测试
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pathmagic

with pathmagic.Context():
    from aimod.vector_db_manager import VectorDBManager
    from aimod.note_enhancer import local_process_note, deepseek_process_note
    from joplinai import CONFIG as CONFIG_JA
    from src.qa_config import CONFIG as CONFIG_QA

config_all = {**CONFIG_JA, **CONFIG_QA}

TEST_NOTES = [
    ("684a69aadc6241978f8db160797af2db", "新意向客户打款前铺网点验证动销的工作要求"),
    ("b836f9482c484e33836febd484b10e20", "生活原则和框架"),
    ("bbcf64ca21ff4df09afe7a11f4a6f058", "做大事的人身上的四种特质"),
]

def sep(char="=", w=80):
    print(char * w)

# 加载笔记
print("加载笔记内容...")
vector_db = VectorDBManager(
    config_all["db_path"], config_all["ollama_embedding_model"], for_creation=False
)

notes_data = []
for note_id, title in TEST_NOTES:
    results = vector_db.collection.get(
        where={"source_note_id": note_id},
        include=["documents", "metadatas"]
    )
    full_text = "\n".join(results["documents"])
    notes_data.append({"id": note_id, "title": title, "text": full_text,
                       "chunks": len(results["ids"])})

# 展示笔记内容
for nd in notes_data:
    sep("─")
    print(f"\n笔记: {nd['title']}")
    print(f"ID: {nd['id']}  |  块数: {nd['chunks']}  |  总字数: {len(nd['text'])}")
    print("─" * 80)
    print(nd['text'])

# 执行对比
all_results = []

for nd in notes_data:
    sep()
    print(f"\n  ==== 测试: {nd['title']} ====\n")
    
    note_result = {"title": nd["title"], "id": nd["id"],
                   "text_len": len(nd['text']), "text": nd["text"],
                   "tasks": {}}
    
    for task in ["summary", "tags"]:
        print(f"  [{task}] 调用中...\n")
        
        # 本地模型
        print(f"    🖥️  本地 qwen2.5:1.5b ...", end=" ", flush=True)
        t0 = time.time()
        local_result = local_process_note(nd["text"], task=task,
                                          model="qwen2.5:1.5b", use_cache=False,
                                          ollama_host="http://127.0.0.1:11434")
        local_time = round(time.time() - t0, 1)
        print(f"({local_time}s)")
        print(f"    {local_result[:200] if local_result else 'ERROR: 无结果'}")
        
        # DeepSeek API
        print(f"    ☁️  DeepSeek API ...", end=" ", flush=True)
        t0 = time.time()
        ds_result = deepseek_process_note(nd["text"], task=task, use_cache=False)
        ds_time = round(time.time() - t0, 1)
        print(f"({ds_time}s)")
        print(f"    {ds_result[:200] if ds_result else 'ERROR: 无结果'}")
        
        note_result["tasks"][task] = {
            "ollama": {"result": local_result, "time": local_time},
            "deepseek": {"result": ds_result, "time": ds_time},
        }
        print()
    
    all_results.append(note_result)

# 汇总
sep()
print("\n  ==== 汇总对比 ====\n")

for nr in all_results:
    print(f"  笔记: {nr['title']} ({nr['text_len']}字)")
    print(f"  {'─'*76}")
    print(f"  {'任务':<8s} {'模型':<20s} {'耗时':>8s} {'输出长度':>8s}")
    print(f"  {'─'*76}")
    for task in ["summary", "tags"]:
        for model_key, model_name in [("ollama", "qwen2.5:1.5b"), ("deepseek", "DeepSeek API")]:
            tr = nr["tasks"][task][model_key]
            result_text = tr["result"] or "N/A"
            time_str = f"{tr['time']}s"
            print(f"  {task:<8s} {model_name:<20s} {time_str:>8s} {len(result_text):>8d}字")
            print(f"           → {result_text}")
            print()
    print()

# 速度汇总
print("  ⏱️  速度对比:")
print(f"  {'笔记':<30s} {'本地摘要':>10s} {'云端摘要':>10s} {'本地标签':>10s} {'云端标签':>10s}")
print(f"  {'─'*75}")
for nr in all_results:
    local_s = f"{nr['tasks']['summary']['local']['time']}s"
    ds_s = f"{nr['tasks']['summary']['deepseek']['time']}s"
    local_t = f"{nr['tasks']['tags']['local']['time']}s"
    ds_t = f"{nr['tasks']['tags']['deepseek']['time']}s"
    print(f"  {nr['title'][:30]:<30s} {local_s:>10s} {ds_s:>10s} {local_t:>10s} {ds_t:>10s}")

sep()
print("  测试完成。")
sep()

# 保存结果
output = os.path.join(os.path.dirname(__file__), "benchmark_local_vs_cloud.json")
with open(output, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存至: {output}")
