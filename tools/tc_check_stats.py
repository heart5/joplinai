#!/usr/bin/env python3
"""TC 生产环境 — AI增强缓存 + 探测缓存基本数据"""
import sqlite3

db = '/home/baiyefeng/work/joplinai/data/joplinai_center.db'
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

print('=== AI增强缓存 ===')
cur = conn.execute('SELECT COUNT(*) as total FROM enhance_cache')
print(f'总缓存条数: {cur.fetchone()[0]}')

for r in conn.execute('SELECT task, COUNT(*) as cnt FROM enhance_cache GROUP BY task'):
    print(f'  {r["task"]}: {r["cnt"]}')

cur = conn.execute('SELECT SUM(total_hits) as hits FROM enhance_cache')
print(f'累计命中: {cur.fetchone()[0] or 0}')

print('验证状态分布:')
for r in conn.execute("""SELECT validation_result, COUNT(*) as cnt FROM enhance_cache
    WHERE validation_result IS NOT NULL GROUP BY validation_result"""):
    print(f'  {r[0]}: {r[1]}')

print('近7天新增:')
for r in conn.execute("""SELECT DATE(created_at) as d, COUNT(*) as cnt FROM enhance_cache
    WHERE created_at >= DATE('now', '-7 days') GROUP BY d ORDER BY d DESC LIMIT 7"""):
    print(f'  {r[0]}: {r[1]}条')

print()
print('=== 文本块长度探测缓存 ===')
cur = conn.execute('SELECT COUNT(*) as total FROM probe_cache')
print(f'总探测条数: {cur.fetchone()[0]}')

for r in conn.execute('SELECT model_name, COUNT(*) as cnt, AVG(safe_len) as avg_len, '
                       'AVG(chunk_size) as avg_chunk FROM probe_cache GROUP BY model_name'):
    print(f'  模型={r["model_name"]}: {r["cnt"]}条, '
          f'平均安全长度={r["avg_len"]:.0f}字符, '
          f'平均块大小={r["avg_chunk"]:.0f}字符')

cur = conn.execute('SELECT MIN(safe_len) as mn, MAX(safe_len) as mx, AVG(safe_len) as av FROM probe_cache')
r = cur.fetchone()
print(f'安全长度范围: {r[0]:.0f}~{r[1]:.0f}, 均值={r[2]:.0f}字符')

print('按块大小分布:')
for r in conn.execute('SELECT chunk_size, COUNT(*) as cnt, AVG(safe_len) as avg_safe '
                       'FROM probe_cache GROUP BY chunk_size ORDER BY chunk_size DESC'):
    print(f'  chunk_size={r["chunk_size"]}: {r["cnt"]}条, '
          f'平均安全={r["avg_safe"]:.0f}字符')

conn.close()
