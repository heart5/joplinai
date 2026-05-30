"""Find corrupted notes in 烟物缭绕 via raw HTTP (bypass joppy ID validation)."""
import pathmagic, requests
with pathmagic.Context():
    from func.jpfuncs import jpapi

nbid = "d91ab154f0204205a58755f7922d3fe1"
base = jpapi.url.rstrip("/")
token = jpapi.token

all_items = []
page = 1
while True:
    resp = requests.get(f"{base}/notes", params={
        "token": token, "parent_id": nbid,
        "fields": "id,title,parent_id", "page": page
    })
    data = resp.json()
    items = data.get("items", [])
    if not items:
        break
    all_items.extend(items)
    if not data.get("has_more"):
        break
    page += 1

print(f"Total: {len(all_items)} items")
bad = []
for item in all_items:
    pid = item.get("id", "")
    ppid = item.get("parent_id", "")
    if len(pid) != 32 or (ppid and len(ppid) != 32):
        print(f"CORRUPT: id={pid!r} parent_id={ppid!r} title={item.get('title')!r}")
        bad.append(item)
if not bad:
    print("No corruption in 烟物缭绕")
else:
    print(f"\nFound {len(bad)} bad items. Run delete below:")
    for item in bad:
        print(f"Delete: id={item.get('id')!r} title={item.get('title')!r}")
