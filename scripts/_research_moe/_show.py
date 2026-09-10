import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

for f, tag in [("w_mmap", "MMAP"), ("w_gemma4", "GEMMA4")]:
    j = json.load(open(rf"C:\Users\sucot\AppData\Local\Temp\dsh-QXV39g\{f}.json", encoding="utf-8"))
    print(f"===== {tag} =====")
    for t in j["top"]:
        if t.get("alive"):
            print(f"[{t['handle']}] rel={t['relevance']} {t['title'][:70]}")
            print("   ", t["url"])
            print("   ", t["text"][:300].replace("\n", " "))
            print()
