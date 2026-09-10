import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

for f in ["w_quantml", "w_gemma3060"]:
    j = json.load(open(rf"C:\Users\sucot\AppData\Local\Temp\dsh-QXV39g\{f}.json", encoding="utf-8"))
    print(f"===== {f}: source={j.get('source')} chars={j.get('chars')} =====")
    wrapped = j.get("text", "")
    for i in range(0, min(len(wrapped), 20000), 4000):
        print(wrapped[i:i + 4000])
        print("---- chunk ----")
