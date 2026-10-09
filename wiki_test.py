import requests

UA = "multi-agent-pipeline-coursework/1.0 (educational project; no commercial use)"
BASE = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

for title in ["Mona_Lisa", "The_Starry_Night", "Girl_with_a_Pearl_Earring"]:
    for label, headers in [("project UA", {"User-Agent": UA}),
                           ("no UA", {})]:
        try:
            r = requests.get(BASE.format(title), headers=headers, timeout=8)
            extract = r.json().get("extract", "")[:60] if r.status_code == 200 else ""
            print(f"{title:28} {label:10} status={r.status_code} extract={extract!r}")
        except Exception as e:
            print(f"{title:28} {label:10} ERROR {e}")