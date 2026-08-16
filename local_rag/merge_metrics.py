import json
from pathlib import Path

ckpt = Path("data/checkpoints")
backup = json.loads((ckpt / "step3_metrics__parent_child_BACKUP.json").read_text())
latest = json.loads((ckpt / "step3_metrics__parent_child.json").read_text())

merged = {**backup, **latest}   # latest wins on any overlapping key
(ckpt / "step3_metrics__parent_child.json").write_text(json.dumps(merged, indent=2))
print("Merged. Combined entries:", list(merged.keys()))