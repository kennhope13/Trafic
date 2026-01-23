from pathlib import Path
import sys, shutil, os
import json
import pandas as pd

BEST_PT  = r"pretrain_backbone_yolo11n\runs\detect\train2\weights\best.pt"
DATA_YAML = r"yolo11n\data.yaml"
# =================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)) 

import ultralytics.nn.tasks as tasks
from custom_layers.nn.modules.uib import UIB, UIBDown
tasks.UIB = UIB
tasks.UIBDown = UIBDown

from ultralytics.utils.torch_utils import strip_optimizer
from ultralytics import YOLO


def main():
    best = Path(BEST_PT)
    assert best.exists(), f"Không thấy best.pt: {best}"

    best_strip = best.with_name(best.stem + "_uib.pt")
    shutil.copy2(best, best_strip)

    print(f"Original best.pt: {os.path.getsize(best)/1024/1024:.2f} MB")
    strip_optimizer(str(best_strip))
    print(f"Stripped best_strip.pt: {os.path.getsize(best_strip)/1024/1024:.2f} MB")

    # ====== ĐÁNH GIÁ TRÊN TEST ======
    model = YOLO(str(best_strip))

    results = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        batch=16,
        workers=0
    )

    print("Done. Test metrics:")
    print(results)


    out_dir = best.parent / "eval_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = getattr(results, "results_dict", None)
    if overall is None:
        overall = {"results_str": str(results)}

    overall_path = out_dir / "test_overall_metrics.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print(f"Saved overall metrics: {overall_path}")

    if not hasattr(results, "summary"):
        raise RuntimeError("Ultralytics version của bạn không có results.summary() để lấy per-class metrics.")

    rows = results.summary(decimals=6)  
    df = pd.DataFrame(rows)

    cols_want = ["Class", "Instances", "Box-P", "Box-R", "Box-F1", "mAP50", "mAP50-95"]
    cols_exist = [c for c in cols_want if c in df.columns]
    df = df[cols_exist].copy()

    if "mAP50" in df.columns:
        df = df.sort_values("mAP50", ascending=False, na_position="last")

    csv_path = out_dir / "test_map50_per_class.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Saved per-class CSV: {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
