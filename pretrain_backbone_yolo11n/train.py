from pathlib import Path
import sys, inspect, re

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)) 

import ultralytics
print("Ultralytics loaded from:", ultralytics.__file__)

import ultralytics.nn.tasks as tasks
from custom_layers.nn.modules.uib import UIB, UIBDown

tasks.UIB = UIB
tasks.UIBDown = UIBDown
print("UIBDown registered?", "UIBDown" in tasks.__dict__)

print("YAML path:", Path("yolo11n_uib_backbone.yaml").resolve())

def patch_parse_model_for_uib():
    src = inspect.getsource(tasks.parse_model)

    if "UIBDown" in src and "base_modules" in src and "set(base_modules)" in src:
        return

    lines = src.splitlines()
    out = []
    inserted = False
    in_base = False
    bal = 0
    indent_after = ""

    def delta_balance(line: str):
        return (line.count("{") - line.count("}")) + (line.count("(") - line.count(")"))

    for line in lines:
        out.append(line)

        if (not inserted) and (not in_base) and re.match(r"^\s*base_modules\s*=", line):
            in_base = True
            bal = delta_balance(line)
            indent_after = re.match(r"^(\s*)", line).group(1)
            if bal == 0:
                out.append(indent_after + "base_modules = set(base_modules) | {UIB, UIBDown}")
                out.append(indent_after + "base_modules = frozenset(base_modules)")
                inserted = True
                in_base = False

        elif in_base:
            bal += delta_balance(line)
            if bal == 0:
                out.append(indent_after + "base_modules = set(base_modules) | {UIB, UIBDown}")
                out.append(indent_after + "base_modules = frozenset(base_modules)")
                inserted = True
                in_base = False

    if not inserted:
        raise RuntimeError("Could not auto-patch parse_model(): base_modules definition not found.")

    patched = "\n".join(out)

    g = tasks.__dict__
    g.update({"UIB": UIB, "UIBDown": UIBDown})
    exec(patched, g)
    tasks.parse_model = g["parse_model"]
    print("Patched tasks.parse_model to include UIB/UIBDown in base_modules")

patch_parse_model_for_uib()

from ultralytics import YOLO

model = YOLO("yolo11n_uib_backbone.yaml") 
model.train(
    data=r"yolo11n\data.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    workers=0
)
