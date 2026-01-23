from ultralytics import YOLO
import pandas as pd

model = YOLO(r"yolo11n\runs\detect\train\weights\best.pt")

results = model.val(
    data="data.yaml",
    split="test",
    imgsz=640,
    batch=16,
    workers=0)

print("Done. Overall test metrics:")
print(results.results_dict) 

per_class = results.summary(decimals=6) 
df = pd.DataFrame(per_class)

df = df[["Class", "Images", "Instances", "Box-P", "Box-R", "Box-F1", "mAP50", "mAP50-95"]]
df = df.sort_values("mAP50", ascending=False).reset_index(drop=True)

print("\nPer-class metrics (TEST):")
print(df.to_string(index=False))

df.to_csv("test_per_class_map50.csv", index=False)
df.to_excel("test_per_class_map50.xlsx", index=False)
