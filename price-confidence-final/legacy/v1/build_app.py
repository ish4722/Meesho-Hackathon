"""Build the single-file web app: inject model.json and engine.js into app_template.html."""
import json, os
model = json.load(open("model.json"))
engine = open("engine.js", encoding="utf-8").read()
tpl = open("app_template.html", encoding="utf-8").read()
out = tpl.replace("/*__MODEL__*/null", json.dumps(model, separators=(",", ":"))).replace("/*__ENGINE__*/", engine)
os.makedirs("dist", exist_ok=True)
open("dist/index.html", "w", encoding="utf-8").write(out)
print("dist/index.html", round(len(out.encode()) / 1024), "KB")
