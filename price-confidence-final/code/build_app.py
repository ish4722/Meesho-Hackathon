"""Build the single-file offline web app: inject model.json, engine.js and an evaluation summary into app_template.html.

  python build_app.py            # writes dist/index.html
The model is validated (model_schema.py) before it is embedded, so the app never ships with a model it cannot read.
"""
import json
import os

import model_schema

HERE = os.path.dirname(os.path.abspath(__file__))


def eval_summary(path):
    if not os.path.exists(path):
        return None
    ev = json.load(open(path))
    h = ev["task_a_one_week_ahead"]["holdout"]["overall"]
    c = ev["task_c_recursive_12_weeks"]["holdout"]["orders_observed_weeks"]
    d = ev["decisions"]["policies"]
    return dict(a_wape=h["v2, online listing effect"]["wape"], a_persist=h["last week's orders"]["wape"],
                c_wape=c["v2 engine, recursive (median)"]["wape"], c_cat=c["category x week average"]["wape"],
                within10=d["Balanced (v2)"]["share_within_10pct_of_best"],
                within10_v1=d["v1 rule (point estimate, no risk limits)"]["share_within_10pct_of_best"], synthetic=True)


def main():
    model = json.load(open(os.path.join(HERE, "model.json")))
    model_schema.validate(model)
    engine = open(os.path.join(HERE, "engine.js"), encoding="utf-8").read()
    tpl = open(os.path.join(HERE, "app_template.html"), encoding="utf-8").read()
    ev = eval_summary(os.path.join(HERE, "reports", "evaluation.json"))
    out = (tpl.replace("/*__MODEL__*/null", json.dumps(model, separators=(",", ":")))
              .replace("/*__EVAL__*/null", json.dumps(ev))
              .replace("/*__ENGINE__*/", engine))
    os.makedirs(os.path.join(HERE, "dist"), exist_ok=True)
    with open(os.path.join(HERE, "dist", "index.html"), "w", encoding="utf-8") as f:
        f.write(out)
    print("dist/index.html", round(len(out.encode()) / 1024), "KB")


if __name__ == "__main__":
    main()
