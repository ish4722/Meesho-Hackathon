# Legacy v1 (frozen)

The exact code, model, data and demo behind the Round 2 deck and the team's demo video
(Seller 6: floor ₹145, start ₹209). Kept unchanged so those numbers stay reproducible.

- `Price_Confidence_demo_v1.html`: the v1 demo, open in a browser.
- `engine.js`, `model.json`, `app_template.html`, `build_app.py`: v1 engine and model.
- `generate_dataset.py`, `train_models.py`, `data/`: v1 generator, training and the committed CSVs.
  Note: re-running the v1 generator with numpy 2.x does not reproduce the committed t-shirt rows
  (see `../../docs/AUDIT.md`, item 4). Use `data/` for exact v1 results.
- `truth_test.py`, `truth_compare.py`, `model_compare.js`, `truth.json`: v1-only diagnostics that
  hard-code v1 generator mechanics. Superseded by `code/evaluate.py`.
- `make_models_pdf.py`, `Price_Confidence_Models_Explained.pdf`: the v1 model explainer.
  Superseded by `docs/METHODOLOGY.md`.

The v2 code in `../../code/` replaces all of this.
