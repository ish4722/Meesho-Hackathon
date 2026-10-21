"""model.json contract: the shipped model validates; broken models are refused in Python and in the engine."""
import copy
import json
import os
import subprocess

import pytest

import model_schema

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_shipped_model_is_valid(shipped_model):
    model_schema.validate(shipped_model)
    assert shipped_model["meta"]["synthetic"] is True
    assert "SYNTHETIC" in shipped_model["meta"]["provenance"]


def test_small_model_is_valid(small_model):
    model_schema.validate(small_model[0])


@pytest.mark.parametrize("breakage", [
    lambda m: m.pop("assumptions"),
    lambda m: m.__setitem__("model_schema_version", "1.0"),
    lambda m: m["meta"].__setitem__("synthetic", False),
    lambda m: m["categories"]["soap"]["conv"]["coef"].__setitem__("hidden_true_quality", 1.0),
    lambda m: m["categories"]["soap"]["conv"]["coef"].pop("rel"),
    lambda m: m["categories"]["soap"]["returns"]["coef"].__setitem__("rel_pos", -1.0),
    lambda m: m["categories"]["soap"]["param_blocks"][0]["chol"].pop(),
    lambda m: m["categories"]["soap"]["product_effects"]["chol"].pop(),
    lambda m: m["categories"]["soap"]["subcats"].popitem(),
    lambda m: m["categories"]["soap"].__setitem__("resale", 1.5),
])
def test_broken_models_are_refused(shipped_model, breakage):
    m = copy.deepcopy(shipped_model)
    breakage(m)
    with pytest.raises((model_schema.ModelSchemaError, KeyError)):
        model_schema.validate(m)


def test_engine_refuses_wrong_schema_version(shipped_model, tmp_path):
    m = copy.deepcopy(shipped_model)
    m["model_schema_version"] = "1.0"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    script = "const E=require(process.argv[1]);const M=require(process.argv[2]);try{E.recommend(M,{category:'soap',product:60});console.log('ran')}catch(e){console.log('refused:'+e.message)}"
    out = subprocess.run(["node", "-e", script, os.path.join(CODE, "engine.js"), str(p)], capture_output=True, text=True, check=True).stdout
    assert out.startswith("refused")
