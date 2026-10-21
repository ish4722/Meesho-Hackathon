"""Runs the JavaScript engine tests (tests/engine.test.js) under node, so `pytest` covers both languages."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def test_engine_js_suite():
    res = subprocess.run(["node", "--test", os.path.join(HERE, "engine.test.js")], capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-2000:]
