"""Qwen-only v2 artifacts do not require a dummy tfidf.pkl."""
import json
import pickle

import analyze
import server


def test_v2_model_loads_without_tfidf_file(monkeypatch, tmp_path):
    (tmp_path / "tree.pkl").write_bytes(pickle.dumps({"no_llm": "dummy-tree"}))
    (tmp_path / "thresholds.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    (tmp_path / "centroids.json").write_text(json.dumps({"backend": "api"}), encoding="utf-8")
    monkeypatch.setattr(analyze, "MODEL_DIR", tmp_path)
    analyze.load_model.cache_clear()
    try:
        tfidf, tree, thresholds, centroids = analyze.load_model()
        assert tfidf is None
        assert tree == {"no_llm": "dummy-tree"}
        assert thresholds["schema_version"] == 2
        assert centroids["backend"] == "api"
    finally:
        analyze.load_model.cache_clear()


def test_health_accepts_qwen_model_without_tfidf(monkeypatch, tmp_path):
    for name in ("tree.pkl", "thresholds.json", "centroids.json"):
        (tmp_path / name).write_bytes(b"placeholder")
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    assert server.health()["model_ready"] is True


def test_manual_email_input_is_single_line(monkeypatch, capsys):
    import builtins
    from test_my_text import get_text

    message = "Subject: Meeting Request. Hello, please confirm the meeting. Best regards, Alex."
    monkeypatch.setattr(builtins, "input", lambda: message)
    assert get_text(None) == message
    assert "одной строкой" in capsys.readouterr().out
