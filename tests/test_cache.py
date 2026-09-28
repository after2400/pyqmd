from pyqmd_mlx.llm._cache import get_or_load


def test_get_or_load_caches_by_model_id():
    calls = []

    def loader(model_id):
        calls.append(model_id)
        return f"model-for-{model_id}"

    cache: dict = {}
    result1 = get_or_load(cache, "a", loader)
    result2 = get_or_load(cache, "a", loader)

    assert result1 == "model-for-a"
    assert result2 == "model-for-a"
    assert calls == ["a"]


def test_get_or_load_reloads_for_different_model_id():
    calls = []

    def loader(model_id):
        calls.append(model_id)
        return f"model-for-{model_id}"

    cache: dict = {}
    get_or_load(cache, "a", loader)
    get_or_load(cache, "b", loader)

    assert calls == ["a", "b"]
