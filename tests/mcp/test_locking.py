import concurrent.futures
import threading
import time

from pyqmd_mlx.mcp.server import _locked


def test_locked_serializes_concurrent_calls():
    """The mcp SDK dispatches sync tool functions via anyio.to_thread.run_sync
    (a real thread pool) -- two concurrent tool calls can run on two
    different threads at once. This proves _locked actually serializes
    access: two 50ms "critical sections" run through _locked from a real
    ThreadPoolExecutor never interleave -- one task's enter+exit is always
    contiguous before the other's enter appears."""
    events: list[str] = []
    record_lock = threading.Lock()

    def make_task(name: str):
        def _inner():
            with record_lock:
                events.append(f"{name}-enter")
            time.sleep(0.05)
            with record_lock:
                events.append(f"{name}-exit")
            return name

        return lambda: _locked(_inner)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(make_task("a")), executor.submit(make_task("b"))]
        results = {f.result(timeout=5) for f in futures}

    assert results == {"a", "b"}
    assert events in (
        ["a-enter", "a-exit", "b-enter", "b-exit"],
        ["b-enter", "b-exit", "a-enter", "a-exit"],
    )
