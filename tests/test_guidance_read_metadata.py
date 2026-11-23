"""Reading guidance may update access time without changing its content."""
from types import SimpleNamespace
import pytest
from adj_partner import profiles


@pytest.mark.parametrize("changed,allowed", [("st_atime_ns", True), ("st_mtime_ns", False), ("st_ctime_ns", False)])
def test_guidance_read_distinguishes_access_from_modification(tmp_path, monkeypatch, changed, allowed):
    path = tmp_path / "guide.md"
    path.write_text("Guidance.")
    original = profiles.os.fstat
    calls = 0
    def stat(fd):
        nonlocal calls
        result = original(fd)
        values = {key: getattr(result, key) for key in dir(result) if key.startswith("st_")}
        calls += 1
        if calls == 2:
            values[changed] += 1
        return SimpleNamespace(**values)
    monkeypatch.setattr(profiles.os, "fstat", stat)
    if allowed:
        assert profiles.load_trusted_guidance(path) == "Guidance."
    else:
        with pytest.raises(ValueError, match="changed"):
            profiles.load_trusted_guidance(path)
