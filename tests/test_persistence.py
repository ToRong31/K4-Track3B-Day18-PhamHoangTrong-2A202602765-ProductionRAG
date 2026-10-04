import errno
import json

import pytest

from src import persistence


def test_temporary_windows_error_retries_without_truncating_cache(tmp_path, monkeypatch):
    target = tmp_path / "cache.json"
    target.write_text('{"previous": true}', encoding="utf-8")
    replace = persistence.os.replace
    attempts = []

    def temporarily_locked(source, destination):
        attempts.append(source)
        assert json.loads(target.read_text(encoding="utf-8")) == {"previous": True}
        if len(attempts) == 1:
            raise OSError(errno.EINVAL, "Invalid argument")
        replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", temporarily_locked)
    monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
    persistence.write_json_atomic(target, {"text": "Tiếng Việt"})
    assert len(attempts) == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"text": "Tiếng Việt"}
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("error_code,expected_attempts", [(errno.EINVAL, 5), (errno.ENOSPC, 1)])
def test_failed_write_preserves_last_checkpoint(tmp_path, monkeypatch, error_code, expected_attempts):
    target = tmp_path / "cache.json"
    target.write_text('{"previous": true}', encoding="utf-8")
    attempts = []

    def fail(source, destination):
        attempts.append(source)
        raise OSError(error_code, "Write failed")

    monkeypatch.setattr(persistence.os, "replace", fail)
    monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
    with pytest.raises(OSError) as caught:
        persistence.write_json_atomic(target, {"new": True})
    assert caught.value.errno == error_code
    assert len(attempts) == expected_attempts
    assert json.loads(target.read_text(encoding="utf-8")) == {"previous": True}
    assert list(tmp_path.iterdir()) == [target]
