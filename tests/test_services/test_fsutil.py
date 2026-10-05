"""replace_file: thay file có thử lại khi Windows báo file đang bị giữ."""

import os

import pytest

from src.services import fsutil


def test_replace_file_retries_when_target_is_briefly_locked(tmp_path, monkeypatch):
    tmp, dst = tmp_path / "a.tmp", tmp_path / "a.json"
    tmp.write_text("new")
    dst.write_text("old")
    real, calls = os.replace, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) < 3:  # hai lần đầu: file đích đang bị process khác mở (WinError 5)
            raise PermissionError(13, "Access is denied")
        real(a, b)

    monkeypatch.setattr(fsutil.os, "replace", flaky)
    fsutil.replace_file(tmp, dst, delay_s=0)
    assert dst.read_text() == "new" and len(calls) == 3


def test_replace_file_gives_up_after_all_attempts(tmp_path, monkeypatch):
    def locked(a, b):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(fsutil.os, "replace", locked)
    with pytest.raises(PermissionError):
        fsutil.replace_file(tmp_path / "a.tmp", tmp_path / "a.json", attempts=3, delay_s=0)
