from __future__ import annotations

import functools
import os

_INITIALIZED = False


def ensure_akshare_timeout(timeout: int = 30) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    try:
        import akshare as ak

        _orig_ak = ak.session.request

        @functools.wraps(_orig_ak)
        def _with_timeout(method: str, url: str, **kwargs: object) -> object:
            kwargs.setdefault("timeout", timeout)
            # P3 审计修复：显式启用 SSL 验证，禁止 akshare 静默关闭
            if "verify" not in kwargs:
                kwargs["verify"] = True
            return _orig_ak(method, url, **kwargs)

        ak.session.request = _with_timeout
    except Exception:
        pass

    try:
        import requests

        _orig_req = requests.Session.request

        @functools.wraps(_orig_req)
        def _session_with_timeout(self: requests.Session, method: str, url: str, **kwargs: object) -> object:
            kwargs.setdefault("timeout", timeout)
            # P3 审计修复：显式启用 SSL 验证，使用系统证书存储
            if "verify" not in kwargs:
                kwargs["verify"] = True
            return _orig_req(self, method, url, **kwargs)

        requests.Session.request = _session_with_timeout
    except Exception:
        pass

    _INITIALIZED = True
