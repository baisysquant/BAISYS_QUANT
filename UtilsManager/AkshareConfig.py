from __future__ import annotations

import functools
import os

_INITIALIZED = False

# FIX(E1): 企业内网代理自签名证书场景的显式降级开关（默认关闭，保持安全基线）。
# 项目安全策略（AShareHubClient）默认"绝不降级 verify=False"，仅信任系统 CA 或
# SSL_CERT_FILE/REQUESTS_CA_BUNDLE。但 StPitSync / ListingDaysSync 通过 akshare
# 拉取 ST/退市/上市日数据时，若代理证书链不被系统信任，会 SSL 验证失败 → 同步 0 条
# → 回测 ST 禁买/退市强平/IPO 豁免全部失效（数据可信度问题 ⑤）。
# 在可控内网环境可设 BAISYS_AKSHARE_SSL_INSECURE=1 显式允许 verify=False 降级重试；
# 不设置时行为与旧版一致（尊重系统 CA，绝不过验）。
_SSL_INSECURE = os.environ.get("BAISYS_AKSHARE_SSL_INSECURE", "").strip().lower() in ("1", "true", "yes", "on")


def akshare_ssl_insecure() -> bool:
    """当前是否启用 akshare SSL 降级（BAISYS_AKSHARE_SSL_INSECURE=1）。"""
    return _SSL_INSECURE


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
            if "verify" not in kwargs:
                kwargs["verify"] = not _SSL_INSECURE
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
            if "verify" not in kwargs:
                kwargs["verify"] = not _SSL_INSECURE
            return _orig_req(self, method, url, **kwargs)

        requests.Session.request = _session_with_timeout
    except Exception:
        pass

    _INITIALIZED = True