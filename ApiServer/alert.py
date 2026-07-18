"""
告警通知模块 — 通过 Webhook 发送管线异常/完成通知。

支持：
  - 企业微信机器人 Webhook
  - 飞书机器人 Webhook
  - 钉钉机器人 Webhook
  - 通用 HTTP POST 回调
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from loguru import logger


class AlertSender:
    """异常/完成告警发送器。"""

    def __init__(self, webhook_url: str = "",
                 alert_on_failure: bool = True,
                 alert_on_success: bool = False,
                 alert_channel: str = "generic") -> None:
        self._webhook_url = webhook_url
        self._alert_on_failure = alert_on_failure
        self._alert_on_success = alert_on_success
        self._channel = alert_channel

    @property
    def enabled(self) -> bool:
        return bool(self._webhook_url)

    def send(self, title: str, message: str,
             level: Literal["info", "warning", "error"] = "info") -> bool:
        """发送告警消息。

        Returns:
            True 表示发送成功（或未配置），False 表示发送失败。
        """
        if not self._webhook_url:
            return True

        try:
            payload = self._build_payload(title, message, level)
            self._do_post(payload)
            logger.info(f"[AlertSender] 告警已发送: {title}")
            return True
        except Exception as e:
            logger.warning(f"[AlertSender] 发送失败: {e}")
            return False

    def notify_pipeline_result(self, run_id: str, trade_date: str,
                               status: str, summary: str = "",
                               duration: float = 0.0) -> None:
        """发送管线执行结果通知。"""
        if status == "completed" and not self._alert_on_success:
            return
        if status == "failed" and not self._alert_on_failure:
            return

        emoji = "✅" if status == "completed" else "❌"
        title = f"{emoji} BAISYS_QUANT 管线 {status}"
        message = (
            f"交易日: {trade_date}\n"
            f"RunID: {run_id}\n"
            f"状态: {status}\n"
            f"耗时: {duration:.0f}s\n"
        )
        if summary:
            message += f"摘要: {summary}\n"

        self.send(title, message, "info" if status == "completed" else "error")

    # ── 内部 ──────────────────────────────────────────────

    def _build_payload(self, title: str, message: str, level: str) -> dict[str, Any]:
        """根据 channel 构建不同格式的 payload。"""
        if self._channel == "wecom":
            return {
                "msgtype": "markdown",
                "markdown": {
                    "content": f"## {title}\n{message}\n---\n_BAISYS_QUANT 自动通知_"
                },
            }
        elif self._channel == "feishu":
            color_map = {"info": "green", "warning": "yellow", "error": "red"}
            return {
                "msg_type": "interactive",
                "card": {
                    "header": {"title": {"tag": "plain_text", "content": title},
                               "template": color_map.get(level, "green")},
                    "elements": [{"tag": "markdown", "content": message}],
                },
            }
        elif self._channel == "dingtalk":
            return {
                "msgtype": "text",
                "text": {"content": f"{title}\n{message}"},
            }
        else:
            return {
                "title": title,
                "message": message,
                "level": level,
                "source": "BAISYS_QUANT",
                "timestamp": datetime.now().isoformat(),
            }

    def _do_post(self, payload: dict[str, Any]) -> None:
        import httpx
        resp = httpx.post(
            self._webhook_url,
            json=payload,
            timeout=15.0,
        )
        resp.raise_for_status()
