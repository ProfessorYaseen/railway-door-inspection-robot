"""Telegram alerts (text + annotated photo) for inspection events."""

from __future__ import annotations

import datetime
import os
import time
from typing import Dict, Iterable, Optional

import requests

API = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    """Minimal Telegram bot client.

    Credentials are read from the environment (never hard-code them):

    * ``TELEGRAM_TOKEN``   - token from @BotFather
    * ``TELEGRAM_CHAT_ID`` - optional; if missing it is discovered from the
      bot's most recent incoming message (send your bot "hi" once).

    If no token is configured, messages are printed instead of sent, so the
    rest of the system runs unchanged without Telegram.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        alert_classes: Iterable[str] = ("Door_Open", "Rubber_missing"),
        cooldown_s: float = 15.0,
    ):
        self.token = token if token is not None else os.environ.get("TELEGRAM_TOKEN")
        self.chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID")
        self.alert_classes = set(alert_classes)
        self.cooldown_s = cooldown_s
        self._last_alert: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def _url(self, method: str) -> str:
        return API.format(token=self.token, method=method)

    def _discover_chat_id(self) -> Optional[str]:
        try:
            r = requests.get(self._url("getUpdates"), timeout=5).json()
            if r.get("result"):
                return str(r["result"][-1]["message"]["chat"]["id"])
        except Exception as e:  # network / JSON errors must never crash the robot
            print("[TG] chat id discovery failed:", e)
        return None

    def send(self, text: str, img=None) -> None:
        """Send a text message, optionally with a BGR ``numpy`` image."""
        if not self.token:
            print("[TG] (no token set) ", text)
            return
        if self.chat_id is None:
            self.chat_id = self._discover_chat_id()
            if self.chat_id is None:
                print("[TG] no chat id; message your bot once.")
                return
        try:
            if img is not None:
                import cv2  # local import: keeps this module importable without OpenCV

                _, buf = cv2.imencode(".jpg", img)
                requests.post(
                    self._url("sendPhoto"),
                    data={"chat_id": self.chat_id, "caption": text},
                    files={"photo": ("detection.jpg", buf.tobytes())},
                    timeout=10,
                )
            else:
                requests.post(
                    self._url("sendMessage"),
                    data={"chat_id": self.chat_id, "text": text},
                    timeout=10,
                )
        except Exception as e:
            print("[TG] send failed:", e)

    def maybe_alert(self, cls_name: str, frame) -> bool:
        """Alert on a defect class, rate-limited per class. Returns True if sent."""
        now = time.time()
        if cls_name not in self.alert_classes:
            return False
        if now - self._last_alert.get(cls_name, 0.0) <= self.cooldown_s:
            return False
        self._last_alert[cls_name] = now
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.send(f"[{ts}] Detected: {cls_name}", frame)
        return True
