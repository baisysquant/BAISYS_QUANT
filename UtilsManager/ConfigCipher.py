from __future__ import annotations

import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet


class ConfigCipher:
    """
    本地对称加密工具，用于保护配置文件中的敏感信息（如数据库密码）。

    密钥路径优先级：
      1. 实例化时传入的 key_path
      2. ConfigCipher.default_key_path（在 ConfigParser 启动时设置）
      3. 默认 ~/.baisys_quant_key

    加密后的密文以 ENC: 为前缀存储在配置文件中。

    用法：
        cipher = ConfigCipher()
        encrypted = cipher.encrypt("my_password")
        plaintext = cipher.decrypt(encrypted)
    """

    default_key_path: str | Path | None = None

    def __init__(self, key_path: str | Path | None = None) -> None:
        self._key_path = key_path or ConfigCipher.default_key_path or Path.home() / ".baisys_quant_key"
        self._fernet: Fernet | None = None

    @property
    def key_file(self) -> Path:
        return Path(self._key_path)

    def _get_fernet(self) -> Fernet:
        if self._fernet is None:
            self._fernet = Fernet(self._load_or_create_key())
        return self._fernet

    def _load_or_create_key(self) -> bytes:
        key_file = self.key_file
        if key_file.exists():
            return key_file.read_bytes()
        key = Fernet.generate_key()
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(key)
        os.chmod(str(key_file), 0o600)
        return key

    def encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        return self._get_fernet().encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        if not ciphertext:
            return ""
        return self._get_fernet().decrypt(ciphertext.encode()).decode()

    @staticmethod
    def is_encrypted(value: str) -> bool:
        return bool(value) and value.startswith("ENC:")

    @staticmethod
    def looks_like_fernet_token(value: str) -> bool:
        """检测是否是裸 Fernet token（缺失 ENC: 前缀）。"""
        return bool(value) and value.startswith("gAAAAA")

    @staticmethod
    def strip_prefix(value: str) -> str:
        return value[4:] if value.startswith("ENC:") else value

    @staticmethod
    def maybe_decrypt(value: str) -> str:
        if ConfigCipher.is_encrypted(value):
            return ConfigCipher().decrypt(ConfigCipher.strip_prefix(value))
        if ConfigCipher.looks_like_fernet_token(value):
            return ConfigCipher().decrypt(value)
        return value


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("encrypt", "decrypt"):
        print("用法: python -m UtilsManager.ConfigCipher encrypt|decrypt <文本>")
        sys.exit(1)

    cipher = ConfigCipher()
    action = sys.argv[1]
    text = sys.argv[2]

    if action == "encrypt":
        result = cipher.encrypt(text)
        print(f"ENC:{result}")
    else:
        result = cipher.decrypt(text)
        print(result)


if __name__ == "__main__":
    main()
