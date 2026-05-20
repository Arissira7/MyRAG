"""配置加载模块。

从 config.yaml 加载所有配置，支持环境变量覆盖。
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml


class Config:
    """配置管理类，统一管理所有配置项。"""

    _instance: Optional["Config"] = None
    _config: dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """从 config.yaml 加载配置。"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def reload(self) -> None:
        """重新加载配置。"""
        self._load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号路径访问。

        Args:
            key: 配置键，支持 "section.subsection.key" 格式
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def get_llm_config(self) -> dict[str, Any]:
        return self._config.get("llm", {})

    def get_embedding_config(self) -> dict[str, Any]:
        return self._config.get("embedding", {})

    def get_reranker_config(self) -> dict[str, Any]:
        return self._config.get("reranker", {})

    def get_document_config(self) -> dict[str, Any]:
        return self._config.get("document", {})

    def get_retrieval_config(self) -> dict[str, Any]:
        return self._config.get("retrieval", {})

    def get_dynamic_scheduler_config(self) -> dict[str, Any]:
        return self._config.get("dynamic_scheduler", {})

    def get_query_rewriter_config(self) -> dict[str, Any]:
        return self._config.get("query_rewriter", {})

    def get_tenant_config(self) -> dict[str, Any]:
        return self._config.get("tenant", {})

    def get_storage_config(self) -> dict[str, Any]:
        return self._config.get("storage", {})

    def get_evaluation_config(self) -> dict[str, Any]:
        return self._config.get("evaluation", {})


def get_config() -> Config:
    """获取全局配置单例。"""
    return Config()
