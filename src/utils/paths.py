from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """将相对路径解析为项目根目录下的绝对路径。"""
    path = Path(path)
    if path.is_absolute():
        raise ValueError("路径必须是相对项目根目录的相对路径")
    return project_root() / path
