from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


NON_ASCII_CLASS_WARNING = (
    "检测到中文类别名，Ultralytics 可能下载 Arial.Unicode.ttf。离线环境建议使用英文类别名 data.yaml。"
)


class DataYamlError(RuntimeError):
    pass


@dataclass(frozen=True)
class DataYamlInspection:
    path: Path
    names: list[str]
    has_non_ascii: bool
    ascii_copy_path: Path
    ascii_copy_exists: bool


def inspect_data_yaml_names(data_yaml_path: str | Path) -> DataYamlInspection:
    path = Path(data_yaml_path)
    data = _load_yaml(path)
    names = _names_to_list(data)
    ascii_copy = path.with_name("data_ascii.yaml")
    return DataYamlInspection(
        path=path,
        names=names,
        has_non_ascii=any(_contains_non_ascii(name) for name in names),
        ascii_copy_path=ascii_copy,
        ascii_copy_exists=ascii_copy.exists(),
    )


def generate_ascii_data_yaml(data_yaml_path: str | Path, target_path: str | Path | None = None) -> Path:
    source = Path(data_yaml_path)
    data = _load_yaml(source)
    original_names = data.get("names")
    class_count = _class_count(data)
    if class_count <= 0:
        raise DataYamlError("data.yaml 缺少 names 或 nc，无法生成英文类别名副本。")

    ascii_names = [f"class_{index}" for index in range(class_count)]
    if isinstance(original_names, dict):
        keys = list(original_names.keys())
        if len(keys) != class_count:
            keys = list(range(class_count))
        data["names"] = {key: ascii_names[index] for index, key in enumerate(keys)}
    else:
        data["names"] = ascii_names

    target = Path(target_path) if target_path else source.with_name("data_ascii.yaml")
    target.write_text(yaml.safe_dump(data, allow_unicode=False, sort_keys=False), encoding="utf-8")
    return target


def classify_backend_error(output_text: str) -> str:
    text = output_text.lower()
    if "arial.unicode.ttf" in text or "ultralytics.com/assets/arial.unicode.ttf" in text:
        return "font_download_failed"
    return "backend_failed"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DataYamlError(f"data.yaml 不存在：{path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    except Exception as exc:
        raise DataYamlError(f"读取 data.yaml 失败：{exc}") from exc
    if not isinstance(loaded, dict):
        raise DataYamlError("data.yaml 内容不是有效的 YAML 字典。")
    return loaded


def _names_to_list(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [str(value) for _, value in sorted(names.items(), key=lambda item: _name_key(item[0]))]
    if isinstance(names, list):
        return [str(value) for value in names]
    nc = data.get("nc")
    if isinstance(nc, int) and nc > 0:
        return [f"class_{index}" for index in range(nc)]
    return []


def _class_count(data: dict[str, Any]) -> int:
    names = data.get("names")
    if isinstance(names, dict | list):
        return len(names)
    nc = data.get("nc")
    return int(nc) if isinstance(nc, int) else 0


def _contains_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def _name_key(key: Any) -> tuple[int, str]:
    try:
        return (0, f"{int(key):08d}")
    except Exception:
        return (1, str(key))
