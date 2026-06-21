from __future__ import annotations

from typing import Any


class RemoteExperimentClient:
    """预留给未来远程实验部署的客户端边界。"""

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = base_url
        self.token = token

    def submit_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("远程实验提交尚未实现。")

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        raise NotImplementedError("远程任务状态轮询尚未实现。")

    def get_logs(self, task_id: str, offset: int = 0) -> dict[str, Any]:
        raise NotImplementedError("远程日志流尚未实现。")

    def download_results(self, task_id: str, destination: str) -> dict[str, Any]:
        raise NotImplementedError("远程结果下载尚未实现。")
