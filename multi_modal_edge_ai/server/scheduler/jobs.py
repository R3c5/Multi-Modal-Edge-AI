import logging
import threading
from typing import Any

from flwr.common import Scalar

open_federated_server_lock = threading.Lock()

current_federated_workload: dict[str, Any] = {}


def open_federated_server_job(federated_server, client_keeper, config: dict[str, Scalar], log_file_path: str) -> None:

    if open_federated_server_lock.acquire(False):
        try:
            client_keeper.set_start_federation(True)
            current_federated_workload["config"] = config
            federated_server.start_server(config, log_file_path)
        finally:
            current_federated_workload.clear()
            client_keeper.set_start_federation(False)
            open_federated_server_lock.release()
    else:
        logging.error("This federated learning workload cannot start, there is another one currently being executed!")


def reset_all_daily_information_job(client_keeper) -> None:
    client_keeper.reset_all_daily_information()


def is_federated_workload_running() -> dict[str, Scalar]:
    return current_federated_workload["config"] if current_federated_workload else {}
