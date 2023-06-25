import logging
import threading
from typing import Any

from flwr.common import Scalar

open_federated_server_lock = threading.Lock()

current_workload: dict[str, Any] = {}


def open_federated_server_job(federated_server, client_keeper, config: dict[str, Scalar], log_file_path: str) -> None:
    """
    This function will open the federated server. It will acquire a lock in the meantime that will only be released
    when the federated workload is finished
    :param federated_server: The federated server object to open
    :param client_keeper: The client keeper with the models
    :param config: The config with which run the federated workload
    :param log_file_path: The path of the log
    :return:
    """
    if open_federated_server_lock.acquire(False):
        try:
            client_keeper.set_start_workload("start_federation", True)
            current_workload["config"] = config
            current_workload["workload_type"] = "federation"
            federated_server.start_server(config, log_file_path, True)
        finally:
            current_workload.clear()
            client_keeper.set_start_workload("start_federation", False)
            open_federated_server_lock.release()
    else:
        logging.error(
            "This federated learning workload cannot start, there is another workload currently being executed!")


def open_personalization_job(federated_server, client_keeper, config: dict[str, Scalar], log_file_path: str):
    """
    This function will open the federated server. It will acquire a lock in the meantime that will only be released
    when the federated workload is finished
    :param federated_server: The federated server object to open
    :param client_keeper: The client keeper with the models
    :param config: The config with which run the federated workload
    :param log_file_path: The path of the log
    :return:
    """
    if open_federated_server_lock.acquire(False):
        try:
            client_keeper.set_start_workload("start_personalization", True)
            current_workload["config"] = config
            current_workload["workload_type"] = "personalization"
            federated_server.start_server(config, log_file_path, False)
        finally:
            current_workload.clear()
            client_keeper.set_start_workload("start_personalization", False)
            open_federated_server_lock.release()
    else:
        logging.error("This personalization workload cannot start, there is another one currently being executed!")


def reset_all_daily_information_job(client_keeper) -> None:
    """
    This function will reset all the daily information of the server to 0.
    :param client_keeper: The client keeper with the daily information
    :return:
    """
    client_keeper.reset_all_daily_information()


def get_current_workload() -> dict[str, Scalar]:
    """
    This function will return the config with which the workloads are running.
    :return: If a workload is running, it will return the config with which it is running, else empty dictionary
    """
    return current_workload
