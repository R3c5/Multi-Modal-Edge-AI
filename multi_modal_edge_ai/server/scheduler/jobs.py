import logging
import threading

open_federated_server_lock = threading.Lock()

current_federated_workload = {}


def open_federated_server_job(config, log_file_path):
    from multi_modal_edge_ai.server.main import federated_server, client_keeper

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


def reset_all_daily_information_job():
    from multi_modal_edge_ai.server.main import client_keeper
    client_keeper.reset_all_daily_information()


def is_federated_workload_running():
    return current_federated_workload["config"] if current_federated_workload else {}
