import logging
import threading

from multi_modal_edge_ai.server.main import federated_server, client_keeper

open_federated_server_lock = threading.Lock()

current_federated_workload = {}


def open_federated_server(config, log_file_path):
    if open_federated_server_lock.acquire(False):
        try:
            federated_server_thread = threading.Thread(target=federated_server.start_server,
                                                       args=(config, log_file_path))
            federated_server_thread.start()
            client_keeper.set_start_federation(True)
            current_federated_workload["config"] = config
        finally:
            current_federated_workload.clear()
            client_keeper.set_start_federation(False)
            open_federated_server_lock.release()
    else:
        logging.error("This federated learning workload cannot start, there is another one currently being executed!")


def is_federated_workload_running():
    return current_federated_workload["config"] if current_federated_workload else {}
