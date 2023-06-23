import threading
import uuid
from datetime import datetime
from functools import wraps
from typing import Any

from apscheduler.jobstores.base import JobLookupError
from apscheduler.triggers.cron import CronTrigger
from flask import request, jsonify, Blueprint, Response, send_file

from multi_modal_edge_ai.server.scheduler.jobs import open_federated_server_job, open_personalization_job, \
    get_current_workload


class DashboardBlueprint(Blueprint):
    """
    Class created for mypy to aknowledge the following variables are available in the blueprint
    """
    client_keeper: Any
    models_keeper: Any
    dashboard_token_path: Any
    federated_log_path: Any
    federated_server: Any
    scheduler: Any


# Create the blueprint object
dashboard_connection_blueprint = DashboardBlueprint('dashboard_connection', __name__)


def authenticate(func):
    @wraps(func)
    def decorated_function(*args, **kwargs) -> tuple[Response, int] | Any:
        dashboard_token_path = dashboard_connection_blueprint.dashboard_token_path
        file = open(dashboard_token_path, 'r')

        token = file.read().strip()

        request_token = request.headers.get('Authorization')

        # Check if the token is valid
        if request_token == token:  # Replace with your generated token
            return func(*args, **kwargs)
        else:
            return jsonify({'message': 'Unauthorized'}), 401

    return decorated_function


@dashboard_connection_blueprint.route('/dashboard/get_client_info', methods=['GET'])
@authenticate
def get_clients_info() -> Response:
    """
    This is the API called by the dashboard to access all the client info
    :return: a list of all the connected clients, where each client is represented as a dictionary
    the clients have the following fields: ip, status, last_seen, num_adls, num_anomalies.
    """
    client_keeper = dashboard_connection_blueprint.client_keeper

    client_keeper.update_clients_statuses()

    clients = client_keeper.connected_clients
    return jsonify({'connected_clients': clients})


@dashboard_connection_blueprint.route('/dashboard/schedule_federation_workload', methods=['POST'])
@authenticate
def schedule_federation_workload() -> tuple[Response, int]:
    """
    This function will schedule a federated learning workload according to the scheduling type and config_dict provided
    In case the type is "one-time", the job is schedule for a one time execution at the provided "date". In case the
    type is "recurrent", a CronTrigger is created by parsing the crontab in the "crontab" field. Finally, in the case
    "immediate" is the provided scheduling type, the job is done immediately.
    In case any of the parameters are missing, or the date/crontab is in a wrong format, a 400 error coded will be
    returned
    :return: If 200, the job_id of the job scheduled. If 400, the reason responsible to this being a bad request
    """
    federated_log_path = dashboard_connection_blueprint.federated_log_path
    scheduler = dashboard_connection_blueprint.scheduler
    client_keeper = dashboard_connection_blueprint.client_keeper
    federated_server = dashboard_connection_blueprint.federated_server

    data = request.get_json()

    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    config_dict = data.get('config')
    schedule_type = data.get("schedule_type")

    if not schedule_type:
        return jsonify({'error': 'Missing schedule_type'}), 400

    if not config_dict:
        return jsonify({'error': 'Missing config'}), 400

    run_date = None
    job_id = str(uuid.uuid4())

    try:
        if schedule_type == "immediate":
            thread = threading.Thread(target=open_federated_server_job,
                                      args=(federated_server, client_keeper, config_dict, federated_log_path))
            thread.start()

        elif schedule_type in ["recurrent", "one-time"]:
            if schedule_type == "recurrent":
                cron_str = data.get("crontab")
                if not cron_str:
                    return jsonify({'error': 'Missing crontab'}), 400
                trigger = CronTrigger.from_crontab(cron_str)

            else:  # "one-time"
                date_str = data.get("date")
                if not date_str:
                    return jsonify({'error': 'Missing date'}), 400

                try:
                    job_date = datetime.fromisoformat(date_str[:-1])
                except ValueError:
                    return jsonify({'error': 'Invalid date format'}), 400
                trigger = "date"
                run_date = job_date

            scheduler.add_job(open_federated_server_job, trigger,
                              args=[federated_server, client_keeper, config_dict, federated_log_path], id=job_id,
                              run_date=run_date)

        else:
            return jsonify({'error': f'Invalid schedule_type: {schedule_type}'}), 400

    except Exception as e:
        return jsonify({'error': f'Error scheduling job: {str(e)}'}), 500

    return jsonify({'job_id': job_id}), 200


@dashboard_connection_blueprint.route('/dashboard/schedule_personalization_workload', methods=['POST'])
@authenticate
def schedule_personalization_workload():
    """
    This function will schedule a personalization workload according to the scheduling type and config_dict provided
    In case the type is "one-time", the job is schedule for a one time execution at the provided "date". In case the
    type is "recurrent", a CronTrigger is created by parsing the crontab in the "crontab" field. Finally, in the case
    "immediate" is the provided scheduling type, the job is done immediately.
    In case any of the parameters are missing, or the date/crontab is in a wrong format, a 400 error coded will be
    returned
    :return: If 200, the job_id of the job scheduled. If 400, the reason responsible to this being a bad request
    """
    federated_log_path = dashboard_connection_blueprint.federated_log_path
    scheduler = dashboard_connection_blueprint.scheduler
    client_keeper = dashboard_connection_blueprint.client_keeper
    federated_server = dashboard_connection_blueprint.federated_server

    data = request.get_json()

    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    config_dict = data.get('config')
    schedule_type = data.get("schedule_type")

    if not schedule_type:
        return jsonify({'error': 'Missing schedule_type'}), 400

    if not config_dict:
        return jsonify({'error': 'Missing config'}), 400

    run_date = None
    job_id = str(uuid.uuid4())

    try:
        if schedule_type == "immediate":
            thread = threading.Thread(target=open_personalization_job,
                                      args=(federated_server, client_keeper, config_dict, federated_log_path))
            thread.start()

        elif schedule_type in ["recurrent", "one-time"]:
            if schedule_type == "recurrent":
                cron_str = data.get("crontab")
                if not cron_str:
                    return jsonify({'error': 'Missing crontab'}), 400
                trigger = CronTrigger.from_crontab(cron_str)

            else:  # "one-time"
                date_str = data.get("date")
                if not date_str:
                    return jsonify({'error': 'Missing date'}), 400

                try:
                    job_date = datetime.fromisoformat(date_str[:-1])
                except ValueError:
                    return jsonify({'error': 'Invalid date format'}), 400
                trigger = "date"
                run_date = job_date

            scheduler.add_job(open_personalization_job, trigger,
                              args=[federated_server, client_keeper, config_dict, federated_log_path], id=job_id,
                              run_date=run_date)

        else:
            return jsonify({'error': f'Invalid schedule_type: {schedule_type}'}), 400

    except Exception as e:
        return jsonify({'error': f'Error scheduling job: {str(e)}'}), 500

    return jsonify({'job_id': job_id}), 200


@dashboard_connection_blueprint.route('/dashboard/is_workload_running', methods=['GET'])
@authenticate
def is_workload_running() -> tuple[Response, int]:
    """
    This function will return the config file of the current workload being run, if any.
    :return: The config file, or a message saying that there are no configs being run
    """
    config = get_current_workload()
    if config:
        return jsonify(config), 200
    else:
        return jsonify({"message": "There are no workloads currently being ran"}), 200


@dashboard_connection_blueprint.route('/dashboard/fetch_all_federation_workloads', methods=['GET'])
@authenticate
def fetch_all_workloads() -> tuple[Response, int]:
    """
    This function will return all the federated learning and personalization workloads currently scheduled.
    All jobs will include id, scheduled_time, the config_dict, and a flag representing if it is a cron job, and the
    crontab, which will be the respective crontab if it is a cron job, or empty otherwise. It will also contain a
    field workload_type representing if it is a federation workload or personalization workload
    :return: A list with a dict representing each job. This dict has: id, scheduled_time, and config
    """
    scheduler = dashboard_connection_blueprint.scheduler

    workloads = [job for job in scheduler.get_jobs() if
                 (job.func == open_federated_server_job or job.func == open_personalization_job)]

    workloads.sort(key=lambda job: job.next_run_time)

    workloads_info = []
    for job in workloads:
        trigger = job.trigger
        if isinstance(trigger, CronTrigger):
            crontab_field_names = ['minute', 'hour', 'day', 'month', 'day_of_week']
            fields_dict = {field.name: str(field) for field in trigger.fields}
            crontab = ' '.join(fields_dict[field] for field in crontab_field_names)
        else:
            crontab = ""

        workloads_info.append({
            "id": job.id,
            'scheduled_time': str(job.next_run_time),
            "config": job.args[2],
            "cron_job": isinstance(trigger, CronTrigger),
            "crontab": crontab,
            "workload_type": "federation" if job.func == open_federated_server_job else "personalization"
        })

    return jsonify(workloads_info), 200


@dashboard_connection_blueprint.route('/dashboard/remove_federation_workload', methods=['DELETE'])
@authenticate
def remove_federation_workload() -> tuple[Response, int]:
    """
    This function will remove a specific federated learning workload, given its id.
    :return: If 200, the job_id of the removed federated learning workload. If 410, the id of the job that didn't exist,
    and thus couldn't be removed.
    """
    scheduler = dashboard_connection_blueprint.scheduler

    data = request.get_json()
    job_id = data.get("job_id")

    try:
        scheduler.remove_job(job_id)
        return jsonify({'job_id': job_id}), 200
    except JobLookupError:
        return jsonify({'job_id': job_id}), 410


@dashboard_connection_blueprint.route('/dashboard/get_error_log', methods=['GET'])
@authenticate
def get_error_log() -> Response:
    log_path = './app.log'
    return send_file(log_path)


@dashboard_connection_blueprint.route('/dashboard/get_federation_log', methods=['GET'])
@authenticate
def get_federation_log() -> Response:
    log_path = './multi_modal_edge_ai/server/federated_learning/server_log'
    return send_file(log_path)
