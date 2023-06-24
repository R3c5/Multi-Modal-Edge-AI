# Server API Documentation

## Overview
The server is the central point of the system, and it keeps a general overview of all the connected clients as well as global models that will be used by the clients. To that extent the first 2 APIs are ways for the clients to communicate with the server, while the rest are intended for the dashboard to gather all the required information stored on the server.

## Base URL
We did not deploy the server, so you can run it locally and access all the apis on `http://127.0.0.1:5000`.

## Endpoints
---
## Client APIs
These APIs are available to all clients in the virtual network.

### 1. **Set Up Connection**

This is an API intended to be called when the client is first started. This will make the server store the new client and send a zip containing the global models for *ADL Inference* and *Anomaly Detection* (which are stored as pickle files on the server).

- URL: `/api/set_up_connection`
- Method: `GET`
- Response:
    - Status Code: `200 OK` if successful.
    - Body:
        > The response contains a zip file called `model_zip` containing the `adl_model` and `anomaly_detection_model` files. The client should handle the zip file accordingly, extracting the necessary files for further usage.
        
    - Status Code: `500 Internal Server Error` if the server is not able to find the IP of the client.
    - Body:
        ```json
        {
        "message": "Error occurred setting up the connection"
        }
        ```
- Example Request:
    ```http
    GET /api/set_up_connection HTTP/1.1
    Host: 127.0.0.1:5000
    ```


### 2. **Heartbeat**

This is an API intended to be called by the client regularly. The server updates the last seen field of the client, as well as the number of adls and anomalies detected with the amount specified in the request. As a response, the server sends a zip file containing the models that were updated since the last request from the client. Note that the zip file can be empty, contain one of the `adl_model` and `anomaly_detection_model` or both. Also, the request can contain a header `start_federation_client_flag` that notifies the client to start the federation client

- URL: `/api/heartbeat`
- Method: `POST`
- Request Body:
  The request body should be a JSON object with the following properties:
    - `recent_adls` (integer): The number of ADLs that were predicted by the client since the last heartbeat / connection
    - `recent_anomalies` (integer): The number of anomalies detected since the last heartbeat / connection
- Response:
    - Status Code: `200 OK` if successful.
    - Body:
        > The response contains a zip file called `model_zip`. The zip can contain none, either the `adl_model` or the `anomaly_detection_model` or both, depending on if the model was updated since the client last sent a request to the server. The client should handle the zip file accordingly, extracting the necessary files for further usage. The response also contains a header `start_federation_client_flag` that is a boolean representing if the client should or should not start the federation client.
    - Status Code: `404 Not Found` if the server can not find the client throughout the stored list
    - Body:
        ```json
        {
        "message": "Client not found"
        }
        ```
    - Status Code: `400 Bad Request Error` if the request does not contain the required `recent_adls` and `recent_anomalies` in the body.
    - Body:
        ```json
        {
        "message": "Invalid JSON payload"
        }
        ```
    - Status Code: `500 Internal Server Error` if the server is not able to find the IP of the client.
    - Body:
        ```json
        {
        "message": "Error occurred setting up the connection"
        }
        ```
- Example Request:
    ```http
    POST /api/heartbeat HTTP/1.1
    Host: 127.0.0.1:5000
    Content-Type: application/json
    Content-Length: 50

    {
    "recent_adls": 10,
    "recent_anomalies": 2
    }
    ```

---
## Dashboard APIs
These APIs are only intended for communication between the dashboard and the server. This was done because of the two being written in different languages. In order for the clients to not be able to access these APIs are protected using a token. The token resides in a file located on the server.
The token must be included as an `Authorization` header in the request.

### 1. **Get Clients Info**

This API is called by the dashboard to access all the client information.

- URL: `/dashboard/get_client_info`
- Method: `GET`
- Response:
    - Status Code: `200 OK` if successful.
    - Body:
    > A list of all the connected clients, where each client is represented as a dictionary. The clients have the following fields: `ip`, `status`, `last_seen`, `last_model_aggregation`, `start_federation`, `num_adls`, and `num_anomalies`.
- Example Request:
    ```http
    GET /dashboard/get_client_info HTTP/1.1
    Host: 127.0.0.1:5000
    Authorization: super_secure_token_here_123
    ```

### 2. **Schedule Federation Workload**

This API schedules a federated learning workload according to the provided scheduling type and configuration.

- URL: `/dashboard/schedule_federation_workload`
- Method: `POST`
- Request Body:
    The request body should be a JSON object with the following properties:
    - `config (object)`: The configuration dictionary for the federated learning workload.
    - `schedule_type (string)`: The type of scheduling for the workload (one-time, recurrent, or immediate).
    - `crontab (string, required for recurrent scheduling type)`: The cron expression for recurrent scheduling.
    - `date (string, required for one-time scheduling type)`: The date and time for one-time scheduling.
- Response:
    - Status Code: `200 OK` if successful.
    - Body: 
       ```json
       {
            "job_id": "Job ID"
       }
       ```
    - Status Code: `400 Bad Request` if any parameters are missing or if the date/crontab format is incorrect.
    - Body: 
        ```json
        {
            "error": "Error reason"
        }
        ```
- Example Request:
    ```http
    POST /dashboard/schedule_federation_workload HTTP/1.1
    Host: 127.0.0.1:5000
    Authorization: super_secure_token_here_123
    Content-Type: application/json
    Content-Length: 144

    {
    "config": {
        "num_rounds": "5",
        "min_fit_clients": "10"
    },
    "schedule_type": "one-time",
    "date": "2023-06-30T12:00:00Z"
    }
    ```

### 3. **Is Federation Workload Running**
This API returns the configuration file of the currently running federated learning workload, if any.

- URL: `/dashboard/is_federation_workload_running`
- Method: `GET`
- Response:
    - Status Code: `200 OK`
    - Body: Depending on if the federated workload is running:
        > Returns the configuration file of the current federated learning workload.

        or
    
        ```json
        {
            "message": "There are no federated workloads currently being ran"
        }
        ```
- Example Request:
    ```http
    GET /dashboard/is_federation_workload_running HTTP/1.1
    Host: 127.0.0.1:5000
    Authorization: super_secure_token_here_123
    ```

### 4. **Fetch All Federation Workloads**
This API returns all the federated learning workloads currently scheduled.

- URL: `/dashboard/fetch_all_federation_workloads`
- Method: `GET`
- Response:
    - Status Code: `200 OK`
    - Body: Returns a list of dictionaries, where each dictionary represents a scheduled job. A job looks like:
        ```json
        {
            "id": "Job ID",
            "scheduled_time": "2023-06-30 09:00:00",
            "config": "Config",
            "cron_job": true,
            "crontab": "0 9 * * *"
        }
        ```
- Example Request:
    ```http
    GET /dashboard/fetch_all_federation_workloads HTTP/1.1
    Host: 127.0.0.1:5000
    Authorization: super_secure_token_here_123
    ```


### 5. **Remove Federation Workload**
This API removes a specific federated learning workload based on its ID.

- URL: `/dashboard/remove_federation_workload`
- Method: `DELETE`
- Request Body:
    The request body should be a JSON object with the following property:
    - `job_id (string)`: The ID of the federated learning workload to remove.
- Response:
    - Status Code: `200 OK` if successful
    - Status Code: `410 Gone` if specified ID does not exist
    - Body: 
        ```json
        {
            "job_id": "Job ID"
        }
        ```
- Example Request:
    ```http
    DELETE /dashboard/remove_federation_workload HTTP/1.1
    Host: 127.0.0.1:5000
    Authorization: super_secure_token_here_123
    Content-Type: application/json
    Content-Length: 31

    {
        "job_id": "123456789"
    }
    ```

### 6. **Get Error Log**
This API retrieves the error log file.

- URL: `/dashboard/get_error_log`
- Method: `GET`
- Response:
    - Status Code: `200 OK`
    - Body: 
    > Returns the error log file
- Example Request:
    ```http
    GET /dashboard/get_error_log HTTP/1.1
    Host: 127.0.0.1:5000
    Authorization: super_secure_token_here_123
    ```

### 7. **Get Federation Log**
This function retrieves the federation log file.

- URL: `/dashboard/get_federation_log`
- Method: `GET`
- Response:
    - Status Code: `200 OK`
    - Body: 
    > Returns the federation log file
- Example Request:
    ```http
    GET /dashboard/get_federation_log HTTP/1.1
    Host: 127.0.0.1:5000
    Authorization: super_secure_token_here_123
    ```