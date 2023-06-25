# Databases

This module incorporates all functionality required to interact with all the MongoDB databases(ADL, Sensor and Anomaly databases).

It contains 5 files:

### ```database_connection.py```:

This file contains all function required to connect to the databases and obtain the databases and the collections. Before running, do not forget to set up the ssh tunnel.

It contains 3 methods:
- ```get_database_client(username, password)```: This function creates and returns a MongoDB client with the specified username, password. Do not forget to make sure that the SSH Tunnel is running.
- ```get_database(client, database_name)```: This function retrieves and returns a specific database from the MongoDB client.
- ```get_collection(database, collection_name)```: This function retrieves and returns a specific collection from the MongoDB database. If the collection does not exist, it creates it.

### ```sensor_queries.py```:

This file contains all functions required to query the sensor database.

- ```get_contact_sensors``` - returns all contact sensors entries in the database with their relevant fields(described bellow)
- ```get_pir_sensors``` - returns all pir sensors entries in the database with their relevant fields(described bellow)
- ```get_power_sensors``` - returns all power sensors entries in the database with their relevant fields(described bellow)
- ```get_all_documents``` - returns all sensor entries in the database(except buttons as they do not provide any actual data. They can be queried separately.), but only the relevant fields for each type of sensor(described bellow)
- ```get_all_documents_all_fields``` - returns all sensor entries in the database
- ```get_past_x_seconds_of_all_sensor_entries``` - returns the entries that have a start_time no longer than x seconds ago.

 Relevant fields for each type of sensor:
- PIR sensor: ```{`occupancy`, `last_seen`, `device.friendlyName`}```
- Power sensor: ```{`state`, `last_seen`, `device.friendlyName`}```
- Contact sensor: ```{`contact`, `last_seen`, `device.friendlyName`}```
- For the sensors mentioned above, the query functions methods explained above will also apply a preprocessing function.
- Button sensor: ```{`last_seen`, `device.friendlyName`}

### ```sensor_database_preprocessing_methods.py```:

This file contains auxiliary methods that are used in the ```sensor_queries.py``` file to preprocess the data before returning it.

- ```get_data_from_cursor``` - a cursor is an object that contains the results of a database query. This method transform the last_seen field which holds the number of milliseconds from 1st of January 1970 to 2 fields: date and time and returns a list of dictionaries, where each dictionary is a sensor entry. It also adds the sensor type that it receives in the call to each entry(default is undefined). Method is meant as an auxiliary method for the methods.
- ```is_time_difference_smaller_than_x_seconds``` - checks if the time difference between two entries is smaller or **equal** than a given number of seconds. If the number of seconds is negative it will return True. 
- ```aggregate_similar_entries``` takes data from a list of dictionaries and returns a list of dictionaries, where identical signals are aggregated to form a new signal with a start and end time.
- ```group_sensors_on_friendly_names_and_aggregate_entries```groups on instances of the same sensor(only use on power and contact sensors) and applies the ```aggregate_similar_entries``` function on each group. Returns a flattened list with the results.

### ```adl_queries.py```:

This file contains all functions required to query the ADL database.

It contains the following methods:

- ```get_all_activities(collection: Collection)```: Gets all activities from the specified collection.
- ```add_activity(collection: Collection, start_time: pd.Timestamp, end_time: pd.Timestamp, activity: str)```: Adds an activity to the specified collection. If the last activity in the collection is the same as the one being added, the entries are concatenated.
- ```get_past_x_activities(collection: Collection, x: int)```: Gets the last x activities from the specified collection.
- ```get_past_x_minutes(collection: Collection, x: int, clip: bool)```: Gets all activities from the specified collection that have an end time that occurred in the last x minutes.
- ```delete_all_activities(collection: Collection)```: Deletes all activities from the specified collection.
- ```delete_last_x_activities(collection: Collection, x: int)```: Deletes the last x activities from the specified collection.

### ```anomaly_queries.py```:

This file contains all functions required to query the anomaly database.

It contains the following methods:

- ```add_anomaly``` - adds an anomaly to the database
- ```delete_all_anomalies``` - deletes all anomalies from the database
- ```delete_past_x_anomalies``` - deletes the last x anomalies from the database
- ```get_all_anomalies``` - returns all anomalies from the database
- ```get_past_x_anomalies``` - returns the last x anomalies from the database
- ```get_past_x_minutes_anomalies``` - returns all anomalies from the database that have an end time that occurred in the last x minutes

#### For more information about each method check the doc strings in the code.