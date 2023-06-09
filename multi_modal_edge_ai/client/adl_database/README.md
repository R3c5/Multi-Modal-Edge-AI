# ADL Database

This module incorporates all functionality required to interact with the MongoDB ADL database.

### ```adl_database.py```:

This file contains all function required to connect to the database and obtain the databases and the collections.

It contains 3 methods:
- ```get_database_client(username, password)```: This function creates and returns a MongoDB client with the specified username, password. Do not forget to make sure that the SSH Tunnel is running.
- ```get_database(client, database_name)```: This function retrieves and returns a specific database from the MongoDB client.
- ```get_collection(database, collection_name)```: This function retrieves and returns a specific collection from the MongoDB database. If the collection does not exist, it creates it.

### ```adl_queries.py```:

This file contains all functions required to query the database.

It contains the following methods:

- ```get_all_activities(collection: Collection)```: Gets all activities from the specified collection.
- ```add_activity(collection: Collection, start_time: pd.Timestamp, end_time: pd.Timestamp, activity: str)```: Adds an activity to the specified collection. If the last activity in the collection is the same as the one being added, the entries are concatenated.
- ```get_past_x_activities(collection: Collection, x: int)```: Gets the last x activities from the specified collection.
- ```get_past_x_minutes(collection: Collection, x: int, clip: bool)```: Gets all activities from the specified collection that have an end time that occurred in the last x minutes.
- ```delete_all_activities(collection: Collection)```: Deletes all activities from the specified collection.
- ```delete_last_x_activities(collection: Collection, x: int)```: Deletes the last x activities from the specified collection.

For more information about each method check the doc strings in the code.