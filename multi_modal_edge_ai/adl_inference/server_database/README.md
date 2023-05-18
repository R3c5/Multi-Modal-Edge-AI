# Database Readme

 ## Setup Instructions:

 Before running the script, make sure that the SSH tunnel is running:\
    # 1. Open WSL\
    # 2. Run the following command: ```ssh ***REMOVED*** -N -L 27017:localhost:27018```\
    # 3. Input this password: ***REMOVED***\
    # 4. Leave the WSL window open and running in the background
 
 ## DatabaseTunnel:
 
This class is initialised with the database name you want to connect to. Currently, we only have 4 available:
 - sensor_data_AB123456789C
 - sensor_data_AB123456789D
 - sensor_data_00B31ECE4385
 - sensor_data_1234567ABC89

After initialising an instance of this class you have access to the following methods:

- ```get_all_documents_all_fields()``` - returns all sensor entries in the database
- ```get_all_documents``` - returns all sensor entries in the database(except buttons as they do not provide any actual data. They can be queried separately.), but only the relevant fields for each type of sensor(described bellow)
- ```(static)get_data_from_cursor``` - a cursor is an object that contains the results of a database query. This method transform the last_seen field which holds the number of milliseconds from 1st of January 1970 to 2 fields: date and time and returns a list of dictionaries, where each dictionary is a sensor entry. It also adds the sensor type that it receives in the call to each entry(default is undefined). Method is meant as an auxiliary method for the methods.
- ```get_contact_sensors``` - returns all contact sensors entries in the database with their relevant fields(described bellow)
- ```get_pir_sensors``` - returns all pir sensors entries in the database with their relevant fields(described bellow)
- ```get_button_sensors``` - returns all button sensors entries in the database with their relevant fields(described bellow)
- ```get_power_sensors``` - returns all power sensors entries in the database with their relevant fields(described bellow)
- ```plot_distribution_week_days``` - plots the distribution of the sensor entries on each day of the week
- ```plot_distribution_hourly``` - plots the distribution of the sensor entries on each hour of the day
- ```plot_distributions_for_all_entries``` - plots the two distributions for all sensor entries(except buttons because they do not provide any actual data. They can be queried separately.)
- ```plot_distributions_for_power_sensor_entires``` - plots the two distributions for all power sensor entries
- ```plot_distributions_for_contact_sensor_entires``` - plots the two distributions for all contact sensor entries
- ```plot_distributions_for_pir_sensor_entires``` - plots the two distributions for all pir sensor entries
- ```plot_distributions_for_button_sensor_entires``` - plots the two distributions for all button sensor entries

In the same file, the following preprocessing methods are also available:
- ```is_time_difference_smaller_than_x_seconds``` - checks if the time difference between two entries is smaller or **equal** than a given number of seconds. If the number of seconds is negative it will return True. 
- ```preprocess_data_to_start_and_end_times``` takes data from a list of dictionaries and returns a list of dictionaries, where identical signals are aggregated to form a new signal with a start and end time.
- ```group_sensors_on_friendly_names_and_preprocess```groups on instances of the same sensor(only use on power and contact sensors) and applies the preprocessing function on each group.

### Relevant fields for each type of sensor:
- PIR sensor: ```{`occupancy`, `last_seen`, `device.friendlyName`}```
- Power sensor: ```{`state`, `last_seen`, `device.friendlyName`}```
- Contact sensor: ```{`contact`, `last_seen`, `device.friendlyName`}```
- For the sensors mentioned above, the query functions methods explained above will also apply a preprocessing function.
- Button sensor: ```{`last_seen`, `device.friendlyName`}```

## database-calls

Script that contains the code for the database calls and plots. It is meant to be used as an aid in the development process.