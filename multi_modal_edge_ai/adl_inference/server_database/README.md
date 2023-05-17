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
 - sensor_data_BC123456789A

After initialising an instance of this class you have access to the following methods:

- ```get_all_documents_all_fields()``` - returns all sensor entries in the database
- ```get_all_documents``` - returns all sensor entries in the database, but only the relevant fields for each type of sensor(described bellow)
- ```get_data_from_cursor``` - a cursor is an object that contains the results of a database query. This method transform the last_seen field which holds the number of milliseconds from 1st of January 1970 to 2 fields: date and time and returns a list of dictionaries, where each dictionary is a sensor entry. Method is meant as an auxiliary method for the methods.
- ```get_contact_sensors``` - returns all contact sensors entries in the database with their relevant fields(described bellow)
- ```get_pir_sensors``` - returns all pir sensors entries in the database with their relevant fields(described bellow)
- ```get_button_sensors``` - returns all button sensors entries in the database with their relevant fields(described bellow)
- ```get_power_sensors``` - returns all power sensors entries in the database with their relevant fields(described bellow)
- ```plot_distribution_week_days``` - plots the distribution of the sensor entries on each day of the week
- ```plot_distribution_hourly``` - plots the distribution of the sensor entries on each hour of the day
- ```plot_distributions_for_all_entries``` - plots the two distributions for all sensor entries
- ```plot_distributions_for_power_sensor_entires``` - plots the two distributions for all power sensor entries
- ```plot_distributions_for_contact_sensor_entires``` - plots the two distributions for all contact sensor entries
- ```plot_distributions_for_pir_sensor_entires``` - plots the two distributions for all pir sensor entries
- ```plot_distributions_for_button_sensor_entires``` - plots the two distributions for all button sensor entries

### Relevant fields for each type of sensor:
- Contact sensor: ```{`contact`, `last_seen`, `device.friendlyName`}```
- PIR sensor: ```{`occupancy`, `illuminance`, `motion_sensitivity`, `last_seen`, `device.friendlyName`}```
- Button sensor: ```{`last_seen`, `device.friendlyName`}```
- Power sensor: ```{`state`, `last_seen`, `device.friendlyName`}```

## database-calls

Script that contains the code for the database calls and plots. It is meant to be used as an aid in the development process.