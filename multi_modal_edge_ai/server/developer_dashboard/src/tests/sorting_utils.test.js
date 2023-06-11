import SortingUtils from "../utils/sorting_utils";

const clients =
    [
        {'ip':'19.0.0.3', 'last_seen':'2023-01-02\n12:00:00', 'num_adls':4, 'num_anomalies':3, 'status':'Connected'},
        {'ip':'19.0.0.4', 'last_seen':'2023-01-02\n13:00:00', 'num_adls':3, 'num_anomalies':5, 'status':'Disconnected'},
        {'ip':'19.0.0.1', 'last_seen':'2023-01-01\n14:00:00', 'num_adls':1, 'num_anomalies':2, 'status':'Connected'},
    ]


describe('Sorting', () => {
    test('should sort clients by IP address ascending', () => {
        // Sort the clients
        const sortedClients = SortingUtils.sortByIPAddress(clients, 'ip', 'asc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual(    [
            {'ip':'19.0.0.1', 'last_seen':'2023-01-01\n14:00:00', 'num_adls':1, 'num_anomalies':2, 'status':'Connected'},
            {'ip':'19.0.0.3', 'last_seen':'2023-01-02\n12:00:00', 'num_adls':4, 'num_anomalies':3, 'status':'Connected'},
            {'ip':'19.0.0.4', 'last_seen':'2023-01-02\n13:00:00', 'num_adls':3, 'num_anomalies':5, 'status':'Disconnected'}
        ]);
    });
    test('should sort clients by IP address descending', () => {
        // Sort the clients
        const sortedClients = SortingUtils.sortByIPAddress(clients, 'ip', 'desc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual(    [
            {'ip':'19.0.0.4', 'last_seen':'2023-01-02\n13:00:00', 'num_adls':3, 'num_anomalies':5, 'status':'Disconnected'},
            {'ip':'19.0.0.3', 'last_seen':'2023-01-02\n12:00:00', 'num_adls':4, 'num_anomalies':3, 'status':'Connected'},
            {'ip':'19.0.0.1', 'last_seen':'2023-01-01\n14:00:00', 'num_adls':1, 'num_anomalies':2, 'status':'Connected'}
        ]);
    });
    test('should sort clients alphabetically ascending', () => {
        // Sort the clients
        const sortedClients = SortingUtils.sortAlphabetically(clients, 'status', 'asc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual(    [
            {'ip':'19.0.0.3', 'last_seen':'2023-01-02\n12:00:00', 'num_adls':4, 'num_anomalies':3, 'status':'Connected'},
            {'ip':'19.0.0.1', 'last_seen':'2023-01-01\n14:00:00', 'num_adls':1, 'num_anomalies':2, 'status':'Connected'},
            {'ip':'19.0.0.4', 'last_seen':'2023-01-02\n13:00:00', 'num_adls':3, 'num_anomalies':5, 'status':'Disconnected'}
        ]);
    });
    test('should sort clients alphabetically descending', () => {
        // Sort the clients
        const sortedClients = SortingUtils.sortAlphabetically(clients, 'status', 'desc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual(    [
            {'ip':'19.0.0.4', 'last_seen':'2023-01-02\n13:00:00', 'num_adls':3, 'num_anomalies':5, 'status':'Disconnected'},
            {'ip':'19.0.0.3', 'last_seen':'2023-01-02\n12:00:00', 'num_adls':4, 'num_anomalies':3, 'status':'Connected'},
            {'ip':'19.0.0.1', 'last_seen':'2023-01-01\n14:00:00', 'num_adls':1, 'num_anomalies':2, 'status':'Connected'}
        ]);
    });
    test('should sort clients by datetime ascending', () => {
        // Sort the clients
        const sortedClients = SortingUtils.sortByDateTime(clients, 'last_seen', 'asc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual(    [
            {'ip':'19.0.0.1', 'last_seen':'2023-01-01\n14:00:00', 'num_adls':1, 'num_anomalies':2, 'status':'Connected'},
            {'ip':'19.0.0.3', 'last_seen':'2023-01-02\n12:00:00', 'num_adls':4, 'num_anomalies':3, 'status':'Connected'},
            {'ip':'19.0.0.4', 'last_seen':'2023-01-02\n13:00:00', 'num_adls':3, 'num_anomalies':5, 'status':'Disconnected'}
        ]);
    });
    test('should sort clients by datetime descending', () => {
        // Sort the clients
        const sortedClients = SortingUtils.sortByDateTime(clients, 'last_seen', 'desc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual(    [
            {'ip':'19.0.0.4', 'last_seen':'2023-01-02\n13:00:00', 'num_adls':3, 'num_anomalies':5, 'status':'Disconnected'},
            {'ip':'19.0.0.3', 'last_seen':'2023-01-02\n12:00:00', 'num_adls':4, 'num_anomalies':3, 'status':'Connected'},
            {'ip':'19.0.0.1', 'last_seen':'2023-01-01\n14:00:00', 'num_adls':1, 'num_anomalies':2, 'status':'Connected'}
        ]);
    });
    test('should sort clients numerically ascending', () => {
        // Sort the clients
        const sortedClients = SortingUtils.sortNumerically(clients, 'num_adls', 'asc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual(    [
            {'ip':'19.0.0.1', 'last_seen':'2023-01-01\n14:00:00', 'num_adls':1, 'num_anomalies':2, 'status':'Connected'},
            {'ip':'19.0.0.4', 'last_seen':'2023-01-02\n13:00:00', 'num_adls':3, 'num_anomalies':5, 'status':'Disconnected'},
            {'ip':'19.0.0.3', 'last_seen':'2023-01-02\n12:00:00', 'num_adls':4, 'num_anomalies':3, 'status':'Connected'}
        ]);
    });
    test('should sort clients numerically descending', () => {
        // Sort the clients
        const sortedClients = SortingUtils.sortNumerically(clients, 'num_anomalies', 'desc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual(    [
            {'ip':'19.0.0.4', 'last_seen':'2023-01-02\n13:00:00', 'num_adls':3, 'num_anomalies':5, 'status':'Disconnected'},
            {'ip':'19.0.0.3', 'last_seen':'2023-01-02\n12:00:00', 'num_adls':4, 'num_anomalies':3, 'status':'Connected'},
            {'ip':'19.0.0.1', 'last_seen':'2023-01-01\n14:00:00', 'num_adls':1, 'num_anomalies':2, 'status':'Connected'}
        ]);
    });
    test('should sort empty clients', () => {
        // Declare variables
        const clientsEmpty  = []
        // Sort the clients
        const sortedClients = SortingUtils.sortNumerically(clientsEmpty, 'num_anomalies', 'desc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual( []);
    });
});