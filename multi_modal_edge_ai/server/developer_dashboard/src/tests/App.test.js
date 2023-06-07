import { render } from '@testing-library/react';
import App from '../App';
import Dashboard from "../App";

describe('Sorting', () => {
    test('should sort clients by IP address', () => {
        // Define your sample data
        const clients = [
            { id: 1, ip: '19.0.0.2', connectionStatus: 'Connected', lastUpdate: '2023-05-28 10:31:45' },
            { id: 2, ip: '19.0.0.10', connectionStatus: 'Disconnected', lastUpdate: '2023-05-28 10:25:12' },
            { id: 3, ip: '19.0.0.1', connectionStatus: 'Connected', lastUpdate: '2023-05-28 15:25:37'}
        ];

        // Render the component
        render(<App />);

        // Create a dashboard instance
        const dashboard = new Dashboard();

        // Sort the clients
        const sortedClients = dashboard.sortByIPAddress(clients, 'ip', 'asc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual([
            { id: 3, ip: '19.0.0.1', connectionStatus: 'Connected', lastUpdate: '2023-05-28 15:25:37' },
            { id: 1, ip: '19.0.0.2', connectionStatus: 'Connected', lastUpdate: '2023-05-28 10:31:45' },
            { id: 2, ip: '19.0.0.10', connectionStatus: 'Disconnected', lastUpdate: '2023-05-28 10:25:12' }
        ]);
    });

    test('should sort clients alphabetically', () => {
        // Define your sample data
        const clients = [
            { id: 1, ip: '19.0.0.2', connectionStatus: 'Connected', lastUpdate: '2023-05-28 10:31:45' },
            { id: 2, ip: '19.0.0.10', connectionStatus: 'Disconnected', lastUpdate: '2023-05-28 10:25:12' },
            { id: 3, ip: '19.0.0.1', connectionStatus: 'Connected', lastUpdate: '2023-05-28 15:25:37'}
        ];

        // Render the component
        render(<App />);

        // Create a dashboard instance
        const dashboard = new Dashboard();

        // Sort the clients
        const sortedClients = dashboard.sortAlphabetically(clients, 'connectionStatus', 'asc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual([
            { id: 1, ip: '19.0.0.2', connectionStatus: 'Connected', lastUpdate: '2023-05-28 10:31:45' },
            { id: 3, ip: '19.0.0.1', connectionStatus: 'Connected', lastUpdate: '2023-05-28 15:25:37' },
            { id: 2, ip: '19.0.0.10', connectionStatus: 'Disconnected', lastUpdate: '2023-05-28 10:25:12' }
        ]);
    });

    test('should sort clients by datetime', () => {
        // Define your sample data
        const clients = [
            { id: 1, ip: '19.0.0.2', connectionStatus: 'Connected', lastUpdate: '2023-05-28 10:31:45' },
            { id: 2, ip: '19.0.0.10', connectionStatus: 'Disconnected', lastUpdate: '2023-05-28 10:25:12' },
            { id: 3, ip: '19.0.0.1', connectionStatus: 'Connected', lastUpdate: '2023-05-28 15:25:37'}
        ];

        // Render the component
        render(<App />);

        // Create a dashboard instance
        const dashboard = new Dashboard();

        // Sort the clients
        const sortedClients = dashboard.sortByDateTime(clients, 'lastUpdate', 'asc');

        // Assert that the sorting is correct
        expect(sortedClients).toEqual([
            { id: 2, ip: '19.0.0.10', connectionStatus: 'Disconnected', lastUpdate: '2023-05-28 10:25:12' },
            { id: 1, ip: '19.0.0.2', connectionStatus: 'Connected', lastUpdate: '2023-05-28 10:31:45' },
            { id: 3, ip: '19.0.0.1', connectionStatus: 'Connected', lastUpdate: '2023-05-28 15:25:37' }
        ]);
    });
});