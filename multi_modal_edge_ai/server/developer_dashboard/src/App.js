import './App.css';
import React from 'react';
import {Card, HTMLTable, Icon} from '@blueprintjs/core';
import 'normalize.css';
import "@blueprintjs/core/lib/css/blueprint.css";




const CircleCard = ({ title, value }) => (
    <Card style={{ width: '200px', margin: '10px' }}>
        <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '10px' }}>{title}</div>
            <div style={{ width: '125px', height: '125px', borderRadius: '50%', backgroundColor: '#e6e6e6', margin: '0 auto', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '24px', fontWeight: 'bold' }}>
                {value}
            </div>
        </div>
    </Card>
);
class Dashboard extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            numClients: 10,
            lastModelUpdate: '2023-05-28 10:30:00',
            clients: [
                { 'id': 1, 'ip': '192.168.0.1', 'connectionStatus': 'Connected', 'lastUpdate': '2023-05-28 10:31:45' },
                { 'id': 2, 'ip': '192.168.0.2', 'connectionStatus': 'Disconnected', 'lastUpdate': '2023-05-28 10:25:12' },
                { 'id': 3, 'ip': '192.168.0.3', 'connectionStatus': 'Connected', 'lastUpdate': '2023-05-28 15:25:37' },
                { 'id': 4, 'ip': '192.168.0.4', 'connectionStatus': 'Connected', 'lastUpdate': '2023-05-28 15:25:37' },
                { 'id': 5, 'ip': '192.168.0.5', 'connectionStatus': 'Disconnected', 'lastUpdate': '2023-05-28 15:25:37' },
                { 'id': 6, 'ip': '192.168.0.6', 'connectionStatus': 'Disconnected', 'lastUpdate': '2023-05-14 10:12:37' },
                { 'id': 7, 'ip': '192.168.0.7', 'connectionStatus': 'Connected', 'lastUpdate': '2023-05-14 10:12:37' },
                { 'id': 8, 'ip': '192.168.0.8', 'connectionStatus': 'Connected', 'lastUpdate': '2023-05-28 15:25:37' },
                { 'id': 9, 'ip': '192.168.0.9', 'connectionStatus': 'Disconnected', 'lastUpdate': '2023-05-28 15:25:37' },
                { 'id': 10, 'ip': '192.168.0.10', 'connectionStatus': 'Connected', 'lastUpdate': '2023-05-28 15:25:37' },
                { 'id': 11, 'ip': '192.168.0.11', 'connectionStatus': 'Disconnected', 'lastUpdate': '2023-04-27 13:25:50' },
                { 'id': 12, 'ip': '192.168.0.12', 'connectionStatus': 'Connected', 'lastUpdate': '2023-05-14 10:12:37' },
                { 'id': 13, 'ip': '192.168.0.13', 'connectionStatus': 'Connected', 'lastUpdate': '2023-05-28 15:25:37' }
            ],
            sortColumn: null,
            sortDirection: 'asc'
        };
    }

    /**
     * Handles sorting of the clients based on the specified column.
     * @param {string} column - The column to sort by.
     */
    handleSort(column) {
        const { clients, sortColumn, sortDirection } = this.state;
        const direction =
            sortColumn === column && sortDirection === 'asc' ? 'desc' : 'asc';

        let sortedClients;
        if (column === 'connectionStatus') {
            sortedClients = this.sortAlphabetically(clients, column, direction);
        } else if (column === 'lastUpdate') {
            sortedClients = this.sortByDateTime(clients, column, direction);
        } else if (column === 'ip') {
            sortedClients = this.sortByIPAddress(clients, column, direction);
        } else {
            // Default sorting if column not recognized
            sortedClients = clients;
        }

        this.setState({
            clients: sortedClients,
            sortColumn: column,
            sortDirection: direction,
        });
    }

    /**
     * Sorts the data array alphabetically based on the specified column and direction.
     * @param {Array} data - The array of data to be sorted.
     * @param {string} column - The column to sort by.
     * @param {string} direction - The sorting direction ('asc' for ascending, 'desc' for descending).
     * @returns {Array} The sorted array.
     */
    sortAlphabetically(data, column, direction) {
        return [...data].sort((a, b) =>
            direction === 'asc'
                ? a[column].localeCompare(b[column])
                : b[column].localeCompare(a[column])
        );
    }

    /**
     * Sorts the data array by date and time based on the specified column and direction.
     * @param {Array} data - The array of data to be sorted.
     * @param {string} column - The column to sort by.
     * @param {string} direction - The sorting direction ('asc' for ascending, 'desc' for descending).
     * @returns {Array} The sorted array.
     */
    sortByDateTime(data, column, direction) {
        return [...data].sort((a, b) => {
            const dateA = new Date(a[column]);
            const dateB = new Date(b[column]);
            return direction === 'asc' ? dateA - dateB : dateB - dateA;
        });
    }

    /**
     * Sorts the data array by IP address based on the specified column and direction.
     * @param {Array} data - The array of data to be sorted.
     * @param {string} column - The column to sort by.
     * @param {string} direction - The sorting direction ('asc' for ascending, 'desc' for descending).
     * @returns {Array} The sorted array.
     */
    sortByIPAddress(data, column, direction) {
        return [...data].sort((a, b) => {
            const ipA = a[column].split('.').map(Number);
            const ipB = b[column].split('.').map(Number);

            for (let i = 0; i < ipA.length; i++) {
                if (ipA[i] < ipB[i]) {
                    return direction === 'asc' ? -1 : 1;
                }
                if (ipA[i] > ipB[i]) {
                    return direction === 'asc' ? 1 : -1;
                }
            }

            return 0;
        });
    }


    render() {
        const { numClients, clients, sortColumn, sortDirection } = this.state;

        return (
            <>
            <div> <!-- This div deals with the top row of circles -->
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <CircleCard title="Number of clients" value={numClients} />
                    <CircleCard title="Last update" value={7 + " hours"} />
                    <CircleCard title="Next update in" value={10 + " hours"} />
                </div>
            </div>
            <div style={{ width: '100%'}}> <!-- This div deals with the table -->
                <h2>Clients</h2>
                <div style={{maxHeight: '50vh', overflow: 'auto', width: '100%'}}>
                    <HTMLTable striped interactive style={{ width: '100%' }}>
                        <colgroup>
                            <col style={{ width: '30%' }} />
                            <col style={{ width: '30%' }} />
                            <col style={{ width: '30%' }} />
                        </colgroup>
                        <thead>
                            <tr> <!-- Handles sorting functionality of the table -->
                                <th onClick={() => this.handleSort('ip')}>
                                    Client IP {sortColumn === 'ip' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                                </th>
                                <th onClick={() => this.handleSort('connectionStatus')}>
                                    Connection Status {sortColumn === 'connectionStatus' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                                </th>
                                <th onClick={() => this.handleSort('lastUpdate')}>
                                    Last Update {sortColumn === 'lastUpdate' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                                </th>
                            </tr>
                        </thead>
                        <tbody> <!-- Display each client as a table entry -->
                        {clients.map(client => (
                            <tr key={client.id}>
                                <td>{client.ip}</td>
                                <td>{client.connectionStatus}</td>
                                <td>{client.lastUpdate}</td>
                            </tr>
                        ))}
                        </tbody>
                    </HTMLTable>
                </div>
            </div>
            </>
        );
    }
}

export default Dashboard;