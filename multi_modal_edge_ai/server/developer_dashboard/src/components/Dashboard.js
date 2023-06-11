import 'normalize.css';
import "@blueprintjs/core/lib/css/blueprint.css";
import SECRET_TOKEN from '../secrets'
import React from 'react';
import '../App.css';
import {getClientInfo, getErrorLog} from "../api";
import hljs from "highlight.js/lib/core";
import SortingUtils from "../utils/sorting_utils";
import CircleCard from "../components/CircleCard";
import {HTMLTable, Icon} from "@blueprintjs/core";
import python from 'highlight.js/lib/languages/python';

hljs.registerLanguage('python', python);

// const exampleClients =
//     {
//         "connected_clients": {
//             "127.0.0.1": {
//                 "last_seen": "Tue, 06 Jun 2023 09:18:26 GMT",
//                 "num_adls": 3,
//                 "num_anomalies": 4,
//                 "status": "Connected"
//             },
//             "127.0.0.2": {
//                 "last_seen": "Wed, 07 Jun 2023 09:18:26 GMT",
//                 "num_adls": 2,
//                 "num_anomalies": 10,
//                 "status": "Disconnected"
//             }
//         }
//     }
class Dashboard extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            numClients: 0,
            lastModelUpdate: '',
            clients: {},
            sortColumn: null,
            sortDirection: 'asc',
            isLoading: true,
            logs: null,
        };
    }
    componentDidMount() {
        this.fetchServerData()
    }

    async fetchServerData() {
        const clientData = await getClientInfo(SECRET_TOKEN)
        const logData = await getErrorLog(SECRET_TOKEN)
        console.log(clientData)
        const formattedClientData = this.convertClientJSONtoList(clientData)

        // Update state with received data
        this.setState({
            numClients: formattedClientData.length,
            lastModelUpdate: '2023-05-28 10:30:00', // TODO: when federation is implemented
            clients: formattedClientData,
            logs: hljs.highlight('python', await logData).value,
            isLoading: false,
        });
    }

    formatDateTime(string){
        const date = new Date(string)
        return `${date.toJSON().slice(0,10)}\n${date.toJSON().slice(11,19)}`
    }

    convertClientJSONtoList(clients){
        const clientDicts = clients["connected_clients"];
        let clientList = [];
        Object.keys(clientDicts).forEach(ip => clientList.push(
            {
                "ip":ip,
                "last_seen":this.formatDateTime(clientDicts[ip]["last_seen"]),
                "num_adls":clientDicts[ip]["num_adls"],
                "num_anomalies":clientDicts[ip]["num_anomalies"],
                "status":clientDicts[ip]["status"],
                "last_model_update":this.formatDateTime(clientDicts[ip]["last_seen"]) // TODO: change last_seen during federation
            })
        );
        return clientList
    }

    /**
     * Handles sorting of the clients based on the specified column.
     * @param {string} column - The column to sort by.
     */
    handleSort(column) {
        const { clients, sortColumn, sortDirection } = this.state;
        // const { sortedClients, direction} = SortingUtils.handleSort(column, clients, sortColumn, sortDirection)

        const direction =
            sortColumn === column && sortDirection === 'asc' ? 'desc' : 'asc';

        let sortedClients;
        if (column === 'status') {
            sortedClients = SortingUtils.sortAlphabetically(clients, column, direction);
        } else if (column === 'last_model_update') {
            sortedClients = SortingUtils.sortByDateTime(clients, column, direction);
        } else if (column === 'ip') {
            sortedClients = SortingUtils.sortByIPAddress(clients, column, direction);
        } else if (column === 'num_anomalies' || column === 'num_adls') {
            sortedClients = SortingUtils.sortNumerically(clients, column, direction)
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
    render() {
        const { numClients, clients, sortColumn, sortDirection, logs, isLoading } = this.state;
        if (isLoading) {
            return <div>Loading...</div>;
        }

        return (
            <div>
                <div>
                    <div style={{ display: 'flex', justifyContent: 'center', color:"white"}}>
                        <CircleCard title="Number of clients" value={numClients} />
                        <CircleCard title="Last aggregation" value={this.formatDateTime("Tue, 06 Jun 2023 09:18:26 GMT")} />
                        <CircleCard title="Next aggregation" value={this.formatDateTime("Wed, 07 Jun 2023 10:20:16 GMT")} />
                    </div>
                </div>
                <div style={{ width: '100%', color: "white"}}>
                    <h2>Clients</h2>
                    <div style={{maxHeight: '50vh', overflow: 'auto', width: '100%'}}>
                        <HTMLTable striped interactive style={{ width: '100%' }}>
                            <colgroup>
                                <col style={{ width: '20%' }} />
                                <col style={{ width: '20%' }} />
                                <col style={{ width: '20%' }} />
                                <col style={{ width: '20%' }} />
                                <col style={{ width: '20%' }} />
                            </colgroup>
                            <thead>
                            <tr>
                                <th onClick={() => this.handleSort('ip')} style={{color:"white"}}>
                                    Client IP {sortColumn === 'ip' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                                </th>
                                <th onClick={() => this.handleSort('status')} style={{color:"white"}}>
                                    Connection Status {sortColumn === 'status' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                                </th>
                                <th onClick={() => this.handleSort('last_model_update')} style={{color:"white"}}>
                                    Last Model Update {sortColumn === 'last_model_update' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                                </th>
                                <th onClick={() => this.handleSort('num_adls')} style={{color:"white"}}>
                                    Recently Inferred ADLs {sortColumn === 'num_adls' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                                </th>
                                <th onClick={() => this.handleSort('num_anomalies')} style={{color:"white"}}>
                                    Recently Detected Anomalies {sortColumn === 'num_anomalies' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                                </th>
                            </tr>
                            </thead>
                            <tbody>
                            {Object.keys(clients).map(client => (
                                <tr key={client.id}>
                                    <td style={{color:"white"}}>{clients[client]["ip"]}</td>
                                    <td style={{color:"white"}}>{clients[client]["status"]}</td>
                                    <td style={{color:"white"}}>{clients[client]["last_model_update"]}</td>
                                    <td style={{color:"white"}}>{clients[client]["num_adls"]}</td>
                                    <td style={{color:"white"}}>{clients[client]["num_anomalies"]}</td>
                                </tr>
                            ))}
                            </tbody>
                        </HTMLTable>
                    </div>
                </div>
                <div style={{marginTop: '20px', color: 'white'}}>
                    <h2>Logs</h2>
                    <div style={{ color: 'white', maxHeight: '50vh', overflow: 'auto', width: '100%'}}>

                        {logs ? (
                            <pre style={{ whiteSpace: 'pre-wrap' }}>
                                <code dangerouslySetInnerHTML={{ __html: logs }} />
                            </pre>
                        ) : (
                            <div>No logs available.</div>
                        )}
                    </div>
                </div>
            </div>
        );
    }
}

export default Dashboard;