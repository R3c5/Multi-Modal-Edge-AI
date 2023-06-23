import React, {useState, useContext, useEffect} from 'react';
import { HTMLTable, Icon } from "@blueprintjs/core";
import SortingUtils from "../utils/sorting_utils";
import { DataFetchingContext } from "../context/DataFetchingContext";

const ClientTable = () => {
    const { clients } = useContext(DataFetchingContext);
    const [ clientEntries, setClientEntries] = useState([])
    const [sortColumn, setSortColumn] = useState(null);
    const [sortDirection, setSortDirection] = useState('asc');

    useEffect(() => {
        setClientEntries(convertClientJSONtoList(clients))
        console.log("clientEntries",clientEntries)
        handleSort(sortColumn)
    }, [clients]);

    const formatDateTime = (string) => {
        const date = new Date(string);
        return `${date.toJSON().slice(0,10)}\n${date.toJSON().slice(11,19)}`;
    };

    const convertClientJSONtoList = (clientJSON) => {
        console.log(clientJSON)
        clientJSON = clientJSON ?? {};
        const clientDicts = clientJSON["connected_clients"];
        let clientList = [];
        Object.keys(clientDicts).forEach(ip => clientList.push({
            "ip":ip,
            "last_seen":formatDateTime(clientDicts[ip]["last_seen"]),
            "num_adls":clientDicts[ip]["num_adls"],
            "num_anomalies":clientDicts[ip]["num_anomalies"],
            "status":clientDicts[ip]["status"],
            "last_model_update":formatDateTime(clientDicts[ip]["last_model_aggregation"]) === "0001-01-01\n00:00:00" ? "Never"
                : formatDateTime(clientDicts[ip]["last_model_aggregation"])
        }));
        return clientList;
    };

    const handleSort = (column) => {
        const direction =
            sortColumn === column && sortDirection === 'asc' ? 'desc' : 'asc';

        if (Object.keys(clientEntries).length === 0) {
            setSortColumn(column);
            setSortDirection(direction);
            return;
        }

        let sortedClients;
        if (column === 'status') {
            sortedClients = SortingUtils.sortAlphabetically(clientEntries, column, direction);
        } else if (column === 'last_model_update') {
            sortedClients = SortingUtils.sortByDateTime(clientEntries, column, direction);
        } else if (column === 'ip') {
            sortedClients = SortingUtils.sortByIPAddress(clientEntries, column, direction);
        } else if (column === 'num_anomalies' || column === 'num_adls') {
            sortedClients = SortingUtils.sortNumerically(clientEntries, column, direction);
        } else {
            // Default sorting if column not recognized
            sortedClients = clientEntries;
        }

        setClientEntries(sortedClients);
        setSortColumn(column);
        setSortDirection(direction);
    };

    return (
        <div style={{ maxHeight: '50vh', overflow: 'auto', width: '100%' }}>
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
                    <th onClick={() => handleSort('ip')} style={{ color: "white" }}>
                        Client IP {sortColumn === 'ip' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                    </th>
                    <th onClick={() => handleSort('status')} style={{ color: "white" }}>
                        Connection Status {sortColumn === 'status' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                    </th>
                    <th onClick={() => handleSort('last_model_update')} style={{ color: "white" }}>
                        Last Model Update {sortColumn === 'last_model_update' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                    </th>
                    <th onClick={() => handleSort('num_adls')} style={{ color: "white" }}>
                        Recently Inferred ADLs {sortColumn === 'num_adls' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                    </th>
                    <th onClick={() => handleSort('num_anomalies')} style={{ color: "white" }}>
                        Recently Detected Anomalies {sortColumn === 'num_anomalies' && <Icon icon={sortDirection === 'asc' ? 'caret-up' : 'caret-down'} />}
                    </th>
                </tr>
                </thead>
                <tbody>
                {Object.keys(clientEntries).map(client => (
                    <tr key={client.id}>
                        <td style={{ color: "white" }}>{clientEntries[client]["ip"]}</td>
                        <td style={{ color: "white" }}>{clientEntries[client]["status"]}</td>
                        <td style={{ color: "white" }}>{clientEntries[client]["last_model_update"]}</td>
                        <td style={{ color: "white" }}>{clientEntries[client]["num_adls"]}</td>
                        <td style={{ color: "white" }}>{clientEntries[client]["num_anomalies"]}</td>
                    </tr>
                ))}
                </tbody>
            </HTMLTable>
        </div>
    );
};

export default ClientTable;