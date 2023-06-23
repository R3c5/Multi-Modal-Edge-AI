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

import React, { useState } from 'react';
import 'normalize.css';
import "@blueprintjs/core/lib/css/blueprint.css";
import '../App.css';
import FederationScheduler from "./FederationScheduler";
import FederationTable from "./FederationTable";
import Logs from "./Logs";
import RefreshButton from "./RefreshButton";
import ClientTable from "./ClientTable";
import StatCircles from "./StatCircles";

const Dashboard = () => {
    const [isLoading, setIsLoading] = useState(false);

    if (isLoading) {
        return <div>Loading...</div>;
    }

    return (
        <div>
            <div>
                <RefreshButton />
            </div>
            <div>
                <StatCircles />
            </div>
            <div style={{ width: '100%', color: "white" }}>
                <h2>Federation Scheduler</h2>
                <FederationScheduler />
            </div>
            <div style={{ width: '100%', color: "white" }}>
                <h2>Scheduled Jobs</h2>
                <FederationTable />
            </div>
            <div style={{ width: '100%', color: "white" }}>
                <h2>Clients</h2>
                <ClientTable />
            </div>
            <div style={{ marginTop: '20px', color: 'white', maxHeight: '50vh', overflow: 'auto', width: '100%' }}>
                <Logs />
            </div>
        </div>
    );
};

export default Dashboard;