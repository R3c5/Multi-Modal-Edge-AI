import React, { createContext, useState, useEffect } from 'react';
import {
    getServerLog,
    getFederationLog,
    getFederationWorkloads,
    isFederationWorkloadRunning,
    getClients
} from '../api';
import SECRET_TOKEN from '../secrets'

const DataFetchingContext = createContext({
    clients:{"connected_clients":{}},
    serverLog:'',
    federationLog:'',
    federationWorkloads:[],
    isFederationRunning:null,
    testTime:'',
});
const DataFetchingProvider = ({ children }) => {
    const [clients, setClients] = useState({"connected_clients":{}});
    const [serverLog, setServerLog] = useState('Nothing to display');
    const [federationLog, setFederationLog] = useState('Nothing to display');
    const [federationWorkloads, setFederationWorkloads] = useState([]);
    const [isFederationRunning, setIsFederationRunning] = useState(null);

    useEffect(() => {
        const fetchAllData = async () => {
            try {
                const responseClients = await getClients(SECRET_TOKEN);
                if (responseClients !== undefined) {
                    setClients(responseClients);
                }
            } catch (error) {
                // console.error('Error fetching client info:', error);
            }
            try {
                const responseServerLog = await getServerLog(SECRET_TOKEN);
                if (responseServerLog !== undefined) {
                    setServerLog(responseServerLog);
                }
            } catch (error) {
                // console.error('Error fetching server log:', error);
            }
            try {
                const responseFederationLog = await getFederationLog(SECRET_TOKEN);
                if (responseFederationLog !== undefined) {
                    setFederationLog(responseFederationLog);
                }
            } catch (error) {
                // console.error('Error fetching federation log:', error);
            }
            try {
                const responseFederationWorkloads = await getFederationWorkloads(SECRET_TOKEN);
                if (responseFederationWorkloads !== undefined) {
                    setFederationWorkloads(responseFederationWorkloads)
                }
            } catch (error) {
                // console.error('Error fetching federation workloads:', error);
            }
            try {
                const responseIsFederationRunning = await isFederationWorkloadRunning(SECRET_TOKEN);
                if (responseIsFederationRunning !== undefined) {
                    setIsFederationRunning(responseIsFederationRunning)
                }
            } catch (error) {
                // console.error('Error fetching isFederationRunning:', error);
            }

        };

        fetchAllData();

        const timer = setInterval(() => {
            fetchAllData(); // Fetch data every 5 seconds
        }, 5000);

        return () => {
            clearInterval(timer); // Clean up timer on component unmount
        };
    }, []);

    return (
        <DataFetchingContext.Provider
            value={{
                clients,
                serverLog,
                federationLog,
                federationWorkloads,
                isFederationRunning,
            }}
        >
            {children}
        </DataFetchingContext.Provider>
    );
};

export { DataFetchingContext, DataFetchingProvider };
