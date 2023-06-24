import CircleCard from "./CircleCard";
import React, {useContext, useEffect, useState} from "react";
import { DataFetchingContext } from "../context/DataFetchingContext";
import SortingUtils from "../utils/sorting_utils";

const StatCircles = () => {
    const { clients, isWorkloadRunning, federationWorkloads } = useContext(DataFetchingContext);
    const [numClients, setNumClients] = useState(0);
    const [lastModelAggregation, setLastModelAggregation] = useState('None')
    const [nextScheduledModelAggregation, setNextScheduledModelAggregation] = useState('None');
    const [lastModelPersonalization, setLastModelPersonalization] = useState('None')
    const [nextScheduledModelPersonalization, setNextScheduledModelPersonalization] = useState('None');

    useEffect(() => {
        const federationList = federationWorkloads.filter((item) => item.workload_type === "federation");
        const personalizationList = federationWorkloads.filter((item) => item.workload_type === "personalization");
        setNumClients(Object.keys(clients["connected_clients"]).length);

        if (isWorkloadRunning !== null) {
            if (isWorkloadRunning.hasOwnProperty('workload_type')) {
                if (isWorkloadRunning.workload_type === "federation") {
                    setLastModelAggregation(formatDateTime((new Date()).toString()));
                }
                if (isWorkloadRunning.workload_type === "personalization") {
                    setLastModelPersonalization(formatDateTime((new Date()).toString()));
                }
            }
        }
        setNextScheduledModelPersonalization(
            personalizationList.length > 0
                ? formatDateTime(SortingUtils.sortByDateTime(personalizationList, 'scheduled_time', 'asc')[0]['scheduled_time'])
                : 'None'
        );
        setNextScheduledModelAggregation(
            federationList.length > 0
                ? formatDateTime(SortingUtils.sortByDateTime(federationList, 'scheduled_time', 'asc')[0]['scheduled_time'])
                : 'None'
        );
    }, [federationWorkloads]);

    const formatDateTime = (string) => {
        const date = new Date(string);
        return `${date.toJSON().slice(0, 10)}\n${date.toJSON().slice(11, 19)}`;
    };


    return (
        <div style={{ display: "flex", justifyContent: "center", color: "white" }}>
            <CircleCard title="Number of Clients" value={numClients} />
            <CircleCard title="Last Federated" value={lastModelAggregation} />
            <CircleCard title="Next Federated" value={nextScheduledModelAggregation} />
            <CircleCard title="Last Personalized" value={lastModelPersonalization} />
            <CircleCard title="Next Personalized" value={nextScheduledModelPersonalization} />

        </div>
    );
};

export default StatCircles;