import CircleCard from "./CircleCard";
import React, {useContext, useEffect, useState} from "react";
import { DataFetchingContext } from "../context/DataFetchingContext";
import SortingUtils from "../utils/sorting_utils";

const StatCircles = () => {
    const { clients, isFederationRunning, federationWorkloads } = useContext(DataFetchingContext);
    const [numClients, setNumClients] = useState(0);
    const [lastModelAggregation, setLastModelAggregation] = useState('Unavailable');
    const [nextScheduledModelAggregation, setNextScheduledModelAggregation] = useState('Unavailable');


    useEffect(() => {
        setNumClients(Object.keys(clients["connected_clients"]).length);
        setLastModelAggregation(isFederationRunning ? formatDateTime((new Date()).toString()) : lastModelAggregation)
        setNextScheduledModelAggregation(federationWorkloads.length > 0 ?
            formatDateTime(SortingUtils.sortByDateTime(federationWorkloads, 'scheduled_time', 'asc')[0]['scheduled_time'])
            : 'None')
    }, [clients]);

    const formatDateTime = (string) => {
        const date = new Date(string);
        return `${date.toJSON().slice(0, 10)}\n${date.toJSON().slice(11, 19)}`;
    };


    return (
        <div style={{ display: "flex", justifyContent: "center", color: "white" }}>
            <CircleCard title="Number of Clients" value={numClients} />
            <CircleCard title="Last Aggregation" value={lastModelAggregation} />
            <CircleCard title="Next Aggregation" value={nextScheduledModelAggregation} />
        </div>
    );
};

export default StatCircles;