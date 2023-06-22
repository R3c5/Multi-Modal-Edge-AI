import React, {useContext} from 'react';
import { Button, HTMLTable } from '@blueprintjs/core';
import { Popover2, Classes } from '@blueprintjs/popover2';
import '@blueprintjs/popover2/lib/css/blueprint-popover2.css';
import {removeFederationWorkload} from "../api";
import SECRET_TOKEN from "../secrets";
import {DataFetchingContext} from "../context/DataFetchingContext";

const FederationTable = () => {
    const { federationWorkloads, isFederationRunning } = useContext(DataFetchingContext)

    const handleCancelClick = (index) => {
        removeFederationWorkload(SECRET_TOKEN, federationWorkloads[index]['id'])
    };

    const renderParametersPopover = (parameters) => {
        return (
            <div>
                {Object.entries(parameters).map(([key, value]) => (
                    <div key={key}>
                        <strong>{key}: </strong>
                        {value}
                    </div>
                ))}
            </div>
        );
    };

    const renderTableRows = () => {
        return federationWorkloads.map((entry, index) => (

            <tr key={index}>
                <td style={{ color: 'white' }}>
                    <Popover2
                        content={renderParametersPopover(entry.config)}
                        interactionKind="hover"
                        hoverOpenDelay={150}
                        hoverCloseDelay={150}
                        popoverClassName={Classes.POPOVER2_CONTENT_SIZING}
                        placement="auto"
                    >
                        <span>{entry.scheduled_time}</span>
                    </Popover2>
                </td>
                <td style={{ color: 'white'}}>
                    <span>{entry.workload_type.charAt(0).toUpperCase() + entry.workload_type.slice(1)}</span>
                </td>
                <td style={{ color: 'white' }}>
                    <span>{entry.cron_job.toString()}</span>
                </td>
                <td style={{ color: 'white' }}>
                    <span>{entry.crontab}</span>
                </td>
                <td style={{ textAlign: 'right' }}>
                    <Button
                        text="Cancel"
                        intent="danger"
                        onClick={() => handleCancelClick(index)}
                    />
                </td>
            </tr>
        ));
    };
    const renderStatusMessage = () => {
        if (isFederationRunning === null) {
            return "Can't view running status";
        } else if (isFederationRunning) {
            return "A task is currently running";
        } else {
            return "No task is currently running";
        }
    };

    return(
        <div>
            <p>{renderStatusMessage()}</p>
            <HTMLTable striped style={{ width: '100%' }}>
                <thead>
                <tr>
                    <th style={{ color: 'white' }}>Scheduled Time</th>
                    <th style={{ color: 'white' }}>Workload Type</th>
                    <th style={{ color: 'white' }}>Recurrent Job</th>
                    <th style={{ color: 'white' }}>Cron String</th>
                </tr>
                </thead>
                <tbody>{renderTableRows()}</tbody>
            </HTMLTable>
        </div>
    );
};

export default FederationTable;