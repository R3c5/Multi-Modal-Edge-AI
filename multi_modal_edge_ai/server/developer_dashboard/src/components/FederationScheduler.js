import React, { useRef, useState} from 'react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import NumericInput from 'react-numeric-input';
import { FormGroup, Position, Button } from '@blueprintjs/core';
import {Popover2} from "@blueprintjs/popover2";
import '@blueprintjs/popover2/lib/css/blueprint-popover2.css';
import { Cron } from 'react-js-cron';
import 'react-js-cron/dist/styles.css'
import {scheduleFederationWorkload} from "../api";
import SECRET_TOKEN from "../secrets";

const FederationScheduler = () => {
    const configDisplay = { // You can configure the display name, default value, min value, max value, step size, and description
        "num_rounds": {'name':'Number of Rounds', 'default': '1', 'min':1, 'max':null, 'step':1, 'description':'The number of rounds in the federated learning workload'},
        "window_size": {'name':'Window Size', 'default': '1', 'min':1, 'max':null, 'step':1, 'description':'Amount of activities the anomaly detection algorithm will consider at once'}, // NOT UPDATEABLE
        "window_slide": {'name':'Window Slide', 'default': '1','min':0, 'max':null, 'step':1, 'description':'Amount by which the sliding window algorithm will slide'},
        "one-hot": {'name':'One-Hot Encoding', 'default': 'true', 'description':'Boolean for enabling one-hot encoding or label encoding'}, // NOT UPDATEABLE
        "batch_size": {'name':'Batch Size', 'default': '32', 'min':1, 'max':null, 'step':1, 'description':'Number of windows in each training batch'},
        "learning_rate": {'name':'Learning Rate', 'default': '0.01', 'min':0, 'max':1, 'step':0.01, 'description':'Local learning rate of the optimisation algorithm'},
        "n_epochs": {'name':'Number of Epochs', 'default': '2', 'min':1, 'max':null, 'step':1, 'description':'Number of epochs for the local training'},
        "event_based": {'name':'Event-Based Windowing', 'default': 'true', 'description':'Boolean for whether the anomaly detection is event-based or time-based'}, // NOT UPDATEABLE
        "anomaly_whisker": {'name':'Anomaly Whisker', 'default': '1.75', 'min':0, 'max':null, 'step':1, 'description':'Configures the size of the whisker by which clean and anomalous windows will be separated in terms of duration'},
        "clean_test_data_ratio": {'name':'Clean Test Data Ratio', 'default': '0.2', 'min':0, 'max':1, 'step':0.01, 'description':'Ratio of clean data that will be in the test set'},
        "anomaly_gen_ratio": {'name':'Anomaly Generation Ratio', 'default': '0.1', 'min':0, 'max':null, 'step':0.01, 'description':'Ratio of generated anomalous data to seen anomalous data'},
        "reconstruction_error_quantile": {'name':'Reconstruction Error Quantile', 'default': '0.99', 'min':0, 'max':1, 'step':0.01, 'description':'Quantile in terms of seen training errors for maximum reconstruction error threshold'},
        "fraction_fit": {'name':'Training Ratio', 'default': '1.0', 'min':0, 'max':1, 'step':0.01, 'description':'Fraction of clients used during training. In case `min_fit_clients` is larger than `fraction_fit * available_clients`, `min_fit_clients` will still be sampled.'},
        "fraction_evaluate": {'name':'Validation Ratio', 'default': '0.99', 'min':0, 'max':1, 'step':0.01, 'description':'Fraction of clients used during validation. In case `min_evaluate_clients` is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients` will still be sampled.'},
        "min_fit_clients": {'name':'Minimum Training Clients', 'default': '2', 'min':2, 'max': null, 'step':1, 'description':'Minimum number of clients used during training.'},
        "min_evaluate_clients": {'name':'Minimum Validation Clients', 'default': '2', 'min':2, 'max': null, 'step':1, 'description':'Minimum number of clients used during validation.'},
        "min_available_clients": {'name':'Minimum Total Clients', 'default': '2', 'min':2, 'max':null, 'step':1, 'description':'Minimum number of total clients in the system.'},
    }
    const disabledConfigs = ['window_size', 'one-hot', 'event_based'];
    const [config, setConfig] = useState({
        "num_rounds": 1,
        "window_size": 1,
        "window_slide":1,
        "one-hot": true,
        "batch_size":32,
        "learning_rate": 0.01,
        "n_epochs": 2,
        "event_based": true,
        "anomaly_whisker": 1.75,
        "clean_test_data_ratio": 0.2,
        "anomaly_gen_ratio": 0.1,
        "reconstruction_error_quantile": 0.99,
        "fraction_fit": 1.0,
        "fraction_evaluate": 0.99,
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "min_available_clients": 2
    });
    const [scheduledTime, setScheduledTime] = useState(null);
    const [responseDisplay, setResponseDisplay] = useState('');
    const [schedule_type, setScheduleType] = useState('recurrent');
    const [crontab, setCrontab] = useState('* * * * *z');
    const cronRef = useRef(null);


    const handleParameterChange = (value, parameter) => {
        setConfig((prevState) => ({
            ...prevState,
            [parameter]: { ...prevState[parameter], value: value },
        }));
    };

    const handleDateChange = (date) => {
        setScheduledTime(date || null);
        setResponseDisplay("")
    };

    const handleOptionChange = (event) => {
        const selectedOption = event.target.value;
        setScheduleType(selectedOption);
        setResponseDisplay("")
    };

    const handleCronExpressionChange = (value) => {
        console.log(value)
        setCrontab(value);
    };

    const handleSubmit = async () => {
        const response = await scheduleFederationWorkload(SECRET_TOKEN, config, schedule_type, crontab, scheduledTime)
        if(response === undefined){
            setResponseDisplay("Could not receive response")
        }
        else if(response.status === 200){
            setResponseDisplay("Task scheduled successfully")
        } else if(response.data.error !== undefined){
            setResponseDisplay(response.data.error)
        }
        else {
            setResponseDisplay("There was an error scheduling task")
        }
    };

    return (
        <div style={{ textAlign: 'center' }}>
            <div>
                <FormGroup label="Parameters">
                    <div style={{ margin: '10px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gridGap: '20px' }}>
                        {Object.keys(config).map((parameter) => (
                            <div key={parameter}>
                                <Popover2
                                    content={
                                        <div style={{ padding: '10px', borderRadius: '10px' }}>
                                            {configDisplay[parameter]['description']}
                                        </div>
                                    }
                                    position={Position.BOTTOM}
                                    interactionKind="hover-target"
                                    modifiers={{
                                        arrow: { enabled: false },
                                    }}
                                    style={{ width: '10vw' }}
                                >
                                    <FormGroup label={configDisplay[parameter]['name']}>
                                        <NumericInput
                                            value={config[parameter]}
                                            min={configDisplay[parameter]['min']}
                                            max={configDisplay[parameter]['max'] !== null ? configDisplay[parameter]['max'] : undefined}
                                            step={Number.isInteger(configDisplay[parameter]['step']) ? 1 : 'any'}
                                            onChange={(value) => handleParameterChange(value, parameter)}
                                            placeholder={configDisplay[parameter]['default']}
                                            disabled={disabledConfigs.includes(parameter)}
                                        />
                                    </FormGroup>
                                </Popover2>
                            </div>
                        ))}
                    </div>
                </FormGroup>
            </div>
            <FormGroup label="Scheduled Time" style={{ textAlign: 'center' }}>
                <select value={schedule_type} onChange={handleOptionChange}>
                    <option value="recurrent">Recurrent</option>
                    <option value="one-time">One Time</option>
                    <option value="immediate">Immediate</option>
                </select>
                {schedule_type === "recurrent" && (
                    <div style={{ marginTop: '10px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <Cron ref={cronRef}
                            setValue={(value) => handleCronExpressionChange(value)}
                            value={crontab}
                        />
                    </div>
                )}
                {schedule_type === "one-time" && (
                    <div style={{ marginTop: '10px' }}>
                        <DatePicker
                            placeholderText="Select date and time"
                            placeholder="Select date and time"
                            selected={scheduledTime}
                            onChange={handleDateChange}
                            showTimeInput
                            dateFormat="yyyy-MM-dd HH:mm"
                            popperPlacement="bottom"
                        />
                    </div>
                )}
                {schedule_type === "immediate"}
                <div style={{ marginTop: '10px' }}>
                    <Button onClick={handleSubmit} style={{ justifySelf: 'center' }}>
                        Send Request
                    </Button>
                </div>
            </FormGroup>
            <div>{responseDisplay}</div>
        </div>
    );
};

export default FederationScheduler;