import React, {useContext, useState} from 'react';
import { Tabs, Tab, Pre } from '@blueprintjs/core';
import hljs from "highlight.js/lib/core";
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/default.css';
import {DataFetchingContext} from "../context/DataFetchingContext";
hljs.registerLanguage('python', python);

const Logs = () => {
    const [selectedTabId, setSelectedTabId] = useState(0);
    const { serverLog, federationLog } = useContext(DataFetchingContext);

    const handleTabChange = (newTabId) => {
        setSelectedTabId(newTabId);
    };

    const logs = [
        { id: 0, title: 'Server', content: serverLog },
        { id: 1, title: 'Federation', content: federationLog },
    ];

    return (
        <div style={{ marginTop: '20px', color: 'white' }}>
            <h2>Logs</h2>
            <Tabs
                id="logs-tabs"
                selectedTabId={selectedTabId}
                onChange={handleTabChange}
                renderActiveTabPanelOnly={true}
            >
                {logs.map((log) => (
                    <Tab
                        key={log.id}
                        id={log.id}
                        title={log.title}
                        panel={<Pre>{log.content}</Pre>}
                        style={{
                            background: 'none',
                            color: selectedTabId === log.id ? 'white' : 'gray',
                            borderBottomColor: selectedTabId === log.id ? 'white' : '',
                            boxShadow: selectedTabId === log.id ? '0px -2px 0px 0px white' : ''
                        }}
                    />
                ))}
            </Tabs>
        </div>
    );
};

export default Logs;