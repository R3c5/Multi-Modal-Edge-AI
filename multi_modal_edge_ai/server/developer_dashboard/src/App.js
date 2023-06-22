import 'normalize.css';
import "@blueprintjs/core/lib/css/blueprint.css";
import React from 'react';
import './App.css';
import Dashboard from "./components/Dashboard";
import {DataFetchingProvider} from "./context/DataFetchingContext";

function App() {
    return (
        <DataFetchingProvider>
            <Dashboard />
        </DataFetchingProvider>
    );
}

export default App;