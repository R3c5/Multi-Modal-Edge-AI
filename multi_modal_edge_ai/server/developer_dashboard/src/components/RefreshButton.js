import React from 'react';
import refreshIcon from '../assets/refresh-arrow.png';

const RefreshButton = () => {
    const handleRefresh = () => {
        window.location.reload();
    };

    return (
        <div style={{ position: 'absolute', top: '10px', left: '10px' }}>
            <img
                src={refreshIcon}
                alt="Refresh"
                style={{ width: '30px', height: '30px', cursor: 'pointer' }}
                onClick={handleRefresh}
            />
        </div>
    );
};

export default RefreshButton;