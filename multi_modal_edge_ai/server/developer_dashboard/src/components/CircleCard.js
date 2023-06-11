import {Card} from "@blueprintjs/core";
import 'highlight.js/styles/default.css';
import React from "react";

const CircleCard = ({ title, value }) => (
    <Card style={{ width: '200px', margin: '10px', backgroundColor: '#181a1b' }}>
        <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '10px'}}>{title}</div>
            <div style={{ width: '125px', height: '125px', borderRadius: '50%', backgroundColor: '#26292b', margin: '0 auto', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '24px', fontWeight: 'bold' }}>
                {value}
            </div>
        </div>
    </Card>
);

export default CircleCard;