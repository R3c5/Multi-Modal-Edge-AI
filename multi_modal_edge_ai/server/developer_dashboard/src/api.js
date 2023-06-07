import axios from 'axios';

const BASE_URL = 'http://127.0.0.1:5000';

export const getClientInfo = async (token) => {
    try {
        const config = {
            headers: {
                Authorization: token,
            },
        };
        const response = await axios.get(`${BASE_URL}/dashboard/get_client_info`, config);
        return response.data;
    } catch (error) {
        console.error('Error fetching client info:', error);
        throw error;
    }
};

export const getErrorLog = async (token) => {
    try {
        const config = {
            headers: {
                Authorization: token,
            },
        };
        const response = await axios.get(`${BASE_URL}/dashboard/get_error_log`, config);
        return response.data;
    } catch (error) {
        console.error('Error fetching error log:', error);
        throw error;
    }
};