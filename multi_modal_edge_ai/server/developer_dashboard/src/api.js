import axios from 'axios';

const BASE_URL = 'http://127.0.0.1:5000';

export const getClients = async (token) => {
    try {
        const config = {
            headers: {
                Authorization: token,
            },
        };
        const response = await axios.get(`${BASE_URL}/dashboard/get_client_info`, config);
        return response.data;
    } catch (error) {
        console.error('Error fetching clients:', error);
        throw error;
    }
};

export const getServerLog = async (token) => {
    try {
        const config = {
            headers: {
                Authorization: token,
            },
        };
        const response = await axios.get(`${BASE_URL}/dashboard/get_error_log`, config);
        return response.data;
    } catch (error) {
        console.error('Error fetching server log:', error);
        throw error;
    }
};

export const getFederationLog = async (token) => {
    try {
        const config = {
            headers: {
                Authorization: token,
            },
        };
        const response = await axios.get(`${BASE_URL}/dashboard/get_federation_log`, config);
        return response.data;
    } catch (error) {
        console.error('Error fetching federation log:', error);
        throw error;
    }
};

export const getFederationWorkloads = async (token) => {
    try {
        const config = {
            headers: {
                Authorization: token,
            },
        };
        const response = await axios.get(`${BASE_URL}/dashboard/fetch_all_federation_workloads`, config);
        return response.data;
    } catch (error) {
        console.error('Error fetching federation workloads:', error)
    }
}

export const scheduleFederationWorkload = async (token, config, schedule_type, crontab, date) => {
    try {
        const data = {
            schedule_type: schedule_type,
            config: config,
            crontab: crontab,
            date: date
        };

        const headers = {
            Authorization: token
        };
        console.log("data: " , data)
        const response = await axios.post(`${BASE_URL}/dashboard/schedule_federation_workload`, data, { headers });
        console.log(response)
        return response;
    } catch (error) {
        console.error('Error scheduling federation workload:', error);
    }
};

export const removeFederationWorkload = async (token, job_id) => {
    try {
        const data = {
            job_id: job_id
        };

        const headers = {
            Authorization: token
        };

        const response = await axios.delete(`${BASE_URL}/dashboard/remove_federation_workload`, { data, headers });
        return response.data;
    } catch (error) {
        console.error('Error deleting scheduled federation workload:', error);
    }
};

export const isFederationWorkloadRunning = async (token) => {
    try {
        const config = {
            headers: {
                Authorization: token,
            },
        };
        const response = await axios.get(`${BASE_URL}/dashboard/is_federation_workload_running`, config);
        return response.data;
    } catch (error) {
        console.error('Error checking running federation workload:', error);
        throw error;
    }
};