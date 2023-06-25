# Installation Guide

To install this system, a few steps need to be followed

**Cloning the Repository**

Clone the repository to your machine. This can be done as following:

```bash
git clone https://gitlab.ewi.tudelft.nl/cse2000-software-project/2022-2023-q4/cluster19/multi-modal-edge-ai/multi-modal-edge-ai.git
```

After that, you need to set up your python virtual environment. This can be done with any virtual environment tool, but for purposes of exemplification, `venv` will be used. For this project, Python 3.10 is being used, so it is important to create the virtual environment with that version of the Python interpreter.

First, create the virtual environment:

```bash
python3.10 -m venv venv
```

Then, activating the virtual environment is necessary. This differs depending on your operating system. 

For macOs/Linux users:

```bash
source venv/bin/activate
```

For Windows users:

```bash
myenv\Scripts\activate
```

After activating the virtual environment, it is necessary to install all the dependencies of the project. This can be done with the following command:

```bash
pip install -r requirements.txt # or requirements-dev.txt for development environment
```

After this, you'll also need to download the dependencies for the developer dashboard. First, make sure you have Node JS and Node Package Manager (npm) installed. You can download and install them from the official Node.js [website](https://nodejs.org/).

Then, navigate to the developer dashboard directory. Assuming you are on the root directory, you can do it this way

```bash
cd /multi_modal_edge_ai/server/developer_dashboard
```

Then, you'll need to isntall the necessary dependencies. You can do it with the following command:

```bash
npm install
```

After this, you are all set the run the system.

# Running the system

In order to run any of the system components, an `ssh` connections needs to be established with the MongoDB to be used. Be it in the client side, where the connections will serve as means to store all the personal data, or in the server side, where the MongoDB will serve to store jobs schedule. Establishing this ssh connection depends on the database you are making use of.

After such connection is established, you can now run both parts of system. 

**Server**

In order to run the server, you have to run both the server's `main.py`. To have access to the dashboard, one additional command needs to be executed.

To run the flask and federated server's, you'll only need to execute the following command, if you are on the root directory:

```bash
python multi_modal_edge_ai/server/main.py
```

The server should then start, and you should see something like the following:

```
 * Serving Flask app 'main'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://155.34.224.84:5000
Press CTRL+C to quit

```

After that, if you want to have access to the developer dashboard, navigate to the dashboard directory. You can do it as follows, provided you are on the root directory:

```bash
cd multi_modal_edge_ai/server/developer_dashboard
```

Then, you can simply run the following command and a new tab will be opened in your browser, displaying the dashboard:

```bash
npm start
```

**Client**

In order to run the client, you simply have to run the `main.py` file in the `client` directory. You'll also need to pass the name of the sensor MongoDB database as a program argument. If you are on the root directory, here is an example of how to run the client system:


```bash
python multi_modal_edge_ai/client/main.py sensor_db_123
```

The client will then start executing. Make sure you have a running server, and the IP of this server is set correctly on the client's code. Can be set on the top of [this file](../multi_modal_edge_ai/client/controllers/client_controller.py) for the flask server, and [here](../multi_modal_edge_ai/client/orchestrator.py) for the federated learning server (likely the same). 