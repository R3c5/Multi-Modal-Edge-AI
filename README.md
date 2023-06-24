<h1 align="center">
  <span style="color:grey;font-weight:bold;text-stroke: 1px white;-webkit-text-stroke-width: 1px;-webkit-text-fill-color: white;">Multi-Modal Edge</span>-<span style="color:black;font-weight:bold;text-stroke: 1px white;-webkit-text-stroke-width: 1px;-webkit-text-fill-color: #4287f5; font-weight:bold">AI</span>
</h1>



<h2 align="center">
   Future of Healthcare: Edge AI Technology to Assist Elderly in their Daily Lives with Multi-Modal Federated Learning.
</h2>

<h3 align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#features">Features</a> |
  <a href="#installation">Installation</a> |
  <a href="#usage">Usage</a> |
  <a href="#contributing">Contributing</a> |
  <a href="#contributors">Contributors</a> |
  <a href="#license">License</a>
</h3>

---
<div align="center">
    <img src="https://img.shields.io/badge/Python-3-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/AI-Enabled-red.svg" alt="AI">
    <img src="https://img.shields.io/badge/Pytest-Enabled-darkgreen.svg">
    <img src="https://img.shields.io/badge/scikit--learn-Enabled-orange.svg">
    <img src="https://img.shields.io/badge/Anomaly-Detection-blue.svg">
</div>

---

## **Introduction**

The system was created as part of an IoT homekit that will be used in assisting the caretakers supervise more elderly people that still want to be independent but may require additional support. To that extent, it uses machine learning models to predict anomalies in the daily routines, based on information from sensor data and save the anomalies in a database, from where the caretaker will be notified. It has a few key principles that had to be followed throughout it's whole development proces:

- **Protect data privacy**: The sensor data is stored locally in a database, and never sent to the server
- **Model Improvements**: Federated Learning is used to be able to further train the modelsy, but without violating the previous point.
- **Scalable**: The system was designed to be scalable in the number of clients
- **Maintainability and Reusability**: The codebase makes use of existent frameworks like [PyTorch](https://pytorch.org/docs/stable/index.html) and [Flower](https://flower.dev/docs/) in order for the code to be readable maintainable and of the highest quality.

## **Features**

Here are the most notable features available in our system:

* **Sandboxes**: There are 2 sandboxes one for ADL (activity of daily living) inference and one for anomaly detection. Each one includes multiple models to choose, both deep learning models and non deep learning models. You can create a model, configure it and validate it on a public dataset chosen by us to reflect the requirements of our system.

* **Federated Learning**: A federated round can be scheduled on the server. Then the server will choose multiple clients and initiate their training stage. After they the round is finished, it aggregates all the models and saves it as the new global modal to be redistributed to all the clients.

* **Personalisation**: This allows the client to train on its own data so that it's own model becomes more accostumed to the behaviour of the user. So even if the global model would predict 10 hours of sleep as anomalous, the local one might see that as normal.

* **Dashboard**: Dashboard is the interface that a user can see the status of the system. It displays info about connected clients, it allows the scheduling of federation rounds, while also displaying any errors that appear on the server side.

## **Installation**

To see a guide on how to install and run the system, check this [Installation guide](documentation/INSTALLATION.md)

## **Usage**

For a brief description on how to interact with the system, check the [User guide](documentation/USER_GUIDE.md).

If you are interested in the available server APIs, their documentation can be found [here](documentation/API_DOCS.md).

## **Contributing**

Thank you for wanting to contribute to our project! If you have ideas or improvements that you want to share with us, please follow these guidelines:

- **Bug Reports and Feature Requests**: 
  - Submit clear bug reports with steps to reproduce or feature requests with detailed explanations to our repository. Please use the available template for this.
    
- **Code Contributions**: 
  - Fork the repository and create a branch for your changes.
  - Adhere to our coding style and conventions.
  - Before submiting your changes please run the pipeline locally. To see how the gitlab pipeline is run, check the [Pipeline YAML file](.gitlab-ci.yml)
  - Submit a pull request with a concise description of your changes. Please use the available template.


### *Naming Conventions*
Please follow the following guidelines for naming conventions.
* Commit Naming Conventions
    Commit messages should follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format:

    1. Start with one of the following prefixes: `docs:`, `fix:`, `feat:`, `refactor:`, or `test:`.
    2. Use a verb in the present form that describes the main addition in the commit, starting with a capital letter.
    3. Include one or more words that can describe the subject of the commit.

    For example: `docs: Add meeting notes`

* Issue Naming Conventions

    When creating an issue:

    1. Use a brief but descriptive title that summarizes the problem or task.
    2. Provide a detailed description of the issue or task, including any relevant background information, steps to reproduce the problem, or requirements for the task.
    3. Assign appropriate labels and milestones to the issue to help with organization and tracking.

* File Naming Conventions

    File names should adhere to the following conventions:

    1. Use lowercase letters and separate words with underscores ( `_` ).
    2. Start with a short, descriptive name that reflects the content of the file.
    3. End the file name with the appropriate file extension (e.g., `.py` for Python scripts, `.md` for Markdown files).


## **Contributors**

This project was created by five *Computer Science* students during the *Software Project* at *Delft University of Technology*. It was developed alongside [Maketek](https://www.maketek.nl/).

The students that worked on this project are:

- **Rafael Borges**
- **Alexandru-Sebastian Nechita**
- **Razvan Nistor**
- **Alexandru Preda**
- **Aksel Tacettin**


## **License**

This project is licensed under the MIT License. See the [LICENSE](documentation/LICENCE.md) file for details.

