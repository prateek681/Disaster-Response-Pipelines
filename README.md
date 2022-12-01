# Disaster-Response-Pipelines

## Project Description

## Requirements

## Folder Descriptions
* App: This folder includes the run.py and the templates used for the web application
* Data: This folder includes the .csv, .db and the .py file
* process_data.py: This script utillizes the input csv files which contains the message dataand the message categoires (labels) and creates an SQLite Database which contains a merged and cleaned version of the data
* Models: This folder contains the pickle object and train_classifier.py file
* train_classifier.py: Script to tune a ML Model from a SQLite Database which was generated from process_data.py. Output of the file is the fitted model.
* Notebooks: This folder contains the Notebooks ETL Pipeline Preparation and ML Pipeline Preparation, which were used as a reference to generate the .py files for the Development of the Application [Note: This Folder and its files are not needed for the application to run]

## Project Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Application Images

The following images show the working of the Web Application

### Message Categorization

![Application Images](https://github.com/prateek681/Disaster-Response-Pipelines/blob/main/Screenshot-1.jpg)

### Training Dataset Overview


### Distribution of Categories in the Dataset


## Licensing and Acknowledgments

