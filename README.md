# Disaster Response Pipeline Project

### Introduction:
When a disaster happens it is helpful to know what kind of emergencies are happening.  Quickly categorizing news reports, texts, and calls helps can help the organizer quickly assess the situation and provide assistance.  The disaster reponse pipeline app gives a user interface to test new messages as it takes as input a message and categorizes it in one or more of the 36 categories.  

For example, if you typed "I felt multiple tremors", you would have the 2 categories "related" and "earthquake" returned.

![Alt text](Screenshot1.PNG?raw=true)

Also, included in the app are some visualizations of the training data used.

![Alt text](Screenshot2.PNG?raw=true)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
4. The project is also hosted on Heroku: https://mh-disasterresponse.herokuapp.com/

### Folders:
The web part of the project is in app.  run.py contains the 3 visualizations of the data.
Under the data folder are the csv files with initial data and a database file, DisasterReponse.db, with cleaned data.  process_data.py cleans the initial data.
In models, classifier is the trained model and train_classifier trained the model.  

.
+-- app
+-- Templates
|   +-- go.html 
|   +-- master.html
|   +-- run.py
+-- data
|   +-- DisasterReponse.db
|   +-- disaster_categories.csv
|   +-- disaster_messages.csv
|   +-- process_data.py
+-- models
|   +-- classifier.pkl
|   +-- train_classifier.py
    


