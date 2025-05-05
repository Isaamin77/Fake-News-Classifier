# Fake-News-Classifier
This is a machine learning based fake news detection system that classifies news articles as real or fake using Natural Language Processing. It has a user interface which you can interact with and get your news articles classified.
I used a prelabelled datasets from kaggle containing fake and real news articles to train the model.

Within this repository I uploaded my new and updated IDLE code which contains my whole project and working interface.

Due to Github's file size limit I was unable to upload the datasets however you can find and download them from this google drive folder: https://drive.google.com/drive/folders/1xKxJgHnS_PiqWQpHG2ApbwPhHclD2d-n?usp=drive_link 

After downloading the files you would need to run them in this order:
"combine_and_preprocess"
"model_training"
"app"

The other two IDLE files are my process of dealing with the original 2 datasets before adding the third one.

After running the 3 files open your command prompt and navigate to the downloaded folder containing all of the files and then run the command:
streamlit run app.py

Incase you do not have the required packages please run this code first:

pip install streamlit

pip install scikit-learn

pip install pandas

pip install nltk


