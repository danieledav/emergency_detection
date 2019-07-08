# Using Social Media to Enhance Emergency Situation Awareness
Project for the **Web Information Retrieval** course at La Sapienza, Università di Roma held by Prof. Andrea Vitaletti and Prof. Luca Becchetti.

The project wants to detect an emergency situation in real time through tweets flow scanning using machine learning algorithms and users as sensors.

Link to the [Slideshare presentation](https://www.slideshare.net/DanieleDavoli/using-social-media-to-enhance-emergency-situation-awareness/DanieleDavoli/using-social-media-to-enhance-emergency-situation-awareness). Scientific paper on the repository.

## Authors
- Danilo Marzilli - [Linkedin profile](https://www.linkedin.com/in/danilomarzilli/)
- Andrea Lombardo - [Linkedin profile](https://www.linkedin.com/in/andrea-lombardo-2103ba15a/)
- Daniele Davoli - [Linkedin profile](https://www.linkedin.com/in/danieledavoli/)

## Dataset
For training and validating our machine learning system, we have used a [dataset](http://socialsensing.it/en/datasets) of 5,642 manually annotated tweets in the Italian language.
The tweets are related to 4 different natural disasters occurred in Italy between 2009 and 2014. For each tweet is reported:

1. tweet ID;
2. text;
3. source;
4. author’s screen name;
5. author’s ID;
6. latitude and longitude (if available);
7. time;
8. disaster ID (see below);
9. class.

Tweets have been manually annotated by humans and divided among 3 classes according to the information they convey:

- **damage class:** tweets related to the disaster and carrying information about damages to the infrastructures or on the population;
- **no damage class:** tweets related to the disaster but not carrying relevant information for the assessment of damages;
- **not relevant class:** tweets collected while building the dataset, but not related to any disaster (noise).

The dataset ins also available in this repository. Validations with your datasets are welcome :)

## Data pipeline
We process our dataset in this order:
1. **Import data** from the .csv file;
2. **Preprocessing** our tweets in order to remove punctuation, stop words and digits and to implement the stemming algorithm;
3. **Trasform the tweets in vectors** in a space vector where the axis are the vocabulary terms and give to each vector a TF-IDF (Term Frequency and Inverted Document Frequency) weight;
4. **Cluster our tweets** (now vectors) in main topics;
5. **Train a SVM classifier** in order to distinguish the tweets in relevant and not relevant.
