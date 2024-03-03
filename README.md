Sentiment Analysis using Random Forest Model
Overview:
This project involves performing sentiment analysis on Twitter and Reddit data using a Random Forest classifier. Sentiment analysis aims to determine the sentiment (positive, negative, or neutral) conveyed in textual data. The Random Forest model is a popular ensemble learning technique that operates by constructing a multitude of decision trees during training and outputting the mode of the classes (classification) or the mean prediction (regression) of the individual trees.

Key Components:
Data Collection: The project involves the collection of data from Twitter and Reddit platforms. Twitter and Reddit data are commonly used for sentiment analysis tasks due to their vast user-generated content.

Data Preprocessing: Before feeding the data into the model, preprocessing is necessary. This involves steps such as removing stopwords, lemmatization, tokenization, and converting the text into a format suitable for machine learning models.

Model Training: The Random Forest classifier is trained on the preprocessed data. During training, the model learns patterns and relationships between the input features (text data) and the target variable (sentiment).

Model Evaluation: The trained model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Additionally, a confusion matrix is generated to visualize the model's performance across different sentiment classes.

Deployment and Testing: The trained model can be deployed and tested on new data to make predictions about the sentiment conveyed in unseen text.

Code Structure:
Data Loading: Twitter and Reddit data are loaded from CSV files.
Data Preprocessing: Text preprocessing functions are defined to clean and prepare the text data.
Data Splitting: The data is split into training and testing sets for both Twitter and Reddit datasets.
Model Training: A Random Forest classifier is trained on the combined training data from Twitter and Reddit.
Model Evaluation: The trained model is evaluated on a subset of the combined testing data and optionally on the full testing dataset.
Visualization: A bar plot is generated to visualize the distribution of sentiment across Twitter, Reddit, and combined datasets.
Usage:
To use this project, follow these steps:

Ensure that the required libraries (pandas, nltk, sklearn) are installed.
Download Twitter and Reddit datasets and place them in the specified directory.
Run the provided code to preprocess the data, train the model, and evaluate its performance.
Optionally, apply the trained model to the full dataset for comprehensive evaluation.
Dependencies:
Python 3.x
pandas
nltk
scikit-learn
matplotlib
Future Improvements:
Implement more advanced text preprocessing techniques.
Experiment with different machine learning algorithms for sentiment analysis.
Explore deep learning models such as LSTM or BERT for improved performance.
Develop a web application or API for real-time sentiment analysis.
Contributors:
Nikhil Chezian
