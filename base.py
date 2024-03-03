########### Sentiment analysis using Random Forest Model ###########

# Importing necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(tokens)

# Loading Twitter Data
twitter_data = pd.read_csv("C:/Users/nchez/Desktop/CP_Sentiment/archive/Twitter_Data.csv", encoding="utf-8")
twitter_data = twitter_data.dropna(subset=['category'])
twitter_data['clean_text'] = twitter_data['clean_text'].apply(preprocess_text)

# Splitting Twitter Data into training and testing sets
X_twitter = twitter_data['clean_text']
y_twitter = twitter_data['category']
X_train_twitter, X_test_twitter, y_train_twitter, y_test_twitter = train_test_split(X_twitter, y_twitter, test_size=0.3, random_state=42)

# Loading Reddit Data
reddit_data = pd.read_csv("C:/Users/nchez/Desktop/CP_Sentiment/archive/Reddit_Data.csv", encoding="utf-8")
reddit_data = reddit_data.dropna(subset=['category'])
reddit_data['clean_text'] = reddit_data['clean_text'].apply(preprocess_text)

# Splitting Reddit Data into training and testing sets
X_reddit = reddit_data['clean_text']
y_reddit = reddit_data['category']
X_train_reddit, X_test_reddit, y_train_reddit, y_test_reddit = train_test_split(X_reddit, y_reddit, test_size=0.3, random_state=42)

# Merging Twitter and Reddit Data
X_train_combined = pd.concat([X_train_twitter, X_train_reddit], ignore_index=True)
y_train_combined = pd.concat([y_train_twitter, y_train_reddit], ignore_index=True)

X_test_combined = pd.concat([X_test_twitter, X_test_reddit], ignore_index=True)
y_test_combined = pd.concat([y_test_twitter, y_test_reddit], ignore_index=True)

# Sentiment distribution plot
plt.figure(figsize=(8, 6))
twitter_sentiments = twitter_data['category'].value_counts()
reddit_sentiments = reddit_data['category'].value_counts()
combined_sentiments = y_train_combined.value_counts()

plt.bar(twitter_sentiments.index, twitter_sentiments.values, alpha=0.5, label='Twitter Data')
plt.bar(reddit_sentiments.index, reddit_sentiments.values, alpha=0.5, label='Reddit Data')
plt.bar(combined_sentiments.index, combined_sentiments.values, alpha=0.5, label='Combined Data')

plt.title('Sentiment Distributions')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.legend()
plt.show()

# Vectorizing the text data using CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X_train_combined_vec = vectorizer.fit_transform(X_train_combined)
X_test_combined_vec = vectorizer.transform(X_test_combined)

# Creating a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Running the classifier on the subset with adjustable size for increased speed of execution
subset_size = 10000  
X_train_subset = X_train_combined_vec[:subset_size]
y_train_subset = y_train_combined[:subset_size]

rf_classifier.fit(X_train_subset, y_train_subset)
y_pred_combined_subset = rf_classifier.predict(X_test_combined_vec)

# Evaluating the subset dataset
accuracy_combined_subset = accuracy_score(y_test_combined, y_pred_combined_subset)
print(f"Subset Dataset Accuracy: {accuracy_combined_subset:.2f}")

report_combined_subset = classification_report(y_test_combined, y_pred_combined_subset)
print("Subset Dataset Classification Report:\n", report_combined_subset)

# Calculating and printing the confusion matrix for the subset dataset
confusion_matrix_subset = confusion_matrix(y_test_combined, y_pred_combined_subset)
print("Subset Dataset Confusion Matrix:\n", confusion_matrix_subset)

# Extending option to the whole dataset
apply_to_full_dataset = input("Do you want to apply it to the full dataset? (yes/no): ").strip().lower()
if apply_to_full_dataset == 'yes':
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_combined_vec, y_train_combined)
    y_pred_combined = rf_classifier.predict(X_test_combined_vec)
    
    # Evaluating the full dataset
    accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)
    print(f"Full Dataset Accuracy: {accuracy_combined:.2f}")
    
    report_combined = classification_report(y_test_combined, y_pred_combined)
    print("Full Dataset Classification Report:\n", report_combined)

    # Calculating and printing the confusion matrix for the full dataset
    confusion_matrix_full = confusion_matrix(y_test_combined, y_pred_combined)
    print("Full Dataset Confusion Matrix:\n", confusion_matrix_full)
