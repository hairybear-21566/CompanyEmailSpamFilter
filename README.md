# Naive Bayes Spam Email Filter

This project demonstrates the implementation of a **Naive Bayes** classifier to filter out spam emails. The model is trained on a real-world email dataset provided by an American company, consisting of both legitimate emails and spam messages. The goal is to build a classifier that can accurately identify spam based on email content.

## Key Features

- **Data Preprocessing:** 
  - The dataset contains emails, with some labeled as "spam" and others as "not spam."
  - Emails are cleaned by removing punctuation, common words, and spaces to reduce noise in the data.
  
- **Naive Bayes Model:**
  - The classifier is based on the **Naive Bayes** algorithm, which is particularly well-suited for text classification tasks like spam detection.
  - The algorithm assumes conditional independence between words and uses probability to predict the likelihood that an email is spam.

- **Performance Tuning:** 
  - Various probability equations were tested to maximize the filter's accuracy.
  - The final model is evaluated using accuracy, precision, recall, and F1-score metrics.

## Technologies and Libraries Used

- **Python**: Main programming language.
- **Scikit-learn**: For the Naive Bayes implementation and performance evaluation.
- **Pandas**: For data manipulation and preprocessing.
- **NLTK**: For text preprocessing, such as tokenization and stopword removal.

## How It Works

1. **Dataset Preparation**: The dataset is split into training and testing sets. Preprocessing steps like tokenization, removing stopwords, and stemming are applied to standardize the text data.
2. **Training**: The Naive Bayes classifier is trained on the cleaned training dataset, learning to distinguish between spam and legitimate emails.
3. **Prediction**: After training, the model predicts whether new, unseen emails are spam or not.
4. **Evaluation**: The model's performance is evaluated using the test dataset. Key metrics like **accuracy**, **precision**, **recall**, and **F1-score** are calculated.

## Future Improvements

- **Advanced Preprocessing**: Incorporate more sophisticated text preprocessing techniques such as lemmatization and n-grams to capture context better.
- **Feature Engineering**: Add more features like email subject line analysis, email sender domain, and HTML content analysis.
- **Ensemble Methods**: Explore combining Naive Bayes with other models (e.g., SVM) to improve overall performance.

---

This project showcases the power of Naive Bayes in spam detection. Feel free to experiment with the dataset and adjust the model for even better results.
