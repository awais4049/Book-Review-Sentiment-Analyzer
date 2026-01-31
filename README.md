# Review Sentiment Classifier

A machine learning project that classifies book reviews into positive or negative sentiment using Natural Language Processing (NLP) techniques and various classification algorithms.

## 📋 Project Overview

This project implements a sentiment analysis system that:
- Analyzes Amazon book reviews from a dataset of 10,000 reviews
- Classifies reviews as POSITIVE, NEGATIVE, or NEUTRAL based on rating scores
- Uses TF-IDF vectorization to convert text into numerical features
- Compares multiple machine learning models to find the best classifier
- Achieves ~82% accuracy on test data

## 🎯 Features

- **Multiple ML Models**: Implements and compares SVM, Decision Tree, Naive Bayes, and Logistic Regression
- **Class Balancing**: Handles imbalanced datasets by downsampling the majority class
- **Hyperparameter Tuning**: Uses GridSearchCV to optimize model parameters
- **Model Persistence**: Saves trained models for future use without retraining
- **Comprehensive Evaluation**: Includes accuracy, F1-score, and cross-validation metrics

## 🛠️ Technologies Used

- **Python 3.x**
- **scikit-learn** - Machine learning algorithms and tools
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation (implicit)
- **JSON** - Data loading

## 📊 Dataset

The project uses the **Amazon Book Reviews** dataset (`Books_small_10000.json`), which contains:
- Review text
- Overall rating (1-5 stars)
- 10,000 total reviews

**Sentiment Mapping:**
- Ratings 1-2: NEGATIVE
- Rating 3: NEUTRAL
- Ratings 4-5: POSITIVE

## 🚀 Getting Started

### Prerequisites

```bash
pip install scikit-learn numpy
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/review-sentiment-classifier.git
cd review-sentiment-classifier
```

2. Ensure you have the dataset file `Books_small_10000.json` in the project directory

3. Open the Jupyter notebook:
```bash
jupyter notebook Review_Sentiment_Classifier.ipynb
```

## 💻 Usage

### Running the Notebook

1. **Load and Prepare Data**: The notebook automatically loads reviews from the JSON file and splits them into training/test sets

2. **Train Models**: Run all cells to train multiple classifiers:
   - Support Vector Machine (SVM)
   - Decision Tree
   - Naive Bayes
   - Logistic Regression

3. **Evaluate Performance**: Compare model accuracies and F1-scores

4. **Make Predictions**: Test the model on custom review text:
```python
test_reviews = ['Amazing book, highly recommend!', 'Terrible waste of money']
new_test = vectorizer.transform(test_reviews)
predictions = clf.predict(new_test)
```

### Using the Saved Model

After training, the model is saved as `sentiment_classifier.pkl`. Load it for future predictions:

```python
import pickle

# Load the trained model
with open('sentiment_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(new_reviews_vectorized)
```

## 📈 Results

| Model | Test Accuracy | F1-Score |
|-------|--------------|----------|
| SVM (Linear) | 80.77% | 0.8077 |
| Decision Tree | 63.22% | 0.6307 |
| Naive Bayes | 82.45% | 0.8234 |
| Logistic Regression | 80.77% | 0.8077 |
| **SVM (Tuned)** | **81.97%** | **~0.82** |

The tuned SVM with `C=4` and `rbf` kernel achieved the best performance.

## 🔍 Project Structure

```
review-sentiment-classifier/
│
├── Review_Sentiment_Classifier.ipynb    # Main Jupyter notebook
├── Books_small_10000.json               # Dataset (not included)
├── sentiment_classifier.pkl             # Trained model (generated)
└── README.md                            # This file
```

## 🧠 How It Works

1. **Data Loading**: Reviews are loaded from JSON and converted to Review objects with automatic sentiment labeling

2. **Class Balancing**: The dataset is balanced by matching positive and negative sample counts to prevent bias

3. **Text Vectorization**: TF-IDF converts review text into numerical feature vectors that capture word importance

4. **Model Training**: Multiple classifiers are trained and evaluated on the balanced dataset

5. **Hyperparameter Tuning**: GridSearchCV finds optimal parameters through 5-fold cross-validation

6. **Evaluation**: Models are compared using accuracy and F1-score metrics

## 🔧 Customization

### Adjusting the Train-Test Split
```python
training, test = train_test_split(reviews, test_size=0.33, random_state=42)
# Change test_size to adjust the split ratio
```

### Trying Different Vectorization
```python
# Use CountVectorizer instead of TF-IDF
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
```

### Adding More Hyperparameters
```python
parameters = {
    'C': (1, 4, 8, 16, 32, 64),
    'kernel': ('linear', 'rbf', 'poly'),
    'gamma': ('scale', 'auto')
}
```

## 📝 Code Highlights

### Custom Review Class
```python
class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
    
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE
```

### Balanced Dataset Creation
```python
def evenly_distribute(self):
    negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
    positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
    positive_shrunk = positive[:len(negative)]
    self.reviews = negative + positive_shrunk
    random.shuffle(self.reviews)
```

## 🚦 Future Improvements

- [ ] Include NEUTRAL sentiment in classification (currently excluded)
- [ ] Implement deep learning models (LSTM, BERT, Transformers)
- [ ] Use Word2Vec or GloVe embeddings instead of TF-IDF
- [ ] Collect and train on larger datasets
- [ ] Add sentiment intensity scoring (not just binary classification)
- [ ] Create a web interface for real-time predictions
- [ ] Add visualization of model decision boundaries
- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Amazon Review Dataset providers
- scikit-learn documentation and community
- Various online tutorials and resources on sentiment analysis

## 📞 Contact

If you have any questions or suggestions, feel free to:
- Open an issue on GitHub
- Reach out via email: your.email@example.com

---

**Note**: This project was created for educational purposes to demonstrate sentiment analysis using machine learning techniques.

## 🔗 Related Resources

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Natural Language Processing with Python](https://www.nltk.org/book/)
- [Understanding TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Support Vector Machines Explained](https://scikit-learn.org/stable/modules/svm.html)
