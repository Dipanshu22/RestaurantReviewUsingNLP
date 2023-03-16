import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t' , quoting = 3)

# Clean text
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Create bag of words model
cv = CountVectorizer(max_features = 600)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fit Naive Bayes to the training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict test set results
y_pred = classifier.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Count positive and negative reviews
positive_reviews = df[df['Liked'] == 1].count()[0]
negative_reviews = df[df['Liked'] == 0].count()[0]

# Create bar chart
labels = ['Positive', 'Negative']
values = [positive_reviews, negative_reviews]
plt.bar(labels, values)
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.title('Restaurant Review Analysis')

# Show bar chart
plt.show()
