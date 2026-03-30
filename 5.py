from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

texts = ["I love this movie", "Amazing film",
         "I hate this movie", "Terrible film"]

labels = [1, 1, 0, 0]   # 1 = positive, 0 = negative

# Convert text → numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Discriminative
lr = LogisticRegression()
lr.fit(X, labels)

# Generative
nb = MultinomialNB()
nb.fit(X, labels)

test = ["I love this", "Bad movie"]
X_test = vectorizer.transform(test)

# Predictions
print("Logistic Regression (Discriminative):")
print(lr.predict(X_test))

print("\nNaive Bayes (Generative):")
print(nb.predict(X_test))
