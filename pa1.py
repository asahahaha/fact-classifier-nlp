import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# Download NLTK resources if not already done
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
def load_data(facts_file='facts.txt', fakes_file='fakes.txt'):
    with open(facts_file, 'r', encoding='utf-8') as f:
        facts = f.readlines()
    with open(fakes_file, 'r', encoding='utf-8') as f:
        fakes = f.readlines()
    
    data = facts + fakes
    # label data as binary: 1 for facts, 0 for fakes
    labels = [1] * len(facts) + [0] * len(fakes) 
    return data, labels

# Lowercasing, Punctuation Removal, Stop Word Removal, Lemmatization, and Stemming
def preprocess_data(text):
    # Initialize lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text)  # Tokenize the text
    
    # Lowercase, remove punctuation, stop words, lemmatize, and stem
    filtered_tokens = [
        stemmer.stem(lemmatizer.lemmatize(token.lower())) 
        for token in tokens 
        if token.lower() not in stop_words and token.isalpha()
    ]
    
    return ' '.join(filtered_tokens)  # Join back to string for vectorization

# Preprocess data (e.g., TF-IDF feature extraction)
def feature_extract_data(data):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(data)
    return X, vectorizer

# Split data into train/dev/test sets
def split_data(X, y):
    # 16% for validation, 74% for training, 10% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
    return X_train, X_val, X_test, y_train, y_val, y_test

def split_data_70_15_15(X, y):
    # First split: 70% training and 30% temporary (for validation and test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=100)

    # Second split: 50% of temporary set for validation and 50% for test (15% each of the original data)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=100)
    return X_train, X_val, X_test, y_train, y_val, y_test

def split_data_60_20_20(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=100)

    # Second split: 50% of temporary set for validation and 50% for test (15% each of the original data)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=100)
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data_80_10_10(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=100)

    # Second split: 50% of temporary set for validation and 50% for test (15% each of the original data)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=100)
    return X_train, X_val, X_test, y_train, y_val, y_test



# Train Naive Bayes with hyperparameter tuning
def train_naive_bayes(X_train, y_train, X_val, y_val):
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
    best_alpha = None
    best_accuracy = 0
    best_model = None

    for alpha in alphas:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Validate the model
        y_val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Alpha: {alpha}, Validation Accuracy: {accuracy}")
        
        # Check if this is the best performing alpha
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
            best_model = model

    print(f"Best alpha: {best_alpha} with Validation Accuracy: {best_accuracy}")
    
    return best_model, best_alpha  # Return the model with the best alpha

# Train Logistic Regression with hyperparameter tuning
def train_logistic_regression(X_train, y_train, X_val, y_val):
    C_values = [0.1, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0]
    penalties = ['l1', 'l2']
    

    best_C = None
    best_penalty = None
    best_accuracy = 0
    best_model = None

    for C in C_values:
        for penalty in penalties:
            try:
                model = LogisticRegression(C=C, penalty=penalty, max_iter=1000, solver='saga')
                model.fit(X_train, y_train)

                # Validate the model
                y_val_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_val_pred)
                print(f"C: {C}, Penalty: {penalty}, Validation Accuracy: {accuracy}")

                # Check if this is the best performing combination
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_C = C
                    best_penalty = penalty
                    best_model = model
            except Exception as e:
                # Handle exceptions for incompatible parameter combinations
                print(f"Error with C: {C}, Penalty: {penalty}: {e}")

    print(f"Best C: {best_C}, Best Penalty: {best_penalty} with Validation Accuracy: {best_accuracy}")

    return best_model, best_C, best_penalty  # Return the model with the best parameters

# Train SVM with hyperparameter tuning
def train_svm(X_train, y_train, X_val, y_val):
    C_values = [0.1, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    best_C = None
    best_kernel = None
    best_accuracy = 0
    best_model = None

    for C in C_values:
        for kernel in kernels:
            model = SVC(C=C, kernel=kernel)
            model.fit(X_train, y_train)
            
            # Validate the model
            y_val_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_val_pred)
            print(f"C: {C}, Kernel: {kernel}, Validation Accuracy: {accuracy}")
            
            # Check if this is the best performing combination
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_C = C
                best_kernel = kernel
                best_model = model

    print(f"Best C: {best_C}, Kernel: {best_kernel} with Validation Accuracy: {best_accuracy}")
    
    return best_model, best_C, best_kernel # Return the model with the best parameters

# Evaluate the model on the test set
def evaluate_model(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))
    return acc
def main(description=''):
    # Step 1: Load and preprocess data
    data, labels = load_data()
    #data_preprocessed = data
    data_preprocessed = [preprocess_data(line) for line in data]
    X, vectorizer = feature_extract_data(data_preprocessed)
    
    # Step 2: Split data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_70_15_15(X, labels)
    
    # Step 3a: Train Naive Bayes with hyperparameter tuning
    print("\nTraining Naive Bayes...")
    best_nb_model, best_alpha = train_naive_bayes(X_train, y_train, X_val, y_val)
    
    # Step 3b: Train Logistic Regression with hyperparameter tuning
    print("\nTraining Logistic Regression...")
    best_lr_model, lr_best_C, best_penalty = train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # Step 3c: Train SVM with hyperparameter tuning
    print("\nTraining SVM...")
    best_svm_model, svm_best_C, best_kernel = train_svm(X_train, y_train, X_val, y_val)
    
    # Step 4: Evaluate all models on the test set
    print("\nEvaluating Naive Bayes...")
    nb_acc = evaluate_model(best_nb_model, X_test, y_test)

    print("\nEvaluating Logistic Regression...")
    lr_acc = evaluate_model(best_lr_model, X_test, y_test)

    print("\nEvaluating SVM...")
    svm_acc = evaluate_model(best_svm_model, X_test, y_test)
    
    '''
    # Save the experiment results
    data = {
        'description': [description],
        'nb_best_alpha' : [best_alpha],
        'lr_best_C' : [lr_best_C],
        'lr_best_penalty' : [best_penalty],
        'svm_best_C' : [svm_best_C],
        'svm_best_kernel' : [best_kernel],
        'naive bayes accuracy': [nb_acc],
        'logistic regresion accuracy': [lr_acc],
        'svm accuracy': [svm_acc]
    }
    df = pd.DataFrame(data)
    df.to_csv('history.csv', mode='a', header=True, index=False)
    '''

main('')
