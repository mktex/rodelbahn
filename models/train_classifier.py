import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

sys.path.append(".")
from heimat.eingang.dq_datenbank import SQL
from heimat.nlp.textverarbeitung_nltk import TXTVerarbeitung

_pipe = None
_clf1 = None
_clf2 = None


def get_confusion_matrix_stats(cm, i):
    """
        Given a Confusion Matrix cm, calculates precision, recall and F1 scores
    :param cm: confusion matrix
    :param i: position of the variable, for with the caculation be done
    :return: three statistics: precision, recall and the F1-Score
    """
    tp = cm[i, i]
    fp = np.sum(cm[i, :]) - tp
    fn = np.sum(cm[:, i]) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def has_columns_with_one_value(y, target):
    columns_one_value = pd.DataFrame(y).apply(lambda x: len(set(x)))
    cols = columns_one_value[columns_one_value == 1].index.values.tolist()
    print("Target variables with one value:", [target[c] for c in cols])


# MLP
def check_mlp(clf1, x, y, target):
    labels_zuordnung_mlp = clf1.classes_
    beispiel_mlp_x = x
    beispiel_mlp_y = y[:, 0]
    y_true = np.array(beispiel_mlp_y)
    y_pred = np.array([labels_zuordnung_mlp[np.argmax(t)] for t in clf1.predict_proba(beispiel_mlp_x)])
    accuracy = (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred, labels=labels_zuordnung_mlp)
    if True:
        print("MLP ({})".format(target[0]))
        print("Labels:", labels_zuordnung_mlp)
        print("Confusion Matrix:")
        print(cm)
        for i in range(0, len(cm)):
            get_confusion_matrix_stats(cm, i)
        print("Präzision:", accuracy)


# MultiOutputClassifier(RandomForrestClassifier)
def check_moc(clf2, x, y, target):
    labels_zuordnung_moc = clf2.classes_
    beispiel_moc_x = x
    beispiel_moc_y = y[:, 1:]
    y_true = np.array(beispiel_moc_y)
    y_pred = np.array(clf2.predict(beispiel_moc_x))
    accuracy = (y_pred == y_true).mean()
    print("Präzision:", accuracy)
    for i in range(y_true.shape[1]):
        print("\n------------------------------------------------------------------------")
        categ_labels = labels_zuordnung_moc[i]
        print("Variable '{}', labels {}".format(target[i + 1], categ_labels))
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=labels_zuordnung_moc[i])
        print("Confusion matrix:")
        print(cm)
        for cat in categ_labels:
            i = list(categ_labels).index(cat)
            precision, recall, f1_score = get_confusion_matrix_stats(cm, i)
            print("Label {} \t Precision {} \t Recall {} \t F1-Score {}".format(
                cat,
                np.round(precision, 2), np.round(recall, 2),
                np.round(f1_score, 2)
            ))


def load_data(database_filepath):
    """
        - There was one variable "child_alone" that contains only one value, was removed
    :param database_filepath: e.g. "./data/DisasterResponse.db"
    :return: X (Training and Test data), y (36 target variables), target (names of target variables)
    """
    con = SQL(database_filepath)
    xdf = con.sqlite(tbl_name="msg_tbl")
    X = xdf.message.values
    # ['child_alone']
    target = ['genre', 'related', 'request', 'offer',
              'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
              'security', 'military', 'water', 'food', 'shelter',
              'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
              'infrastructure_related', 'transport', 'buildings', 'electricity',
              'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
              'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
              'other_weather', 'direct_report']
    y = xdf[target].values
    y = np.array([list(map(lambda x: str(x), t)) for t in y])
    has_columns_with_one_value(y, target)
    return X, y, target


def nlp_prep(X_train, y_train):
    """
        Performs a chain calculation to transform input text rows (X_train) into
        one Matrix representation after:
        - url replacements
        - text normalization through lowering, but keeping all-caps words such as 'US' as they are)
        - tokenization
        - normalization of tokens with a lemmatizer
        - removal of stop words
        - counts of remaining tokens and transformation to tf-idf representation
        - finally dimensionality reduction using PCA
    :param X_train: Input dataset, rows of text
    :param y_train: 36 dimensional target variable
    :return: X_transformed_train (Matrix representation) and target variables (to make sure sorting ist kept the same)
    """
    global _pipe
    pca = PCA(n_components=100)
    txt_train = TXTVerarbeitung(X_train, y_train)
    pipe = Pipeline([
        ('txt_v_nltk', txt_train),
        ('pca_output', pca)
    ])
    X_transformed_train = pipe.fit_transform(X_train)
    _pipe = pipe
    return X_transformed_train, y_train


def build_model(X_train, y_train):
    """
        Prepares 2 Models:
            - one MLP for the "genre" Variable
            - 35 Models of type MultiOutputClassifier(RandomForestClassifier); MOC(RFC)
        Text preparation uses a pipeline with a custom NLP chain (nlp_prep)
        Dimensionality reduction is done by the PCA step in NLP chain
        A grid search for best parameters of the MOC(RFC) Models
    :param X_train: input train data (just rows with text content)
    :param y_train: 36 variables
    :return:
    """
    global _clf1, _clf2
    print("\n NLP step .. ")
    X_transformed_train, y_transformed_train = nlp_prep(X_train, y_train)
    print("\n MLP classifier (feature 'genre') .. ")
    mlp = MLPClassifier(random_state=1, max_iter=300)
    clf1 = mlp.fit(X_transformed_train, y_train[:, 0])  # genre
    print("\n MOC(RFC) classifier (features 'related', 'request', 'offer', etc.) .. ")
    forest = RandomForestClassifier(n_estimators=5, random_state=1)
    parameters = {'estimator__n_estimators': [1, 10], 'estimator__max_depth': [5, 10]}
    clf2 = GridSearchCV(MultiOutputClassifier(forest), parameters)
    clf2.fit(X_transformed_train, y_train[:, 1:])  # 'related', 'request', 'offer', ...
    _clf1 = clf1
    _clf2 = clf2
    print("\n Models building done.")


def evaluate_model(X_test, y_test, target):
    global _pipe, _clf1, _clf2
    X_transformed_test = _pipe.transform(X_test)
    # Performance of MLP for 'genre'
    print("===========================================================================")
    print("Performance for the MLP Model (target 'genre'):")
    check_mlp(_clf1, X_transformed_test, y_test, target)
    input("Enter to continue .. ")
    print("===========================================================================")
    print("Performance for the MOC(RFC) Model (targets 'related', 'request', 'offer', etc.):")
    # Performance of MOC rest of target variables
    check_moc(_clf2, X_transformed_test, y_test, target)


def save_model(model_filepath):
    global _pipe, _clf1, _clf2
    rodelbahn_model = {
        "pipe": _pipe,
        "clf1": _clf1,
        "clf2": _clf2
    }
    with open(model_filepath, 'wb') as f:
        pickle.dump(rodelbahn_model, f)
    print("Model successfully saved under {}".format(model_filepath))


def main():
    if len(sys.argv) == 3:
        # database_filepath = "./data/DisasterResponse.db"
        # model_filepath = "./data/clf.pckl"
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    '
              'DATABASE: {}'.format(database_filepath))
        X, y, target = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

        print('\nTrain models MLPClassifier, MultiOutputClassifier(RandomForestClassifier) .. ')
        build_model(X_train, y_train)

        print('\nEvaluating model...')
        evaluate_model(X_test, y_test, target)

        print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. '
              '\n\nExample: '
              'python ./models/train_classifier.py ./data/DisasterResponse.db ./data/rodelbahn_model.pckl')


"""
python ./models/train_classifier.py ./data/DisasterResponse.db ./data/rodelbahn_model.pckl
"""
if __name__ == '__main__':
    main()
