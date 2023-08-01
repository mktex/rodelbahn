
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def get_clf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    labels_zuordnung = clf.classes_
    print("Beispiel:")
    print(clf.predict_proba(X_test))
    print("Labels:", labels_zuordnung)
    y_true = np.array(y_test)
    y_pred = np.array([labels_zuordnung[np.argmax(t)] for t in clf.predict_proba(X_test)])
    accuracy = (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred, labels=labels_zuordnung)
    print("Confusion Matrix:")
    print(cm)
    print("Präzision:", accuracy)
    return clf, labels_zuordnung
