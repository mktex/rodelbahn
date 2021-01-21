import string

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from .detect import urls_in_string
from ..eingang.dq_csv import CSV

"""
from importlib import reload
from heimat.eingang import datenquellen, dq_csv; 
reload(datenquellen); reload(dq_csv)
from heimat.eingang.dq_csv import CSV

http://www.nltk.org/data.html
"""

# sind_alle_uppercase("DE") -> True
sind_alle_uppercase = lambda x: sum(list(map(lambda c: c.upper() == c, x))) == len(x)


class TXTVerarbeitung(BaseEstimator, TransformerMixin):
    DATA_DIR = "./data/"

    df = None

    tfidf_transformer = None
    vect_transformer = None

    def __init__(self, data_dir=None, _X=None, _y=None):
        if data_dir is not None:
            self.DATA_DIR = data_dir
        if _X is not None and _y is not None:
            self.setup(_X, _y)

    @staticmethod
    def startup():
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')

    def setup(self, X, y):
        self.X = X
        self.y = y

    def load_data_csv(self, xfname, encoding, sep, skiprows=0):
        df = CSV(self.DATA_DIR + xfname, encoding=encoding, sep=sep, skiprows=skiprows)
        df.lesen()
        self.set_df(df)

    def set_df(self, df):
        self.df = df

    def set_xy_by_features(self, xnamen, ynamen):
        self.X = np.array([str(t) for t in self.df.data[xnamen].values])
        self.y = self.df.data[ynamen].values

    def ersatz_url(self):
        X_neu = []
        for text in self.X:
            detected_urls = urls_in_string.get_urls(text)
            for url in detected_urls:
                text = text.replace(url, "placeholderurl")
            X_neu.append(text)
        self.X = X_neu

    @staticmethod
    def eigener_tokenizer(text):
        detected_urls = urls_in_string.get_urls(text)
        for url in detected_urls:
            text = text.replace(url, "placeholderurl")
        for c in string.punctuation:
            text = text.replace(c, " ")
        for c in "1234567890":
            text = text.replace(c, " ")
        text = list(map(lambda x: x.strip(), text.split(" ")))
        text = list(filter(lambda x: x != "", text))
        text = " ".join(text)
        tokens_liste = nltk.tokenize.word_tokenize(text)
        return tokens_liste

    @staticmethod
    def eigener_lemmatizer(tokens_liste):
        lemmatizer = WordNetLemmatizer()
        clean_tokens = []
        for tok in tokens_liste:
            clean_tok = tok
            pos_label = (pos_tag(word_tokenize(tok))[0][1][0]).lower()
            if pos_label == 'j': pos_label = 'a'
            if pos_label in ['r']:
                try:
                    lm = wordnet.synset(tok + '.r.1').lemmas()
                    pr = lm[0].pertainyms()
                    clean_tok = pr[0].name()
                except:
                    pass
            elif pos_label in ['a', 's', 'v']:
                clean_tok = lemmatizer.lemmatize(tok, pos=pos_label)
            else:
                clean_tok = lemmatizer.lemmatize(tok)
            clean_tok = clean_tok.lower().strip() if not sind_alle_uppercase(clean_tok) else clean_tok
            clean_tokens.append(clean_tok)
        return clean_tokens

    @staticmethod
    def textverarbeiter_en(text):
        tokens_liste = TXTVerarbeitung.eigener_tokenizer(text)
        stopwords_en = stopwords.words("english")
        tokens_liste = list(filter(lambda x: x not in stopwords_en, tokens_liste))
        clean_tokens = TXTVerarbeitung.eigener_lemmatizer(tokens_liste)
        return clean_tokens

    def count_tokens(self, sprache="english"):
        """
        :param sprache: Auswahl der Sprache, um den Tokenizer zu bestimmen
        :return: transformiertes Datenbestand X und Vector Objekt vect (erlaubt Zugriff zu .vocabulary_)
        """
        if sprache == "english":
            if self.vect_transformer is None:
                print("count_tokens(): learning")
                vect = CountVectorizer(tokenizer=self.textverarbeiter_en)
                vect_fitted = vect.fit(self.X)
            else:
                print("count_tokens(): apply")
                vect_fitted = self.vect_transformer
            self.X_transformed = vect_fitted.transform(self.X).toarray()
            self.vect_transformer = vect_fitted

    def tf_idf(self):
        if self.tfidf_transformer is None:
            print("tf_idf(): learning")
            transformer = TfidfTransformer(smooth_idf=False)
            transformer_fitted = transformer.fit(np.array(self.X_transformed))
        else:
            print("tf_idf(): apply")
            transformer_fitted = self.tfidf_transformer
        self.X_transformed = transformer_fitted.transform(self.X_transformed).toarray()
        self.tfidf_transformer = transformer_fitted

    def pipeline(self):
        self.count_tokens()
        self.tf_idf()

    def fit(self, X, y=None):
        self.tfidf_transformer = None
        self.vect_transformer = None
        self.X = X
        self.y = y
        self.pipeline()
        return self

    def transform(self, X, y=None):
        self.X = X
        self.y = y
        self.pipeline()
        return self.X_transformed

    def fit_transform(self, X, y=None, **fit_params):
        self.X = X
        self.y = y
        self.fit(X, y)
        return self.X_transformed



class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        tokenize = nltk.tokenize.word_tokenize
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        # TODO: nicht implementiert
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
