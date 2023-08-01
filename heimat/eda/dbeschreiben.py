from contextlib import contextmanager
from functools import reduce

import numpy as np
import pandas as pd


@contextmanager
def halt_pandas_warnings():
    with pd.option_context("mode.chained_assignment", None):
        yield


# Anzahl oder Prozente der Nullen in einer Spalte
def nullen(df, xcol): return df[df[xcol].isnull()].shape[0]


def prozent_nullen(df, xcol): return df[df[xcol].isnull()].shape[0] / df.shape[0]


def nullen_in_df(xdf): return pd.DataFrame([(xcol, nullen(xdf, xcol)) for xcol in xdf.columns])


def prozent_nullen_in_df(xdf): return pd.DataFrame([(xcol, prozent_nullen(xdf, xcol)) for xcol in xdf.columns])


# Ergibt die Durchschnittswerte einer kontinuierlicher Variable pro Wert aus einer Kategorialer Variable
def durchschnittswerte_pro_kateg(df, xkateg, xwert): return df[[xkateg, xwert]]\
                                                                .groupby(xkateg).mean().sort_values(by=xwert)


def kateg_werte_liste(xdf_input, xcol, sep=None):
    """ Ergibt die Liste der Werte in kategorialer Variable
        Bleibt der sep None, dann sind die Werte nichts anderes als ein set(xL)
    """
    xliste = list(map(lambda x: str(x), xdf_input[xcol].values.tolist()))
    xliste = list(filter(lambda x: x is not np.nan, xliste))
    if sep is not None:
        xliste = list(map(lambda x: list(set(x.split(sep))), xliste))
        xliste = list(reduce(lambda a, b: a + b, xliste))
    xliste = [str(t).strip() for t in xliste]
    return xliste


def frequenz_werte(xdf_input, xcol="CousinEducation", prozente=False, sep=None):
    """ Anzahl der Datensätze mit xcol == {Wert} ODER Wert in xcol.split(sep)
    """
    df = xdf_input.copy()
    xl = kateg_werte_liste(df, xcol, sep=sep)
    xs = pd.DataFrame({xcol: xl})
    xs['id'] = xs.index.values.tolist()
    xres = xs.groupby(xcol, as_index=True).count()
    xres = xres.sort_values(by="id", ascending=False)
    if prozente:
        xres['id'] = [t / xdf_input.shape[0] for t in xres['id'].values.tolist()]
    return xres


def num_cat(xdf_input):
    """ Spalte den Datenbestand vertikal in kategoriale und numerische Teile
        Beispiel:
            xdf_num, xdf_cat = dbeschreiben.num_cat(dfres)
    """
    xdf_cat = xdf_input.select_dtypes(include=['object']).copy()
    xdf_num = xdf_input[list(filter(lambda xc: xc not in xdf_cat.columns, xdf_input.columns))]
    print("\nKategoriale Felder: {}".format(xdf_cat.columns))
    print("\nNumerische Felder: {}".format(xdf_num.columns))
    return xdf_num, xdf_cat
