import pandas as pd

from ..eda import dbeschreiben


def kategwert2col(xc):
    from string import punctuation
    for c in punctuation:
        xc = xc.replace(c, "")
    xc = "".join(list(map(lambda x: x[0].upper() + (x[1:].lower() if len(x) > 1 else ""),
                          filter(lambda y: y != "",
                                 map(lambda z: z.strip(), xc.split(" "))
                                 )
                          )
                      )
                 )
    return xc


def get_dummies(xdf_input, xcol, xkateg_werte):
    # TODO: spark einsetzen
    xliste = []
    for ik in range(xdf_input.shape[0]):
        xdatensatz_wert = xdf_input[xcol].iloc[ik]
        xliste.append(
            [int(str(t) in str(xdatensatz_wert)) for t in xkateg_werte]
        )
    xdf_res = pd.DataFrame(xliste)
    xdf_res.columns = xkateg_werte
    return xdf_res


def kateg2dummy(xdf_input, sep=None):
    """ Umwandelt die Spalten in Dummy-Variablen mit 1-Hot Encoding
        Falls in der kategorialen Felder mehrere Werte gleich eingesetzt wurden,
        dann kann ein Parameter zB sep=";" verwendet werden, um die voneinander zu trennen
        Beispiel:
            xdfcat_dummy = dbeschreiben.kateg2dummy(xdf_cat, sep=";")
    """
    xliste = []
    xdf = xdf_input.copy()
    xdf = xdf.reset_index(drop=True)

    def hatsep(xcolumn):
        if sep is not None:
            for t in xdf[xcolumn].values:
                if sep in str(t):
                    return True
        return False

    for xcol in xdf.columns:
        explode = False if sep is None else hatsep(xcol)
        if not explode:
            xdfres = pd.get_dummies(xdf[xcol])
        else:
            xkateg_werte = list(set(dbeschreiben.kateg_werte_liste(xdf_input, xcol, sep=sep)))
            xkateg_werte.sort()
            xdfres = get_dummies(xdf, xcol, xkateg_werte)
        xdfres.columns = [xcol + "_" + kategwert2col(xc) for xc in xdfres.columns]
        xliste.append(xdfres)
    xdfres_gesamt = None
    for i in range(len(xliste) - 1):
        if xdfres_gesamt is None:
            xdfres_gesamt = pd.merge(xliste[0], xliste[1], left_index=True, right_index=True)
        else:
            xdfres_gesamt = pd.merge(xdfres_gesamt, xliste[i + 1], left_index=True, right_index=True)
    xdfcat_dummy = xdfres_gesamt
    xdict_count = {}
    xliste_columns = list(xdfcat_dummy.columns)
    for i in range(len(xliste_columns)):
        xcol = xliste_columns[i]
        if xcol not in xdict_count.keys():
            xdict_count[xcol] = 0
        else:
            xdict_count[xcol] += 1
            xliste_columns[i] = "{}_{}".format(xliste_columns[i], xdict_count[xcol])
    xdfcat_dummy.columns = xliste_columns
    return xdfcat_dummy
