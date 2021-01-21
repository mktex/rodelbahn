import numpy as np
import pandas as pd
from sklearn import cluster

from eda import dbeschreiben

xfErsatzNullWerteDurchMean = lambda xdf_in: xdf_in.fillna(value=xdf_in.mean())
xfErsatzNullWerteDurchMedian = lambda xdf_in: xdf_in.fillna(value=xdf_in.median())


def datenbestand_spalten(xdf_input, xcol):
    """ Entsprechend Nullwerte in xcol, spalte den Datenbestand in 2 Gruppen
    """
    xdf_temp = xdf_input.copy()
    xdf_temp[xcol + "_istNull"] = [np.isnan(t) if 'float' in str(type(t)) else False for t in
                                   xdf_temp[xcol].values.tolist()]
    print(xdf_temp[[xcol, xcol + "_istNull"]])

    print(type(xdf_temp[xcol + "_istNull"].values[3]), xdf_temp[xcol + "_istNull"].values[3] is False)
    xdf_vorhanden = xdf_temp[list(map(lambda x: bool(x) is False, xdf_temp[xcol + "_istNull"].values))]
    xdf_fehlend = xdf_temp[list(map(lambda x: bool(x) is True, xdf_temp[xcol + "_istNull"].values))]

    print("[x] Unterschied zwischen den zwei Gruppen:")

    print("\n Gruppe ohne fehlenden Daten (Gruppe A) in {}".format(xcol))
    print(xdf_vorhanden.describe())

    print("\n Gruppe mit fehlenden Daten (Gruppe B) in {}".format(xcol))
    print(xdf_fehlend.describe())

    print("[x] Prozent der Nullwerte pro Feature:")
    xcols_cluster = xdf_vorhanden.describe().columns.tolist()
    xnullwerte_a = [(xc, dbeschreiben.xProzentNullen(xdf_vorhanden, xc)) for xc in xcols_cluster]
    xnullwerte_a = pd.Series(list(map(lambda x: x[1], xnullwerte_a)),
                             index=list(map(lambda x: x[0], xnullwerte_a))).to_dict()
    xnullwerte_b = [(xc, dbeschreiben.xProzentNullen(xdf_fehlend, xc)) for xc in xcols_cluster]
    xnullwerte_b = pd.Series(list(map(lambda x: x[1], xnullwerte_b)),
                             index=list(map(lambda x: x[0], xnullwerte_b))).to_dict()
    print("\n Gruppe A:")
    print(xnullwerte_a)
    print("\n Gruppe B:")
    print(xnullwerte_b)
    # TODO: zeige noch ein DT, entsprechend binärer Variable
    return xdf_vorhanden, xdf_fehlend, xnullwerte_a, xnullwerte_b


def ersatz_mit_knn(xdf_input, xcol="CareerSatisfaction", ignoriere_spalten=("id"),
                   threshold_null_werte_cluster=0.7, nclust=3):
    """ Für jede Zeile, die NAN Werte hat, findet dijenigen Datensätze,
        die in einem Cluster zusammengehören
        threshold_null_werte_cluster:
            Feature für Clustering nicht verwenden, wenn die Nullwerte mehr als so viel Prozent betragen

    """
    print("=============================================================================================")
    xdf = xdf_input[list(filter(lambda x: x not in ignoriere_spalten, xdf_input.columns))].copy()
    if '__cluster__' in xdf.columns:
        print("[x] Intern wird __cluster__ Feld verwendet!")
        return

    xdf_vorhanden, xdf_fehlend, xnwa, xnwb = datenbestand_spalten(xdf, xcol)
    xcols_cluster = xdf_vorhanden.describe().columns.tolist()
    xcols_cluster = list(filter(lambda x: x != xcol, xcols_cluster))
    xcols_cluster = list(
        filter(lambda x: xnwa[x] <= threshold_null_werte_cluster and xnwb[x] <= threshold_null_werte_cluster,
               xcols_cluster))
    # xcols_cluster.sort()
    np.random.seed()
    np.random.shuffle(xcols_cluster)

    xdf_vorhanden = xfErsatzNullWerteDurchMean(xdf_vorhanden[xcols_cluster])
    xdf_fehlend = xfErsatzNullWerteDurchMean(xdf_fehlend[xcols_cluster])

    # TODO: Übergabe an xKMeans
    anzahl_cluster = nclust
    clf = cluster.KMeans(n_clusters=anzahl_cluster)  # random_state=42
    clf.fit(xdf_vorhanden[xcols_cluster].values)
    xdf_vorhanden['__cluster__'] = clf.labels_

    print("\n[x] Beispiel Resultat:")
    print(xdf_vorhanden[xcols_cluster + ["__cluster__"]].sample(20))
    xdf_clustering_data = xfErsatzNullWerteDurchMean(xdf[xcols_cluster])
    xdf["__cluster__"] = clf.predict(xdf_clustering_data)

    print("\n[x] Datensätze klassifiziert:")
    print(xdf[[xcol] + xcols_cluster + ['__cluster__']])

    print("\n[x] Frequenz der Datensätze im jeweiligen Cluster:")
    _ = dbeschreiben.frequenz_werte(xdf, xcol="__cluster__", prozente=True)

    cluster_means = xdf.groupby("__cluster__").mean()[xcol].to_dict()
    print("\n[x] Durchschnitte im Cluster: ", cluster_means)

    xgruppen = []
    for cl in range(anzahl_cluster):
        xgruppen.append(
            xdf[xdf["__cluster__"] == cl].fillna(value={xcol: cluster_means[cl]})
        )
    xdf_res = pd.concat(xgruppen)

    print("\n [x] Wenn alles gut gelaufen ist, haben sich die statistischen Maßen nicht wesentlich geändert:")
    print("\t VORHER:")
    print(xdf_input.describe())
    print("\n\t NACHHER:")
    print(xdf_res.describe())

    xdf_res = xdf_res[list(filter(lambda x: x != '__cluster__' and x != (xcol + "_istNull"), xdf_res.columns))]
    for xc in ignoriere_spalten:
        xdf_res[xc] = xdf_input[xc]

    print("=============================================================================================")
    return xdf_res, clf
