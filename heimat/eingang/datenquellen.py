class Datenquelle:
    data = None

    def __init__(self, fname):
        """
        """
        self.fname = fname

    def inhalt(self):
        """
            Zeigt Inhalt der Datei als Text
        """
        with open(self.fname, 'r') as file:
            inhalt = file.readlines()
        return inhalt

    def features_ausfiltern(self, *args):
        ist_arg_in_featurenamen = lambda xc: len(list(filter(lambda x: x in xc, args))) > 0
        if self.data is not None:
            xselekt = list(filter(lambda xc: not ist_arg_in_featurenamen(xc), self.data.columns))
            print(xselekt)
            self.data = self.data[xselekt]
