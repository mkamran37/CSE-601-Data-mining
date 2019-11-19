class point:
    def __init__(self, point = list(), categoricalData = list(), label=-1, groundTruth = -1, id = -1):
        self.point = point
        self.label = label
        self.groundTruth = groundTruth
        self.id = id
        self.categoricalData = categoricalData