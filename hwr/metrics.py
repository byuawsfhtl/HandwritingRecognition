import editdistance as ed


class ErrorRates:
    def __init__(self):
        self.cer_total_error = 0
        self.wer_total_error = 0
        self.count = 0

    def cer(self, y_true, y_pred):
        y_true = ' '.join(y_true.split())
        y_pred = ' '.join(y_pred.split())

        dist = ed.eval(y_true, y_pred)
        if len(y_true) == 0:
            return len(y_pred)

        self.cer_total_error += float(dist) / float(len(y_true))

    def wer(self, y_true, y_pred):
        y_true = y_true.split()
        y_pred = y_pred.split()

        dist = ed.eval(y_true, y_pred)
        if len(y_true) == 0:
            return len(y_pred)

        self.wer_total_error += float(dist) / float(len(y_true))

    def update(self, y_true, y_pred):
        self.cer(y_true, y_pred)
        self.wer(y_true, y_pred)
        self.count += 1

    def get_error_rates(self):
        return (self.cer_total_error / self.count, self.wer_total_error / self.count) if self.count > 0 else (1.0, 1.0)

    def __call__(self, y_true, y_pred):
        self.update(y_true, y_pred)
