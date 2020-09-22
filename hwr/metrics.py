import tensorflow as tf
import numpy as np
import editdistance as ed


# class WordErrorRate(tf.keras.metrics.Metric):
#     """
#     WordErrorRate
#
#     Keras metric to keep track of the Word Error Rate
#     """
#     def __init__(self, name='word_error_rate', **kwargs):
#         """
#         Create class variables to be used in metric calculation.
#
#         :param name: Name of the metric
#         :param kwargs: Additional arguments to be passed to the keras metrics superclass
#         """
#         super(WordErrorRate, self).__init__(name=name, **kwargs)
#
#         self.total_error = self.add_weight(name='wer_total_error', initializer='zeros')
#         self.count = self.add_weight(name='wer_count', initializer='zeros')
#
#     def update_state(self, y_true, y_pred):
#         """
#         Update the metric given the predicted and actual
#
#         :param y_true: The actual label as a string or list of strings
#         :param y_pred: The predicted label as a string or list of strings
#         :return:
#         """
#         print('Type:', type(y_true))
#         print(y_true)
#
#         # Batch List Version
#         if isinstance(y_true, np.ndarray) or isinstance(y_true, np.ndarray):
#             for y_true_single, y_pred_single in zip(y_true, y_pred):
#                 self.total_error.assign_add(self.wer(y_true_single, y_pred_single))
#
#             self.count.assign_add(len(y_true))
#         # Single String Version
#         else:
#             self.total_error.assign_add(WordErrorRate.wer(y_true, y_pred))
#             self.count.assign_add(1)
#
#         return self.total_error / self.count
#
#     @staticmethod
#     def wer(y_true, y_pred):
#         """
#         The word error rate given the predicted and actual string
#
#         :param y_true: The actual label string
#         :param y_pred: The predicted label string
#         :return: The word error rate
#         """
#         y_true = y_true.numpy() if tf.is_tensor(y_true) else y_true
#         y_pred = y_pred.numpy() if tf.is_tensor(y_pred) else y_pred
#
#         y_true = y_true.split()
#         y_pred = y_pred.split()
#
#         dist = ed.eval(y_true, y_pred)
#         if len(y_true) == 0:
#             return len(y_pred)
#
#         return float(dist) / float(len(y_true))
#
#     def result(self):
#         """
#         The average word error rate of all labels that have been submitted to the metric
#
#         :return: The word error rate
#         """
#         return self.total_error / self.count


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
        return self.cer_total_error / self.count, self.wer_total_error / self.count

    def __call__(self, y_true, y_pred):
        self.update(y_true, y_pred)


# class CharacterErrorRate(tf.keras.metrics.Metric):
#     """
#     CharacterErrorRate
#
#     Keras metric to keep track of the Character Error Rate
#     """
#     def __init__(self, name='character_error_rate', **kwargs):
#         """
#         Create class variables to be used in metric calculation.
#
#         :param name: The name of the metric
#         :param kwargs: Additional arguments to be passed to the keras metrics superclass
#         """
#         super(CharacterErrorRate, self).__init__(name=name, **kwargs)
#
#         self.total_error = self.add_weight(name='cer_total_error', initializer='zeros')
#         self.count = self.add_weight(name='cer_count', initializer='zeros')
#
#     def update_state(self, y_true, y_pred):
#         """
#         Update the metric given the predicted and actual
#
#         :param y_true: The actual label as a string or list of strings
#         :param y_pred: The predicted label as a string or list of strings
#         :return:
#         """
#
#         print('Type:', type(y_true))
#         print(y_true)
#
#         # Batch List Version
#         if isinstance(y_true, list) or isinstance(y_true, np.ndarray):
#             for y_true_single, y_pred_single in zip(y_true, y_pred):
#                 self.total_error.assign_add(self.cer(y_true_single, y_pred_single))
#
#             self.count.assign_add(len(y_true))
#         # Single String Version
#         else:
#             self.total_error.assign_add(CharacterErrorRate.cer(y_true, y_pred))
#             self.count.assign_add(1)
#
#         return self.total_error / self.count
#
#     @staticmethod
#     def cer(y_true, y_pred):
#         """
#         The character error rate given the predicted and actual string
#
#         :param y_true: The actual label string
#         :param y_pred: The predicted label string
#         :return: The character error rate
#         """
#         y_true = y_true.numpy() if tf.is_tensor(y_true) else y_true
#         y_pred = y_pred.numpy() if tf.is_tensor(y_pred) else y_pred
#
#         y_true = ' '.join(y_true.split())
#         y_pred = ' '.join(y_pred.split())
#
#         dist = ed.eval(y_true, y_pred)
#         if len(y_true) == 0:
#             return len(y_pred)
#
#         return float(dist) / float(len(y_true))
#
#     def result(self):
#         """
#         The average character error rate of all labels that have been submitted to the metric
#
#         :return: The character error rate
#         """
#         return self.total_error / self.count
