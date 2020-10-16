import tensorflow as tf

from hwr.wbs.loader import FilePaths


class WordBeamSearch:
    def __init__(self, beam_width, lm_type, lm_smoothing, corpus, chars, word_chars, multithreaded=False):
        self.beam_width = beam_width
        self.lm_type = lm_type
        self.lm_smoothing = lm_smoothing
        self.corpus = corpus.encode('utf8')
        self.chars = chars.encode('utf8')
        self.word_chars = word_chars.encode('utf8')

        if multithreaded:
            self.module = tf.load_op_library(FilePaths.wbs_parallel_8())
        else:
            self.module = tf.load_op_library(FilePaths.wbs_single_threaded())

    def decode(self, mat):
        mat = tf.transpose(mat, [1, 0, 2])  # Transpose to get (SequenceLength x BatchSize x NumClasses)
        mat = tf.roll(mat, shift=-1, axis=2)  # Roll the class axis to place ctc-blank last (which is what wbs expects)
        output = self.module.word_beam_search(mat, self.beam_width, self.lm_type, self.lm_smoothing, self.corpus,
                                              self.chars, self.word_chars)
        int64_output = tf.cast(output, tf.int64)

        return int64_output

    def __call__(self, mat):
        return self.decode(mat)
