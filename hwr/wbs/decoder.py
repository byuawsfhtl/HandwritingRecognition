import tensorflow as tf

from hwr.wbs.loader import FilePaths


class WordBeamSearch:
    def __init__(self, beam_width, lm_type, lm_smoothing, corpus, chars, word_chars, os_type='linux', gpu=True,
                 multithreaded=False):
        self.num_chars = len(chars)
        self.beam_width = beam_width
        self.lm_type = lm_type
        self.lm_smoothing = lm_smoothing
        self.corpus = corpus.encode('utf8')
        self.chars = chars.encode('utf8')
        self.word_chars = word_chars.encode('utf8')

        assert os_type in ['linux', 'mac']  #

        if os_type == 'linux' and gpu and multithreaded:
            op_path = FilePaths.wbs_linux_gpu_parallel_8()
        elif os_type == 'linux' and not gpu and multithreaded:
            op_path = FilePaths.wbs_linux_parallel_8()
        elif os_type == 'linux' and not gpu and not multithreaded:
            op_path = FilePaths.wbs_linux()
        elif os_type == 'linux' and gpu and not multithreaded:
            op_path = FilePaths.wbs_linux_gpu()
        elif os_type == 'mac' and multithreaded:
            op_path = FilePaths.wbs_mac_parallel_8()
        elif os_type == 'mac' and not multithreaded:
            op_path = FilePaths.wbs_mac()
        else:
            raise Exception('Unsupported Platform! Must be [linux, mac]. Windows not supported for TF custom ops')

        self.module = tf.load_op_library(op_path)

    def decode(self, mat):
        mat = tf.transpose(mat, [1, 0, 2])  # Transpose to get (SequenceLength x BatchSize x NumClasses)
        mat = tf.roll(mat, shift=-1, axis=2)  # Roll the class axis to place ctc-blank last (which is what wbs expects)
        output = self.module.word_beam_search(mat, self.beam_width, self.lm_type, self.lm_smoothing, self.corpus,
                                              self.chars, self.word_chars)
        int64_output = tf.cast(output, tf.int64)

        # Reverse the action of the tf.roll to get back to expected indices
        wbs_output = tf.math.floormod(tf.math.add(int64_output, 1), self.num_chars + 1)

        return wbs_output

    def __call__(self, mat):
        return self.decode(mat)
