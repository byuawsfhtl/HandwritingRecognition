import tensorflow as tf

from hwr.wbs.loader import FilePaths


class WordBeamSearch:
    """
    Wrapper class for using the Word Beam Search algorithm through a pre-built tensorflow custom operation.
    Tensorflow custom ops are useful because they allow the decoding to take place in Graph Mode. However, they
    are only supported in Linux/MacOS.

    See CTC-Word-Beam-Search GitHub Readme for more details:
    https://github.com/githubharald/CTCWordBeamSearch
    """
    def __init__(self, beam_width, lm_type, lm_smoothing, corpus, chars, word_chars, os_type='linux', gpu=True,
                 multithreaded=False):
        """
        :param beam_width: The beam width used in the word beam search algorithm
        :param lm_type: The language model type. Most often will use 'Words'. See WBS documentation for more details.
        :param lm_smoothing: Float representing language model smoothing. See WBS documentation for more details.
        :param corpus: String with space delimited words representing the possible words for decoding
        :param chars: String containing all possible characters used in model
        :param word_chars: String containing all non-punctuation characters
        :param os_type: The operating system type: ['linux', 'mac'] Windows is not supported.
        :param gpu: Boolean indicating whether or not the system supports GPU/Cuda
        :param multithreaded: Whether or not to use 8 parallel threads during decoding operation.
        """
        self.num_chars = len(chars)
        self.beam_width = beam_width
        self.lm_type = lm_type
        self.lm_smoothing = lm_smoothing
        self.corpus = corpus.encode('utf8')
        self.chars = chars.encode('utf8')
        self.word_chars = word_chars.encode('utf8')

        assert os_type in ['linux', 'mac']  # Windows not supported

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
        mat = tf.nn.softmax(mat)
        mat = tf.transpose(mat, [1, 0, 2])  # Transpose to get (SequenceLength x BatchSize x NumClasses)
        mat = tf.roll(mat, shift=-1, axis=2)  # Roll the class axis to place ctc-blank last (which is what wbs expects)
        output = self.module.word_beam_search(mat, self.beam_width, self.lm_type, self.lm_smoothing, self.corpus,
                                              self.chars, self.word_chars)

        # Reverse the action of the tf.roll to get back to expected indices
        wbs_output = tf.math.floormod(tf.math.add(output, 1), self.num_chars + 1)

        return wbs_output

    def __call__(self, mat):
        return self.decode(mat)
