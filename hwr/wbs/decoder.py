import tensorflow as tf

from word_beam_search import WordBeamSearch as WbsPyBind

from hwr.wbs.tree import PrefixTree


class WordBeamSearch:
    """
    Wrapper for the Word-Beam-Search object as given in the word_beam_search package. This decoder constrains the
    model's output to dictionary words while allowing for arbitrary punctuation.

    See CTC-Word-Beam-Search GitHub Readme for more details:
    https://github.com/githubharald/CTCWordBeamSearch
    """
    def __init__(self, corpus, chars, word_chars, beam_width=15, mode='Words', lm_smoothing=0.0):
        """
        :param corpus: String with space delimited words representing the possible words for decoding
        :param chars: The possible characters that the model can predict
        :param word_chars: The non-punctuation characters the model can predict
        :param beam_width: The beam_width for the word beam search algorithm
        :param mode: One of the following ['Words', 'NGrams', 'NGramsForecast', 'NGramsForecastAndSample']
        :param lm_smoothing: Language model smoothing value
        """
        self.decoder = WbsPyBind(beam_width, mode, lm_smoothing, corpus, chars, word_chars)

    def decode(self, logits):
        """
        Decode the model output using WordBeamSearch

        :param logits: The model output logits as tensor
        :return: The decoded model output as tensor
        """
        output = tf.nn.softmax(logits).numpy()
        pred = self.decoder.compute(output)
        return tf.constant(pred, dtype=tf.int64)

    def __call__(self, logits):
        """
        Decode the model output using WordBeamSearch

        :param logits: The model output logits as tensor
        :return: The decoded model output as tensor
        """
        return self.decode(logits)


class PythonWordBeamSearch:
    """
    The Word-Beam-Search decoding algorithm written in python. This decoder constrains the model's output to dictionary
    words while allowing for arbitrary punctuation.

    See CTC-Word-Beam-Search GitHub Readme for more details:
    https://github.com/githubharald/CTCWordBeamSearch
    """
    def __init__(self, words, punctuation, beam_width, char2idx, blank_char=0):
        """
        :param words: String with space delimited words representing the possible words for decoding
        :param punctuation: String with no delimiters containing all punctuation characters
        :param beam_width: The beam width used in the word beam search algorithm
        :param char2idx: Tensorflow lookup table that contains mapping between chars and idxs
        :param blank_char: The idx representing the blank character
        """
        chars, idxs = char2idx.export()
        chars = chars.numpy()
        idxs = idxs.numpy()
        self.char2idx = {char.decode('utf-8'): idx for idx, char in zip(idxs, chars)}

        self.prefix_tree = PrefixTree(words, self.char2idx, list(punctuation))
        self.prefix_tree.build_tree()
        self.beam_width = beam_width
        self.blank_char = blank_char

    @staticmethod
    def best_beams(beams, beam_width):
        # Sort and return the top beam_width
        return sorted(beams, key=lambda x: x[1] + x[2], reverse=True)[:beam_width]

    def wbs_single_instance(self, sequence):
        big_b = [([], 0, 1, [])]
        for timestep in sequence:
            best_beams = self.best_beams(big_b, self.beam_width)
            big_b = []
            for b, prob_non_blank, prob_blank, full_b in best_beams:
                # Calculate probability of the beam not being extended (either by a repeating character or blank)
                if len(b) != 0:
                    pnb = prob_non_blank + (prob_non_blank * timestep[b[-1]])
                else:
                    pnb = prob_non_blank

                pb = prob_blank + ((prob_non_blank + prob_blank) * timestep[self.blank_char])
                extender = [self.blank_char] if len(b) == 0 or timestep[self.blank_char] > timestep[b[-1]] else [b[-1]]
                big_b.append((b, pnb, pb, full_b + extender))

                for c in self.prefix_tree.get_possible_chars(b):
                    b_prime = b + [c]
                    if len(b) > 0 and b[-1] == c:
                        pnb = prob_non_blank + (prob_blank * timestep[c])
                    else:
                        pnb = prob_non_blank + ((prob_blank + prob_non_blank) * timestep[c])
                    big_b.append((b_prime, pnb, 0, full_b + [c]))

        big_b = self.best_beams(big_b, 1)  # Reduce top beams to 1

        return big_b[0][3]

    def decode(self, batch):
        """
        :param batch: [Batch, TimeSteps, Classes]
        :return: Most probable Sequence
        """
        batch = tf.nn.softmax(batch).numpy()
        best_beam_batch = []
        for sequence in batch:
            best_beam_batch.append(self.wbs_single_instance(sequence))

        return tf.constant(best_beam_batch, dtype=tf.int64)

    def __call__(self, batch):
        """
        :param batch: [Batch, TimeSteps, Classes]
        :return: Most probable Sequence
        """
        return self.decode(batch)
