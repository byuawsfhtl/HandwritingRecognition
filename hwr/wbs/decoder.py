import tensorflow as tf
import numpy as np

from hwr.wbs.tree import PrefixTree

from word_beam_search import WordBeamSearch

class WordBeamSearchWrapper:
    """
    The Word-Beam-Search decoding algorithm. This decoder constrains the model's output to dictionary words
    while allowing for arbitrary punctuation.

    See CTC-Word-Beam-Search GitHub Readme for more details:
    https://github.com/githubharald/CTCWordBeamSearch
    """
    def __init__(self, words, punctuation, beam_width, charset):
        """
        :param words: String with space delimited words representing the possible words for decoding
        :param punctuation: String with no delimiters containing all punctuation characters
        :param beam_width: The beam width used in the word beam search algorithm
        :param char2idx: Tensorflow lookup table that contains mapping between chars and idxs
        :param blank_char: The idx representing the blank character
        """
        #Calculate wordChars
        #print("NUM CHARS: ", len(charset))
        #print(charset)
        #count = 0
        wordChars = ""
        for c in charset:
            if punctuation.find(c) == -1:
                wordChars += c
            #count+=1
            #print(count, c)

        self.wbs = WordBeamSearch(beam_width, 'Words', 0.0, words.encode('utf8'), charset.encode('utf8'), wordChars.encode('utf8'))
        
        """
        chars, idxs = char2idx.export()
        chars = chars.numpy()
        print(chars)
        idxs = idxs.numpy()
        self.char2idx = {char.decode('utf-8'): idx for idx, char in zip(idxs, chars)}

        self.prefix_tree = PrefixTree(words, self.char2idx, list(punctuation))
        self.prefix_tree.build_tree()
        self.beam_width = beam_width
        self.blank_char = 0 #blank_char
        
        

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
        """

    def decode(self, batch):
        """
        :param batch: [Batch, TimeSteps, Classes]
        :return: Most probable Sequence
        """

        """
        best_beam_batch = []
        for sequence in batch:
            best_beam_batch.append(self.wbs_single_instance(sequence))
        """

        batch = tf.nn.softmax(batch).numpy()
        best_beam_batch = self.wbs.compute(batch)

        return tf.constant(best_beam_batch, dtype=tf.int32)

    def __call__(self, batch):
        """
        :param batch: [Batch, TimeSteps, Classes]
        :return: Most probable Sequence
        """
        return self.decode(batch)
