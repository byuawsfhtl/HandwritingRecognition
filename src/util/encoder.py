import numpy as np
import json


class Encoder:
    """
    Encoder

    Responsible for mapping characters to indices and indices to characters. This mapping
    is necessary to get labels into a format that can be used in the TensorFlow model.
    """

    def __init__(self, char_set_file_path, max_sequence_size=128, blank_character=0):
        """
        Accepts a json file with the corresponding character set. The json file
        should eventually map to python dictionaries, idx_to_char and char_to_idx.
        These will be used to encode/decode characters and indices.

        :param char_set_file_path:
        :param max_sequence_size:
        :param blank_character:
        """
        with open(char_set_file_path) as file:
            self.json = json.load(file)

        self.max_sequence_size = max_sequence_size
        self.blank_character = blank_character

    @staticmethod
    def remove_duplicates(idxs):
        """
        When we do decoding, our index tensor will have repeating characters.
        With best-path decoding, we remove the repeating elements. If a word
        actually contains repeating characters, there should be a blank in-between.

        :param idxs:
        :return:
        """
        new_idxs = []

        for i in range(len(idxs)):
            # Only append if the next character in the sequence is not
            # identical to the current character. If we're at the end of
            # the sequence, add it.
            if i + 1 == len(idxs) or idxs[i] != idxs[i + 1]:
                new_idxs.append(idxs[i])

        return new_idxs

    def idx_to_char(self, idx):
        """
        Convert an index to a character

        :param idx: A single integer index representing a character
        :return: A single character
        """
        if idx == self.blank_character:
            return ''  # Return empty string for the blank character
        else:
            return self.json['idx_to_char'][str(int(idx))]

    def char_to_idx(self, char):
        """
        Convert a character to an index

        :param char: A single character
        :return: A single integer index representing the character
        """
        return int(self.json['char_to_idx'][char])

    def str_to_idxs(self, string):
        """
        Convert a string to a list of indices

        :param string: The label string
        :return: The label as a list of indices
        """
        idxs = []

        zeros = np.full(self.max_sequence_size, self.blank_character)
        for char in string:
            idxs.append(self.char_to_idx(char))

        # Pad the array to the max sequence size
        idxs = np.concatenate((idxs, zeros))[:self.max_sequence_size]

        return idxs

    def idxs_to_str(self, idxs, remove_duplicates=True):
        """
        Convert a list of indices to a string

        :param idxs: The label as indices
        :param remove_duplicates: T/F - Whether or not repeating characters should be removed.
                                  This can be helpful if decoding the target label value and
                                  blank characters have not been introduced.
        :return: The label as a string
        """
        string = ''

        if remove_duplicates:
            idxs = Encoder.remove_duplicates(idxs)

        for idx in idxs:
            string += self.idx_to_char(idx)

        return string

    def str_to_idxs_batch(self, batch):
        """
        Convert a list of strings to a list of indices

        :param batch: The list of strings
        :return: A list of lists of indices
        """
        idxs = []

        for string in batch:
            idx = self.str_to_idxs(string)
            idxs.append(idx)

        return idxs

    def idxs_to_str_batch(self, batch, remove_duplicates=True):
        """
        Convert a list of indices to a list of strings

        :param batch: The list of lists of indices
        :param remove_duplicates: T/F - Whether or not repeating characters should be removed
        :return: A list of strings
        """
        strings = []

        for idxs in batch:
            strings.append(self.idxs_to_str(idxs, remove_duplicates=remove_duplicates))

        return strings
