import sys

import numpy as np

from hwr.wbs.loader import DictionaryLoader


def possible_chars(args):
    """
    Possible Characters

    This script loads text files and provides a set of all the characters used. This is useful for determining
    the charset to provide when training a model for handwriting recognition.

    Usage:
    * python possible_chars.py <FILE_PATH>

    Command Line Arguments:
    * FILE_PATH: The path to the the text file.

    :param args: command line arguments
    :return: None
    """
    if len(args) == 0:
        print('Please include path to text file!')
        return

    words = ''
    for file_path in args:
        print('Loading ', file_path)
        words += DictionaryLoader.from_file(file_path) + ' '

    chars = list(set(''.join(words)))
    chars = np.sort(chars)
    chars = ''.join(chars)

    print('Possible Characters:', chars)
    print('Total Characters:', len(chars))


if __name__ == '__main__':
    possible_chars(sys.argv[1:])
