import os

ENGLISH_WORDS = 'data/english_words.txt'
FRENCH_WORDS = 'data/french_words.txt'
FRENCH_NAMES = 'data/french_names.txt'
ASCII_NAMES = 'data/ascii_names.txt'
CENSUS_NAMES_5 = 'data/census_names_5.txt'
CENSUS_NAMES_10 = 'data/census_names_10.txt'
CENSUS_NAMES_15 = 'data/census_names_15.txt'


class FilePaths:
    """
    FilePaths
    Static class for getting this file's absolute path and the paths to the the word lists.
    This is useful especially when built with a conda package and placed in an arbitrary location.
    """
    @staticmethod
    def current_file_path():
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def english_words():
        return os.path.join(FilePaths.current_file_path(), ENGLISH_WORDS)

    @staticmethod
    def french_words():
        return os.path.join(FilePaths.current_file_path(), FRENCH_WORDS)

    @staticmethod
    def french_names():
        return os.path.join(FilePaths.current_file_path(), FRENCH_NAMES)

    @staticmethod
    def ascii_names():
        return os.path.join(FilePaths.current_file_path(), ASCII_NAMES)

    @staticmethod
    def census_names_5():
        return os.path.join(FilePaths.current_file_path(), CENSUS_NAMES_5)

    @staticmethod
    def census_names_10():
        return os.path.join(FilePaths.current_file_path(), CENSUS_NAMES_10)

    @staticmethod
    def census_names_15():
        return os.path.join(FilePaths.current_file_path(), CENSUS_NAMES_15)


class DictionaryLoader:
    """
    DictionaryLoader
    Static class meant to load word lists from files and return them as newline separated strings.
    """
    @staticmethod
    def from_file(filename, include_cased=False):
        """
        Load dictionary words from a file into a string.
        :param filename: The filename with words separated by newlines.
        :param include_cased: True indicating a desire to add both the lower-cased and capitalized word or False to only
                              include the word as given in the list.
        :return: The words as given in the file separated by a newline.
        """
        f = open(filename)
        words = ''
        for word in f.readlines():
            if not include_cased:
                words += word
            else:
                capitalized = word.capitalize()
                lowercase = word.lower()
                words += capitalized + lowercase

            if not words.endswith('\n'):
                words += '\n'

        return words

    @staticmethod
    def english_words(include_cased=False):
        """
        List of 370,000 english words.
        :param include_cased: True indicating a desire to add both the lower-cased and capitalized word or False to only
                            include the word as given in the list.
        :return: String of english words separated by newline
        """
        return DictionaryLoader.from_file(FilePaths.english_words(), include_cased=include_cased)

    @staticmethod
    def french_words(include_cased=False):
        """
        List of 80,060 french words targeted towards death records.
        :param include_cased: True indicating a desire to add both the lower-cased and capitalized word or False to only
                            include the word as given in the list.
        :return: String of French words separated by newline
        """
        return DictionaryLoader.from_file(FilePaths.french_words(), include_cased=include_cased)

    @staticmethod
    def french_names(include_cased=False):
        """
        List of 25,000 french names as given by Ancestry.
        :param include_cased: True indicating a desire to add both the lower-cased and capitalized word or False to only
                            include the word as given in the list.
        :return: String of French names separated by newline
        """
        return DictionaryLoader.from_file(FilePaths.french_names(), include_cased=include_cased)

    @staticmethod
    def ascii_names(include_cased=False):
        """
        List of nearly 100,000 ascii given names and surnames.
        :param include_cased: True indicating a desire to add both the lower-cased and capitalized word or False to only
                            include the word as given in the list.
        :return: String of ascii names separated by newline.
        """
        return DictionaryLoader.from_file(FilePaths.ascii_names(), include_cased=include_cased)

    @staticmethod
    def census_names_5(include_cased=False):
        """
        List of census names acquired from the 1900-1940 census. A master list was acquired with all possible names.
        This list contains names that occurred at least 5 times in the 5 censuses combined.
        """
        return DictionaryLoader.from_file(FilePaths.census_names_5(), include_cased=include_cased)

    @staticmethod
    def census_names_10(include_cased=False):
        """
        List of census names acquired from the 1900-1940 census. A master list was acquired with all possible names.
        This list contains names that occurred at least 10 times in the 5 censuses combined.
        """
        return DictionaryLoader.from_file(FilePaths.census_names_10(), include_cased=include_cased)

    @staticmethod
    def census_names_15(include_cased=False):
        """
        List of census names acquired from the 1900-1940 census. A master list was acquired with all possible names.
        This list contains names that occurred at least 15 times in the 5 censuses combined.
        """
        return DictionaryLoader.from_file(FilePaths.census_names_15(), include_cased=include_cased)
