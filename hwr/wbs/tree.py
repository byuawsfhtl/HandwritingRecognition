ROOT_NAME = -1


class Node:
    def __init__(self, name: int, is_word: bool):
        self.name = name
        self.is_word = is_word
        self.children = dict()

    def add_child(self, node):
        self.children[node.name] = node

    def get_child(self, name: int):
        return self.children.get(name, None)

    def search(self, node, space, idx2char):
        if node.name != ROOT_NAME:
            nn = str(node.name)
            name = idx2char[nn]
        else:
            name = 'Root'
        if node.is_word:
            name = str(name) + '*'

        self.text += space + name + '\n'

        if space == '':
            more_space = '|--'
        else:
            more_space = '   ' + space

        for child in node.children.values():
            self.search(child, more_space, idx2char)

    def print(self, char2idx):
        self.text = ''
        self.search(self, '', char2idx)
        print(self.text)


class PrefixTree:
    def __init__(self, words, char2idx, punctuation):
        self.words = list(set(words.split()))
        self.root = Node(ROOT_NAME, False)
        self.char2idx = char2idx
        self.punctuation = list(map(lambda x: self.char2idx[x], punctuation))
        self.space = self.char2idx[' ']

    def build_tree(self):
        # Iterate over all words in the list
        for word in self.words:
            current_node = self.root
            # Iterate over each character in the list
            for index, char in enumerate(word):
                # Convert the character to its integer representation
                try:
                    char_idx = self.char2idx[char]
                except KeyError as e:
                    print('The character was not found in the provided character set: {}. Skipping character '
                          'in prefix tree.'.format(e))
                    continue

                # Check if the node has already been added
                child = current_node.get_child(char_idx)

                # Is this the last character in the word?
                is_word = True if index == (len(word) - 1) else False

                # If this particular node hasn't been created, add it
                if child is None:
                    child = Node(char_idx, is_word)
                    current_node.add_child(child)
                elif is_word:
                    child.is_word = is_word

                # Update the current node
                current_node = child

    def search(self, char_list):
        current_node = self.root
        for char in char_list:
            current_node = current_node.children.get(char, None)
            if current_node is None:
                return None
        return current_node

    def find_last_word_start(self, beam):
        for index, char in enumerate(beam[::-1]):
            if char in self.punctuation:
                return len(beam) - index

        return 0

    def get_possible_chars(self, beam):
        """
        Get all possible characters that the current beam could be extended by
        :param beam: List of possible char indices
        :return: All possible char indices to which the beam could be extended
        """
        # If the last character added is punctuation, we are in a non-word state.
        # Thus, we can return any punctuation character and any character that starts a word.
        if len(beam) == 0 or beam[-1] in self.punctuation:
            return list(self.root.children.keys()) + self.punctuation  # Non-Word State

        # Since we are dealing with line-level transcriptions, we need to find the last word
        # Reverse the list find the last space, then subtract by the total length of the list
        # to get the start index of the word
        last_word_start = self.find_last_word_start(beam)

        # Navigate to the last char extended in the word
        char_node = self.search(beam[last_word_start:])

        # If the beam ends with a word, then allow punctuation - Non-Word-State
        if char_node.is_word:
            return list(char_node.children.keys()) + self.punctuation

        # Return the possible chars to be extended - Word-State
        return char_node.children.keys()

    def __repr__(self):
        return self.root.__repr__()

    def __str__(self):
        return self.__repr__()
