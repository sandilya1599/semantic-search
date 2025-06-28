def create_mappings(vocab):
    word2id = {'<PAD>': 0, '<UNK>' : 1}
    id2word = {'0':'<PAD>', '1': '<UNK>'}
    number = 2

    for word in vocab:
        word2id[word] = number
        id2word[number] = word
        number = number + 1
    print("Words in Corpus: ", len(word2id.keys()))
    return word2id, id2word

def build_id_sequences(tokenized_corpus, word2id):
    """
    Convert list of sentences into numbers.
    """
    id_sequences = []
    for line in tokenized_corpus:
        id_line = []
        for word in line:
            if word in word2id.keys():
                id_line.append(word2id[word])
            else:
                id_line.append(word2id['<UNK>'])
        id_sequences.append(id_line)
    print("Number of sentences: ", len(id_sequences))
    return id_sequences


def build_id_sequences_for_sentence(line, word2id):
    """
    Convert list of sentences into numbers.
    """
    id_line = []
    for word in line:
        if word in word2id.keys():
            id_line.append(word2id[word])
        else:
            id_line.append(word2id['<UNK>'])
    print("Number of sentences: ", len(id_line))
    return id_line

def construct_sentence(id2word, sentence):
    const_sentence = ''
    for word in sentence:
        const_sentence += ' ' + id2word[word]
    return const_sentence