from collections.abc import Mapping
from gensim.models import Word2Vec
from dataload import *
import os


# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def embeddings(sequence, size, window, min_count, workers, sg):
    """
    Create embeddings using word2vec
    :param sequence: e.g. MSA... Amino acid sequence
    :param size: vector size
    :param sg: skip-gram =1, cbow =0 = default
    :return: vector for each protein sequence
    """
    w2v_model = Word2Vec(sequence, min_count=min_count, workers=workers, vector_size=size,
                         window=window, sg=sg)
    # print(w2v_model.wv['A'].shape)

    word_vector = []
    for s in sequence:
        sum_vector = []
        for word in s:
            sum_vector.append(w2v_model.wv[word])
        word_vector.append(sum_vector)

    return np.array(word_vector)


