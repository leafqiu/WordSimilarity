from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np


def word_similarity_wordnet(word_pairs):
    path_scores = []
    for (first_word, second_word) in word_pairs:
        first_synset = wn.synsets(first_word)
        second_synset = wn.synsets(second_word)
        maxsim = 0
        for word1 in first_synset:
            for word2 in second_synset:
                lch_sim = word1.path_similarity(word2)
                if lch_sim is not None:
                    maxsim = max(maxsim, lch_sim)
        path_scores.append(maxsim)
    # print(path_scores[:10])
    return path_scores


def compare_similarity(scores, labels):
    pass


if __name__ == '__main__':
    data = pd.read_csv('data/MTURK-771.csv')
    word_pairs = []
    sim_scores = []
    # get word pairs and similarity scores
    for pair in data.get_values():
        word_pairs.append(tuple(pair[:2]))
        sim_scores.append(pair[2])
    path_scores = word_similarity_wordnet(word_pairs, sim_scores)
    compare_similarity(path_scores, sim_scores)