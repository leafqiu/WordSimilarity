# -*- coding: utf-8 -*-

from nltk.corpus import wordnet as wn
import pandas as pd
import gensim
from scipy import stats


# Path based similarity
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
    # print('scores-1:', path_scores[:10])
    return path_scores


# Word2vec based similarity (cosine similarity)
def word_similarity_word2vec(word_pairs):
    scores = []
    model = gensim.models.KeyedVectors.load('data/GoogleNews-vectors.model')
    for (firstw, secondw) in word_pairs:
        sim = model.similarity(firstw, secondw)
        scores.append(sim)
    # print('scores-2:', scores[:10])
    return scores


# Compare computing scores with human scores: Spearman's rank correlation coefficient
def spearman_corr(scores, labels):
    (rho, p_value) = stats.spearmanr(scores, labels)
    print(rho, p_value)


def main():
    data = pd.read_csv('data/MTURK-771.csv')
    word_pairs = []
    sim_scores = []
    # get word pairs and similarity scores
    for pair in data.get_values():
        word_pairs.append(tuple(pair[:2]))
        sim_scores.append(pair[2])
    path_scores = word_similarity_wordnet(word_pairs)
    print('path based:')
    spearman_corr(path_scores, sim_scores)
    word2vec_scores = word_similarity_word2vec(word_pairs)
    print('word2vec based:')
    spearman_corr(word2vec_scores, sim_scores)


if __name__ == '__main__':
    main()