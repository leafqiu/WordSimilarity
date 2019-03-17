from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np



def word_similarity_wordnet(word_pairs, sim_scores):
    for pair in word_pairs:
        # TODO: computing similarity between two words
        pass


if __name__ == '__main__':
    data = pd.read_csv('data/MTURK-771.csv')
    word_pairs = []
    sim_scores = []
    # get word pairs and similarity scores
    for pair in data.get_values():
        word_pairs.append(tuple(pair[:2]))
        sim_scores.append(pair[2])
    word_similarity_wordnet(word_pairs, sim_scores)