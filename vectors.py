# -*- coding: utf-8 -*-

import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('E:\语义计算\hw1\GoogleNews-vectors\GoogleNews-vectors-negative300.bin', binary=True)
model.save('data/GoogleNews-vectors.model')
print('Saving finished.')