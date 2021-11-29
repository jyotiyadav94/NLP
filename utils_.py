# %%
import os
import re
import sys
import shutil
import zipfile
import gensim
import scipy
import numpy as np
import pandas as pd
import tqdm
from operator import add
from urllib import request
from itertools import chain
import collections
from collections import Counter
from functools import reduce
from collections import OrderedDict
import gensim.downloader as gloader
from typing import List, Callable, Dict


# %%
class utility():

        def cleaning():
            folder = os.getcwd()
            print("Current work directory: " + str(folder))
            dataset_folder = os.path.join(os.getcwd(), "Datasets")

            if not os.path.exists(dataset_folder):
                os.makedirs(dataset_folder)

            for filename in os.listdir(dataset_folder):
                file_path = os.path.join(dataset_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

            print("Cleaned")

# %%
        def downloadDataSet(url,Datasets,dependency_treebank):

            # Config
            print("Current work directory: {}".format(os.getcwd()))

            dataset_folder = os.path.join(os.getcwd(), Datasets)

            if not os.path.exists(dataset_folder):
                os.makedirs(dataset_folder)

            url = url

            dataset_path = os.path.join(dataset_folder, dependency_treebank)
            print(dataset_path)

            print(dataset_path)

            def download_dataset(download_path: str, url: str):
                if not os.path.exists(download_path):
                    print("Downloading dataset...")
                    request.urlretrieve(url, download_path)
                    print("Download complete!")

            def extract_dataset(download_path: str, extract_path: str):
                print("Extracting dataset... (it may take a while...)")
                with zipfile.ZipFile(download_path) as loaded_tar:
                    loaded_tar.extractall(extract_path)
                    print(loaded_tar)
                print("Extraction completed!")

            # Download
            download_dataset(dataset_path, url)
            # Extraction
            extract_dataset(dataset_path, dataset_folder)


# %%
        def build_vocabulary(words, special_tokens=[]):
            words = words.map(lambda s: collections.Counter([w.lower().strip() for w in s]))
            #On the sequence words apply the function sum
            count = reduce(add, words)
            w2i = OrderedDict()
            i2w = OrderedDict()

            for i,w in enumerate(chain(special_tokens, count)):
                w2i[w] = i
                i2w[i] = w
            return w2i, i2w

# %%


        def load_embedding_model(model_type: str,embedding_dimension: int) -> gensim.models.keyedvectors.KeyedVectors:
            """
            Loads a pre-trained word embedding model via gensim library.
            :param model_type: name of the word embedding model to load.
            :param embedding_dimension: size of the embedding space to consider
            :return
                - pre-trained word embedding model (gensim KeyedVectors object)
            """

            download_path = ""
            if model_type.strip().lower() == 'glove':
                download_path = "glove-wiki-gigaword-{}".format(embedding_dimension)
            else:
                raise AttributeError("Unsupported embedding model type! Available ones: word2vec, glove, fasttext")

            try:
                emb_model = gloader.load(download_path)
            except ValueError as e:
                raise e
            return emb_model

# %%
        def getPercent(first, second):
            percent = first / second * 100
            return percent

# %% [markdown]
# add old and new vocubulary

# %%

        def combine_vocabularies(old_voc, add_voc):
            """Merges vocabularies keeping consistent indices."""
            voc = old_voc.copy()
            count = 0

            for i, word in enumerate(add_voc.keys()):
                if word not in old_voc.keys():
                    #oov_terms.append(word)
                    voc[word] = count + len(old_voc)
                    count += 1
            return voc

        #v1: tags
        #v2: glove + train
        #v3: glove + train + val
        #v4: glove + train + val + test

# %% [markdown]
# embedding model


    #Builds word-word co-occurrence matrix based on word counts
        def co_occurrence_count(df, idx_to_word, window_size=1, sparse=True):
            vocab_count = len(idx_to_word)

            if sparse:
                co_occurrence_matrix = scipy.sparse.lil_matrix((len(idx_to_word), len(idx_to_word)), dtype=int)
            else:
                co_occurrence_matrix = np.zeros(shape=(vocab_count, vocab_count), dtype='float32')
            co_occurrence_matrix = scipy.sparse.lil_matrix((len(idx_to_word), len(idx_to_word)), dtype=int)
            for doc in tqdm.tqdm(df["tokens"]):
                for i, token in enumerate(doc):
                    window = doc[max(i-window_size, 0) : i+window_size+1]
                    for dd in window:
                        co_occurrence_matrix[token, dd] += 1
            co_occurrence_matrix[np.diag_indices(vocab_count)] = 0 # zeroes diag
            if sparse:
                    return scipy.sparse.csr_matrix(co_occurrence_matrix)
            return co_occurrence_matrix


# %%
        def embedding_model_define(glove,embedding_dimension):
        # download pretrained GloVe embedding
            embedding_dimension = 100 #@param [50, 100, 300] {type:"raw"}
            print("Downloading Glove embedding with dimension:", embedding_dimension)
            print("Be ",int(np.sqrt((embedding_dimension//50 - 1)))*"very ","patient :)",sep='')
            embedding_model = utility.load_embedding_model(glove,embedding_dimension)



# %% [markdown]
# embed oov

        def check_OOV_terms(old_voc, new_voc):
            voc = old_voc.copy()
            oov_terms=[]
            for i, word in enumerate(new_voc.keys()):
                if word not in old_voc.keys():
                    oov_terms.append(word)
            return oov_terms



# %%
        def embedd_OOV_terms(embedding_model, oov_terms, co_occurrence_matrix, w2i, i2w, embedding_dimension,rnd_OOV = False):
            """Embedd OOV words by weighted average of co-occurring neighbors."""
            for i, word in enumerate(oov_terms):
                if rnd_OOV:
                    oov_vec = np.random.rand(embedding_dimension)
                else:
                    oov_vec = np.zeros(embedding_dimension)
                    for count_row in co_occurrence_matrix[w2i[word]]:
                        weights_acc = 0
                        for count, index in zip(count_row.data, count_row.indices):
                            if i2w[index] not in oov_terms:
                                weights_acc += count
                                oov_vec += count*embedding_model[i2w[index]]

                    oov_vec/=weights_acc
                embedding_model.add(word, oov_vec)

            return embedding_model
