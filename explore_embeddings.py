'''
Rhydian Windsor 01/01/20

Used to explore the emebddings generated by the contrastive networks
'''

import sys, os, glob
import pickle
import torch
import numpy as np
from sacred import Experiment
import matplotlib.pyplot as plt
from tqdm import tqdm


ex = Experiment('ExploreEmbeddings')
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."

@ex.config
def config():
    # path to the saved embeddings
    EMBEDDINGS_PATH_MAX = 'saved_test_embeddings/contrastive_test_embeddings_ContrastiveModelsFixedAug_MaxPool.pkl'
    EMBEDDINGS_PATH_AVG = 'saved_test_embeddings/contrastive_test_embeddings_ContrastiveModelsNoJitterNoZoom.pkl'

@ex.capture
def load_embeddings(EMBEDDINGS_PATH:str) -> (torch.Tensor, torch.Tensor):
    """ Returns dictionary containing test embeddings """
    with open(EMBEDDINGS_PATH, 'rb') as f:
        saved_embeddings = pickle.load(f)
        dxa_embeds = saved_embeddings['dxa_embeds'].cpu()
        mri_embeds = saved_embeddings['mri_embeds'].cpu()
    return dxa_embeds, mri_embeds

@ex.automain
def main(EMBEDDINGS_PATH_AVG, EMBEDDINGS_PATH_MAX):

    plt.figure(figsize=(15,10))
    for label in ['Max', 'Average']:
        if label =='Max': path = EMBEDDINGS_PATH_MAX
        elif label =='Average': path = EMBEDDINGS_PATH_AVG
        else: print(f'Error')
        dxa_embeds, mri_embeds = load_embeddings(path)

        # plot cumulative error between embeddings
        pair_distances = (dxa_embeds - mri_embeds).norm(dim=-1)
        non_pair_distances = (dxa_embeds.unsqueeze(1) - mri_embeds.unsqueeze(0)).norm(dim=-1).flatten()
        pair_fracs = []
        non_pair_fracs = []
        vals = np.linspace(0,2,1000)
        for val in tqdm(vals):
            pair_frac = (pair_distances < val).sum().item()/len(pair_distances)
            non_pair_frac = (non_pair_distances < val).sum().item()/len(non_pair_distances)
            pair_fracs.append(pair_frac)
            non_pair_fracs.append(non_pair_frac)

        plt.plot(vals, pair_fracs, label=f'Matching Pairs, {label}')
        plt.plot(vals, non_pair_fracs, label=f'Non Matching Pairs, {label}')
    plt.legend()
    plt.ylabel('Fraction $|x_{MRI}-x_{DXA}| < d$')
    plt.xlabel('Distance, $d$')
    plt.title('Scan Pair Seperation')
    plt.show()



    

