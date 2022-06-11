## This file has all the augmentation done for Task1
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import random

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from random import shuffle


def augment_text(df, samples=150000, pr=0.2):
    new_text = []
    esci_labels = []

    ## selecting the minority class samples
    df_n=df[df.target==1].reset_index(drop=True)

    product_titles = []

    ## data augmentation loop
    for i in tqdm(np.random.randint(0, len(df), samples)):
        text = df.iloc[i]['query']
        esci_label = df.iloc[i]['esci_label']
        product_title = df.iloc[i]['product_title']
        syn_aug = naw.SynonymAug(aug_src='wordnet')
        augmented_text = syn_aug.augment(text)
        new_text.append(augmented_text)
        product_titles.append(product_title)
        esci_labels.append(esci_label)

    esci_label2gain = {'exact': 1.0, 'substitute': 0.1, 'complement': 0.01, 'irrelevant': 0.0}

    ## dataframe
    new = pd.DataFrame({'query': new_text, 'product_title': product_titles, 'esci_label': esci_labels})
    new['gain'] = new['esci_label'].apply(lambda label: esci_label2gain[label])
    df = shuffle(df.append(new).reset_index(drop=True))

    return new

