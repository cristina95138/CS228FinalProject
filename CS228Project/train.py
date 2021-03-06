# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import argparse
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from augmentation import augment_text
import re

def clean_text(text):
    pattern=re.compile(r'<[^>]+>',re.S)
    text=pattern.sub('',text)
    return text

def standardize_units(text):
  text = " " + text + " "
  text = re.sub('( gal | gals | galon )',' gallon ',text)
  text = re.sub('( ft | fts | feets | foot | foots )',' feet ',text)
  text = re.sub('( squares | sq )',' square ',text)
  text = re.sub('( lb | lbs | pounds )',' pound ',text)
  text = re.sub('( oz | ozs | ounces | ounc )',' ounce ',text)
  text = re.sub('( yds | yd | yards )',' yard ',text)
  return text

def preprocess(text):
    text=text.replace('in.',' inch ')
    words=re.split(r'\W+',text)
    words=[word.lower() for word in words]
    res=re.sub("[A-Za-z]+",lambda ele: " " + ele[0] + " ",' '.join(words))
    res=standardize_units(res)
    cleaned=' '.join(res.split())
    return cleaned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path_file", type=str, help="Input training CSV with the pairs of queries and products.")
    parser.add_argument("product_catalogue_path_file", type=str, help="Input product catalogue CSV.")
    parser.add_argument("locale", type=str, choices=['us', 'es', 'jp'], help="Locale of the queries.")
    parser.add_argument("model_save_path", type=str, help="Directory to save the model.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_dev_queries", type=int, default=200, help="Number of development examples.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    """ 0. Init variables """
    col_query_id = "query_id"
    col_query = "query"
    col_query_locale = "query_locale"
    col_esci_label = "esci_label"
    col_product_id = "product_id"
    col_product_title = "product_title"
    col_product_locale = "product_locale"
    esci_label2gain = {
        'exact': 1.0,
        'substitute': 0.1,
        'complement': 0.01,
        'irrelevant': 0.0,
    }
    col_gain = 'gain'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ 1. Load data """
    df = pd.read_csv(args.train_path_file)
    df_product_catalogue = pd.read_csv(args.product_catalogue_path_file)
    df = df[df[col_query_locale] == args.locale]
    df_product_catalogue = df_product_catalogue[df_product_catalogue[col_product_locale] == args.locale]
    df = pd.merge(
        df,
        df_product_catalogue,
        how='left',
        left_on=[col_product_id, col_query_locale],
        right_on=[col_product_id, col_product_locale],
    )
    df = df[df[col_product_title].notna()]
    list_query_id = df[col_query_id].unique()
    dev_size = args.n_dev_queries / len(list_query_id)
    list_query_id_train, list_query_id_dev = train_test_split(list_query_id, test_size=dev_size,
                                                              random_state=args.random_state)
    df[col_gain] = df[col_esci_label].apply(lambda label: esci_label2gain[label])
    df = df[[col_query_id, col_query, col_product_title, col_gain,col_esci_label]]
    df_train = df[df[col_query_id].isin(list_query_id_train)]
    df_dev = df[df[col_query_id].isin(list_query_id_dev)]

    if args.locale == 'us':
        df_train['product_title'] = df_train.apply(lambda x: preprocess(x.product_title), axis=1)  ### added line
        df_train['query'] = df_train.apply(lambda x: preprocess(x.query), axis=1)  ### added line
        augmented_text=augment_text(df_train)
        df_train=df_train[col_query,col_product_title, col_esci_label,col_gain]
        df_train = pd.concat([df_train, augmented_text], ignore_index=True, sort=False)

    """ 2. Prepare data loaders """
    train_samples = []
    for (_, row) in df_train.iterrows():
        train_samples.append(InputExample(texts=[row[col_query], row[col_product_title]], label=float(row[col_gain])))
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size, drop_last=True)
    if args.locale == "us":
        dev_samples = {}
        query2id = {}
        for (_, row) in df_dev.iterrows():
            try:
                qid = query2id[row[col_query]]
            except KeyError:
                qid = len(query2id)
                query2id[row[col_query]] = qid
            if qid not in dev_samples:
                dev_samples[qid] = {'query': row[col_query], 'positive': set(), 'negative': set()}
            if row[col_gain] > 0:
                dev_samples[qid]['positive'].add(row[col_product_title])
            else:
                dev_samples[qid]['negative'].add(row[col_product_title])
        evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

        """ 3. Prepare Cross-enconder model:
            https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_kd.py
        """
        model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
        num_epochs = 1
        num_labels = 1
        max_length = 512
        default_activation_function = torch.nn.Identity()
        model = CrossEncoder(
            model_name,
            num_labels=num_labels,
            max_length=max_length,
            default_activation_function=default_activation_function,
            device=device
        )
        loss_fct = torch.nn.MSELoss()
        evaluation_steps = 5000
        warmup_steps = 5000
        lr = 7e-6
        """ 4. Train Cross-encoder model """
        model.fit(
            train_dataloader=train_dataloader,
            loss_fct=loss_fct,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=f"{args.model_save_path}_tmp",
            optimizer_params={'lr': lr},
            scheduler='warmupcosine',
        )
        model.save(args.model_save_path)
        evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='sts-test')
        evaluator(model)
    else:
        dev_queries = df_dev[col_query].to_list()
        dev_titles = df_dev[col_product_title].to_list()
        dev_scores = df_dev[col_gain].to_list()
        evaluator = evaluation.EmbeddingSimilarityEvaluator(dev_queries, dev_titles, dev_scores)

        """ 3. Prepare sentence transformers model: 
            https://www.sbert.net/docs/training/overview.html 
        """
        model_name = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
        model = SentenceTransformer(model_name)
        train_loss = losses.CosineSimilarityLoss(model=model)
        num_epochs = 1
        evaluation_steps = 1000
        """ 4. Train Sentence transformer model """
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            output_path=args.model_save_path,
        )
        evaluator(model)


if __name__ == "__main__":
    main()