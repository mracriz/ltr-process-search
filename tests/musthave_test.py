import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

import pandas as pd
from manual_evaluation.must_have import MustHaveDataset
from utils.utils import *

musthave_path = '/Users/david/Downloads/' + '[Search] Busca Híbrida (Jusberto vs. ADA) - Av. objetiva.csv'
musthave = pd.read_csv(musthave_path,header=1)
goldenset = pd.read_csv('/Users/david/Downloads/goldenset_musthave.csv')

musthave = musthave[~musthave['id'].isna()]
musthave['id'] = musthave['id'].astype(int).astype(str)

musthave = musthave[~musthave['id.1'].isna()]
musthave['id.1'] = musthave['id.1'].astype(int).astype(str)

musthave = musthave.rename(columns={'Consulta':'query', 'id.1':'doc_id'})

musthave = musthave[ (musthave['Relevância.1'] == 'Relevante') & (musthave['Acertou?.1'] == 'Sim')]

musthave['doc_id'] = musthave['doc_id'].apply(lambda x: 'JURISPRUDENCE:' + x)

musthave['query'] = musthave['query'].apply(normalize_query)

dataset = MustHaveDataset(musthave_df=musthave, query_column_name='query')
dataset.set_goldenset(goldenset_df=goldenset, query_column_name='query')
dataset.get_commom_queries()

results = dataset.get_relevant_positions()

for query, dict in results.items():
    print(query)
    for doc_id, position in dict.items():
        print(doc_id, position)
    print()
