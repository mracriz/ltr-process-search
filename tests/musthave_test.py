import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

import pandas as pd
from manual_evaluation.must_have import MustHaveDataset
from utils.utils import *

musthave_path = '/Users/david/Downloads/' + 'Must Have GoldenSet - Página1.csv'
musthave = pd.read_csv(musthave_path)

path_goldenset = '/Users/david/Documents/phd/JusBrasilData/sdbn_'

goldenset_copycopy = pd.read_csv(path_goldenset + 'copy_examcopy.csv')
goldenset_copyclick = pd.read_csv(path_goldenset + 'copy_examclick.csv')
goldenset_clickclick = pd.read_csv(path_goldenset + 'click_examclick.csv')

musthave = musthave.rename(columns={'document_id':'doc_id'})

dataset = MustHaveDataset(musthave_df=musthave, query_column_name='query')

#Get intersection between sets
intersection_queries = get_values_of_intersection([goldenset_clickclick, goldenset_clickclick, goldenset_copycopy],
                                                  'query',
                                                  musthave['query'].unique())

goldenset_copycopy = filter_values_by_list(goldenset_copycopy, 'query', intersection_queries)
goldenset_copyclick = filter_values_by_list(goldenset_copyclick, 'query', intersection_queries)
goldenset_clickclick = filter_values_by_list(goldenset_clickclick, 'query', intersection_queries)

print('Copy / examCopy')
dataset.set_goldenset(goldenset_df=goldenset_copycopy, query_column_name='query')
results = dataset.get_relevant_positions()
print("MRR: ", dataset.calculate_mrr(results))

print('Copy / examClick')
dataset.set_goldenset(goldenset_df=goldenset_copyclick, query_column_name='query')
results = dataset.get_relevant_positions()
print("MRR: ", dataset.calculate_mrr(results))

print('Click / examClick')
dataset.set_goldenset(goldenset_df=goldenset_clickclick, query_column_name='query')
results = dataset.get_relevant_positions()
print("MRR: ", dataset.calculate_mrr(results))