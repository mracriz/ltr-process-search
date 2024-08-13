import pandas as pd
from model_training.xgboost import XGBoostRanker

path_train = '/Users/david/Documents/phd/JusBrasilData/intersection/train/'
path_test = '/Users/david/Documents/phd/JusBrasilData/intersection/test/'

dataset_type = input('Which dataset?: ')

if dataset_type == 'copyclick':
    file_train = path_train + 'sdbn_copy_examclick_features_train.csv'
    file_test = path_test + 'sdbn_copy_examclick_features_test.csv'

elif dataset_type == 'copycopy':
    file_train = path_train + 'sdbn_copy_examcopy_features_train.csv'
    file_test= path_test + 'sdbn_copy_examcopy_features_test.csv'

elif dataset_type == 'clickclick':
    file_train = path_train + 'sdbn_click_examclick_features_train.csv'
    file_test = path_test + 'sdbn_click_examclick_features_test.csv'

else:
    print('Dataset not found')

def load_read_and_process_collection(file):

    df = pd.read_csv(file)
    
    # Get all column names
    all_features = df.columns.tolist()

    # Columns to remove
    values_to_remove = ['action_prob','shown', 'avg_position', 'examined', 'downvotes','upvotes'] + ['params_query', 'params_query_ner_acronyms', 'params_query_ner_courts',
                        'params_query_ner_datetimes', 'params_query_ner_factual_content',
                        'params_query_ner_identities', 'params_query_ner_normative_acts',
                        'params_query_ner_organizations', 'params_query_ner_persons',
                        'params_query_ner_phrasal_terms', 'params_query_ner_precedents',
                        'params_query_ner_synonyms', 'query_tokens']

    # Select feature columns
    features = [x for x in all_features if x not in values_to_remove]
    df = df[features]

    df  = df.fillna(0)

    return df['beta_action_prob'],  df.drop(['beta_action_prob'], axis=1), df['query'].value_counts().sort_index().values

y_train, X_train, train_groups = load_read_and_process_collection(file_train)
y_test, X_test, test_groups = load_read_and_process_collection(file_test)

ranker = XGBoostRanker()

ranker.set_train_collection(X_train, y_train, train_groups)
ranker.set_test_collection(X_test, y_test, test_groups)

ranker.train_ranker(num_rounds=100)

print(ranker.generate_prediction_dataset())