import xgboost as xgb
import pandas as pd
import numpy as np

class XGBoostRanker:
    def __init__(self) -> None:
        """Constructor of XGBoostRanker
        """
        self.train_collection = None
        self.test_collection = None

        self.test_docs_ids = None
        self.test_queries = None

        self.model = None

    def set_train_collection(self, X, y, group, n_labels=32, doc_id_column_name='document', query_column_name='query'):
        """
        Set the training collection for the ranker by preparing the feature matrix (X), labels (y), and group information.

        Args:
        -----
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix representing the set of feature values without the labels. The DataFrame should contain
            the document and query identifiers, which will be dropped before creating the DMatrix.
        
        y : pandas.Series or numpy.ndarray
            The array representing the set of labels. These labels will be binned into discrete intervals (default is 32 bins).
        
        group : numpy.ndarray
            The array of group values, which is typically the query IDs for ranking tasks. This defines how the data is grouped.
        
        n_labels : int, optional (default=32)
            The number of bins to discretize the labels into. The labels will be binned into this many intervals.
        
        doc_id_column_name : str, optional (default='document')
            The name of the column in X that contains the document identifiers. This column will be dropped before creating the DMatrix.
        
        query_column_name : str, optional (default='query')
            The name of the column in X that contains the query identifiers. This column will be dropped before creating the DMatrix.

        Returns:
        --------
        None
            This method sets the `train_collection` attribute of the instance, which is an `xgb.DMatrix` object containing the feature matrix and labels for training.
        """
        X = X.drop([doc_id_column_name, query_column_name], axis=1)

        y = pd.cut(y, bins=32, labels=np.arange(n_labels), include_lowest=True).astype(int)
        self.train_collection = xgb.DMatrix(X,label=y)
        self.train_collection.set_group(group)

    def set_test_collection(self, X, y, group, n_labels=32, doc_id_column_name='document', query_column_name='query'):
        """
        Set the test collection for the ranker by preparing the feature matrix (X), labels (y), and group information.
        Also stores the document IDs and queries for later use.

        Args:
        -----
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix representing the set of feature values without the labels. The DataFrame should contain
            the document and query identifiers, which will be stored and then dropped before creating the DMatrix.
        
        y : pandas.Series or numpy.ndarray
            The array representing the set of labels. These labels will be binned into discrete intervals (default is 32 bins).
        
        group : numpy.ndarray
            The array of group values, which is typically the query IDs for ranking tasks. This defines how the data is grouped.
        
        n_labels : int, optional (default=32)
            The number of bins to discretize the labels into. The labels will be binned into this many intervals.
        
        doc_id_column_name : str, optional (default='document')
            The name of the column in X that contains the document identifiers. This column will be stored and then dropped before creating the DMatrix.
        
        query_column_name : str, optional (default='query')
            The name of the column in X that contains the query identifiers. This column will be stored and then dropped before creating the DMatrix.

        Returns:
        --------
        None
            This method sets the `test_collection` attribute of the instance, which is an `xgb.DMatrix` object containing the feature matrix and labels for testing.
            It also stores the document IDs and queries in the `test_docs_ids` and `test_queries` attributes for later use in predictions.
        """
        y = pd.cut(y, bins=32, labels=np.arange(n_labels), include_lowest=True).astype(int)
        
        self.test_docs_ids = X[doc_id_column_name]
        self.test_queries = X[query_column_name]

        X = X.drop([doc_id_column_name, query_column_name], axis=1)
        
        self.test_collection = xgb.DMatrix(X,label=y)
        self.test_collection.set_group(group)

    
    def train_ranker(self, num_rounds, params=None):
        """Train the ranker.

        Args:
            num_rounds (int): Number of boosting rounds during training.
            params (dict, optional): Dictionary of model hyperparameters. Defaults to None.
        """
        if params == None:
            params = {
                'objective': 'rank:ndcg',
                'eval_metric': 'ndcg',
                'learning_rate': 0.01,
                'max_depth': 6,
                'min_child_weight': 20,
                'lambda': 0.1,
                'alpha': 0.2,
                'colsample_bytree': 0.8
            }

        self.model = xgb.train(params, self.train_collection, num_boost_round=num_rounds)


    def predict_ranker(self):
        """Make predictions based on the test collection provided.

        Returns:
            numpy.ndarray: Array of ranking scores.
        """

        return self.model.predict(self.test_collection)
    
    def generate_prediction_dataset(self, doc_id_name='document'):
        """
        Generates a dataset containing query, document, true labels, and predictions.

        Parameters:
        -----------
        doc_id_name : str, optional
            Name of the document identifier column (default is 'document').

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing 'query', 'document', 'trues', and 'predictions' columns.
        """
        df = pd.DataFrame()
        df['query'] = self.test_queries
        df['document'] = self.test_docs_ids
        df['trues_ranking_scores'] = self.test_collection.get_label()
        df['predictions_ranking_scores'] = self.model.predict(self.test_collection)
        
        return df