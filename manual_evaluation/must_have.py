import pandas as pd

class MustHaveDataset():
    def __init__(self, musthave_df, query_column_name = 'query') -> None:
        self.musthave_df = musthave_df
        self.goldenset_df = None
        
        self.musthave_query_column_name = query_column_name
        self.goldesent_query_column_name = None
        
        self.common_queries = None

    def set_goldenset(self, goldenset_df, query_column_name):
        self.goldenset_df = goldenset_df
        self.goldesent_query_column_name = query_column_name

    def get_commom_queries(self):
        common_queries = pd.merge(self.musthave_df[[self.musthave_query_column_name]], 
                                  self.goldenset_df[[self.goldesent_query_column_name]], 
                                  on=self.musthave_query_column_name)
        common_query_count = common_queries['query'].nunique()

        self.common_queries = common_queries['query'].unique().tolist()

        print(f"Number of common queries: {common_query_count}")
        print("Common queries:")
        print(common_queries['query'].unique())

    def get_relevant_positions(self, relevance_score_column = 'beta_action_prob', musthave_doc_id_column = 'doc_id', goldenset_doc_id_column = 'doc_id'):

        results = {}
        
        for query in self.common_queries:
            
            subset = self.goldenset_df[self.goldenset_df[self.goldesent_query_column_name] == query]
            subset = subset.sort_values(by=relevance_score_column,ascending=False).reset_index(drop=True)

            musthave_docs = self.musthave_df[self.musthave_df[self.musthave_query_column_name] == query][musthave_doc_id_column]

            query_position_results = {}

            for doc_id in musthave_docs:
                position = subset[subset[goldenset_doc_id_column] == doc_id].index.to_list()

                if position:
                    query_position_results[doc_id] = position[0] + 1
                else:
                    query_position_results[doc_id] = None

            results[query] = query_position_results
            
        return results

