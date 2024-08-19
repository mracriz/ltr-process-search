import pandas as pd
from scipy.stats import kendalltau

def calculate_kendall_tau(data_frames, on='predictions_ranking_scores', query='query', doc_id_col='document'):
    """
    Calculate pairwise Kendall's Tau correlations between predictions from multiple datasets.

    Parameters:
    -----------
    data_frames : list of pandas.DataFrame
        List containing pandas DataFrames, each DataFrame should contain columns 'query', 'predictions',
        and the custom document identifier column specified by `doc_id_col`.

    on : str, optional (default='predictions')
        Column name in each DataFrame representing the predictions to compare.

    query : str, optional (default='qid')
        Column name in each DataFrame representing the query identifier used for grouping.

    doc_id_col : str, optional (default='doc_id')
        Column name in each DataFrame representing the unique document identifier used for sorting.

    Returns:
    --------
    dict
        A dictionary containing pairwise Kendall's Tau correlations between all combinations of datasets.
        Keys are formatted as '{DataFrame1 Name} x {DataFrame2 Name}', where names are extracted from DataFrame objects.
        Values are Kendall's Tau coefficients averaged across query groups, or NaN if no valid comparisons could be made.
    """
    # Extract dataframe names for output keys
    df_names = [f'DF{i+1}' for i in range(len(data_frames))]
    
    # Perform pairwise comparisons
    tau_values = {}
    for i in range(len(data_frames)):
        for j in range(i + 1, len(data_frames)):
            key = f'{df_names[i]} x {df_names[j]}'
            df1 = data_frames[i]
            df2 = data_frames[j]
            merged_df = df1.merge(df2, on=[query, doc_id_col])
            tau = group_kendall_tau(merged_df, f'{on}_x', f'{on}_y', query, doc_id_col)
            tau_values[key] = tau

    return tau_values

def group_kendall_tau(df, col1, col2, query, doc_id_col):
    """
    Calculate Kendall's Tau correlation coefficient between two columns within each query group of a DataFrame,
    handling ties by sorting the subgroup by doc_id_col only for tied values.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing columns 'query', 'col1', 'col2', and `doc_id_col`.

    col1 : str
        Name of the first column for comparison.

    col2 : str
        Name of the second column for comparison.

    query : str
        Column name in DataFrame representing the query identifier used for grouping.

    doc_id_col : str
        Column name in DataFrame representing the unique document identifier used for sorting.

    Returns:
    --------
    float
        Average Kendall's Tau coefficient across query groups, or NaN if no valid comparisons could be made.
    """
    grouped = df.groupby(query)
    tau_values = []
    for name, group in grouped:
        if len(group) > 1:  # Ensure there are at least two items to compare
            # Sort each subgroup by col1 and col2 in descending order
            sorted_group_col1 = group.sort_values(by=[col1], ascending=False)
            sorted_group_col2 = group.sort_values(by=[col2], ascending=False)
            
            # Identify ties based on col1 and col2
            ties_mask = (sorted_group_col1[col1].values == sorted_group_col2[col2].values)
            
            # Sort tied values by doc_id_col
            if any(ties_mask):
                sorted_group_col1.loc[ties_mask, doc_id_col] = sorted_group_col1.loc[ties_mask].sort_values(by=[doc_id_col])[doc_id_col].values
                sorted_group_col2.loc[ties_mask, doc_id_col] = sorted_group_col2.loc[ties_mask].sort_values(by=[doc_id_col])[doc_id_col].values
            
            # Calculate Kendall's Tau for the sorted subgroups
            tau, _ = kendalltau(sorted_group_col1[doc_id_col], sorted_group_col2[doc_id_col])
            tau_values.append(tau)
    
    return sum(tau_values) / len(tau_values) if tau_values else float('nan')

