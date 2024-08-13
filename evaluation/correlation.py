import pandas as pd
from scipy.stats import kendalltau

def calculate_kendall_tau(data_frames, on='predictions', query='qid', doc_id_col='doc_id'):
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

    Examples:
    ---------
    # Example DataFrames
    df1 = pd.DataFrame({
        'qid': [1, 1, 2, 2],
        'predictions': [0.8, 0.6, 0.7, 0.5],
        'custom_doc_id': [101, 102, 201, 202]
    })

    df2 = pd.DataFrame({
        'qid': [1, 1, 2, 2],
        'predictions': [0.7, 0.5, 0.9, 0.4],
        'custom_doc_id': [101, 102, 201, 202]
    })

    df3 = pd.DataFrame({
        'qid': [1, 1, 2, 2],
        'predictions': [0.9, 0.4, 0.8, 0.6],
        'custom_doc_id': [101, 102, 201, 202]
    })

    # Calculate Kendall's Tau correlations
    data_frames = [df1, df2, df3]
    result = calculate_kendall_tau(data_frames, on='predictions', query='qid', doc_id_col='custom_doc_id')
    print(result)
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
    Calculate Kendall's Tau correlation coefficient between two columns within each query group of a DataFrame.

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
            sorted_group_col1 = group.sort_values(by=[col1], ascending=False)
            sorted_group_col2 = group.sort_values(by=[col2], ascending=False)
            tau, _ = kendalltau(sorted_group_col1[doc_id_col], sorted_group_col2[doc_id_col])
            tau_values.append(tau)
    return sum(tau_values) / len(tau_values) if tau_values else float('nan')
