import re
import pandas as pd

def normalize_query(query):
    # Convert to lowercase and remove accents
    query = query.lower()
    query = query.translate(str.maketrans("àáâãäåçèéêëìíîïñòóôõöùúûü", "aaaaaaceeeeiiiinooooouuuu"))
    
    # Remove dots not adjacent to numbers
    query = re.sub(r'(\b\d+\.\d+\b)|\.', r'\1', query)
    
    # Remove slashes not adjacent to numbers
    query = re.sub(r'(\b\d+\/\d+\b)|\/', r'\1', query)
    
    # Remove all but non-alphanumeric characters except quotes, dot, dash, slash, and paragraph mark
    query = re.sub(r'[^a-zA-Z0-9"§\.\-\/]', ' ', query)
    
    # Remove duplicate spaces
    query = re.sub(r' +', ' ', query)
    
    # Return the normalized query
    return query

def get_values_of_intersection(dataframes, column_name, values_list):
    unique_values = set()  # Use a set to store unique values
    
    # Iterate through each DataFrame and extract unique values from column_name
    for df in dataframes:
        if column_name in df.columns:  # Ensure column_name exists in the DataFrame
            unique_values.update(df[df[column_name].isin(values_list)][column_name].unique())
    
    # Convert set of unique values to a list
    unique_values_list = list(unique_values)
    
    return unique_values_list


def filter_values_by_list(df, column_name, list):
    return df[df[column_name].isin(list)]