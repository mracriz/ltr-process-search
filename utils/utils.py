import re

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