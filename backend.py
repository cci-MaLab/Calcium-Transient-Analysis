# This is where all the ultility functions are stored

# The first function is to load in the data

def load_data(path, **kwargs):
    # Load the data in as necessary
    
    # Load in kwargs
    day = kwargs.get('day', None)
    group = kwargs.get('group', None)