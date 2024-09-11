def read_text_file(filename):
    """
    Function to read raw text file line-by-line and return list of sentence.
    """
    with open(filename, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    
    return lines