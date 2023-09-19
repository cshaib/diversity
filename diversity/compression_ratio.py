import gzip 
import os
import lzma as xz

def compression_ratio(path, data, algorithm='gzip'):
    """ Calculates the compression ratio for a collection of text. 

    Args:
        path (str): path to store temporarily zipped files
        data (list): list of strings
        algorithm (str, optional): either 'gzip' or 'xz'. Defaults to 'gzip'.

    Returns:
        float: compression ratio (original size / compressed size)
    """

    with open(path+'original.txt', 'w+') as f:
        f.write(' '.join(data))

    original_size = os.path.getsize(os.path.join(path, "original.txt"))

    if algorithm == 'gzip':

        with gzip.GzipFile(path+'compressed.gz', 'w+') as f:
            f.write(gzip.compress(' '.join(data).encode('utf-8')))

        compressed_size = os.path.getsize(os.path.join(path, "compressed.gz"))

    elif algorithm == 'xz': 

        with xz.open(path+'compressed.gz', 'wb') as f:
            f.write(' '.join(data).encode('utf-8'))

        compressed_size = os.path.getsize(os.path.join(path, "compressed.gz"))


    print(f"Original Size: {original_size}\nCompressed Size: {compressed_size}")

    return original_size / compressed_size
    
