import nltk

def get_pos(data):
    """ Turns a sequence into parts of speech 

    Args:
        data (list): data to tranform into POS tags

    Returns:
        str, tuple: str of POS tags, tuple of (word, pos)
    """


    pos_tuples = [nltk.pos_tag(x.split()) for x in data]

    joined_pos = []
    joined_text = []

    for doc in pos_tuples:
        joined_text.append(' '.join([x[0] for x in doc]))
        joined_pos.append(' '.join([x[1] for x in doc]))

    return joined_pos, pos_tuples


def _find_sub_list(sl,l):
    """_summary_

    Args:
        sl (_type_): _description_
        l (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []
    sll = len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind, ind+sll-1))

    return results


def pos_patterns(text, pattern): 
    """ Matches a pattern to a text and returns the text.

    Args:
        text (_type_): _description_
        pattern (_type_): _description_
    """

    pos = []
    word = []
    
    # text is a list of lists of tuples (word, part of speech)
    for doc in text: 
        pos.append([i[1] for i in doc])
        word.append([i[0] for i in doc])
    
    pos = [' '.join(x) for x in pos]
    word = [' '.join(x) for x in word]

    # return positions of each tag and the corresponding tokens
    for w, p in zip(word, pos):

        test = _find_sub_list(pattern.split(), p.split())


        if test:
            for occ in test: 
                print(w.split()[int(occ[0]):int(occ[1]+1)])
