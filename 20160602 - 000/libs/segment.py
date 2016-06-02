import itertools as it
import string

def segmentText(text):
    terms = text.strip(' \n,.').split()

    ls = []
    for cb in list(it.combinations(range(len(terms) - 1), 2)):
        term1 = ' '.join(terms[:cb[0]+ 1]).strip()
        term2 = ' '.join(terms[cb[0]+1:cb[1]+1]).strip()
        term3 = ' '.join(terms[cb[1]+1:]).strip()
        _ = (1 if term1[-1] in string.punctuation else 0) + \
            (1 if term2[-1] in string.punctuation else 0) + \
            (1 if term3[-1] in string.punctuation else 0)
        if (_ >= 2):
            ls.append([term1, term2, term3])

    return ls