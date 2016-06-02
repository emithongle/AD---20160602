from config import *
import string
import numpy as np
from libs.utils import *
import re

fFeatures = {
    'length': lambda x: len(x),

    '#ascii': lambda x: sum([1 for c in x if c in string.ascii_letters]),

    '#digit': lambda x: sum([1 for c in x if c in string.digits]),

    '#punctuation': lambda x: sum([1 for c in x if c in string.punctuation]),

    '%ascii-adp': lambda x: \
        -1 if sum([1 for c in x if c in string.ascii_letters + string.digits + string.punctuation]) == 0
        else
            sum([1 for c in x if c in string.ascii_letters]) /
            sum([1 for c in x if c in string.ascii_letters + string.digits + string.punctuation]),

    '%digit-adp': lambda x: \
        -1 if sum([1 for c in x if c in string.ascii_letters + string.digits + string.punctuation]) == 0
        else
            sum([1 for c in x if c in string.digits]) /
            sum([1 for c in x if c in string.ascii_letters + string.digits + string.punctuation]),

    '%punctuation-adp': lambda x: \
        -1 if sum([1 for c in x if c in string.ascii_letters + string.digits + string.punctuation]) == 0
        else
            sum([1 for c in x if c in string.punctuation]) /
            sum([1 for c in x if c in string.ascii_letters + string.digits + string.punctuation]),

    'digit-adp/ascii-adp': lambda x: \
        1 if (sum([1 for c in x if c in string.ascii_letters]) == 0) else
            sum([1 for c in x if c in string.digits]) /
            sum([1 for c in x if c in string.ascii_letters]),

    '%ascii': lambda x: sum([1 for c in x if c in string.ascii_letters]) / len(x),
    '%digit': lambda x: sum([1 for c in x if c in string.digits]) / len(x),
    '%punctuation': lambda x: sum([1 for c in x if c in string.punctuation]) / len(x),

    '%keyword-name': lambda x: pctMatchingKeyword(x, nameTermSet),
    '%keyword-address': lambda x: pctMatchingKeyword(x, addressTermSet),
    '%keyword-phone': lambda x: pctMatchingKeyword(x, phoneTermSet),

    'bfirst-term-address': lambda x: 0 if len(x) == 0 else 1 if x.split()[0] in addressTermSet else 0,

    '#max-digit-skip-all-punctuation': lambda x: findMaxString(x, 0)[1],
    'b#max-digit-skip-all-punctuation >= 8': lambda x: 1 if len(findMaxString(x, 0)[1]) >= 8 else 0,
    '%max-digit-skip-all-punctuation': lambda x: findMaxString(x, 0)[1]/len(x),

    '#max-digit-skip-space-&-dot': lambda x: findMaxString(x, 0, skip_punctuation)[1],
    '%max-digit-skip-space-&-dot': lambda x: findMaxString(x, 0, skip_punctuation)[1] / len(x),

    'bfirst-character-digit': lambda x: 0 if len(x) == 0 else 1 if x[0] in string.digits else 0,
    'bfirst-character-ascii': lambda x: 0 if len(x) == 0 else 1 if x[0] in string.ascii_letters else 0,
    'bsecond-character-digit': lambda x: 0 if len(x) < 2 else 1 if x[1] in string.digits else 0,
    'bsecond-character-ascii': lambda x: 0 if len(x) < 2 else 1 if x[1] in string.ascii_letters else 0,
    'blast-character-digit': lambda x: 0 if len(x) == 0 else 1 if x[-1] in string.digits else 0,
    'blast-character-ascii': lambda x: 0 if len(x) == 0 else 1 if x[-1] in string.ascii_letters else 0,

    'b#ascii >= 7': lambda x: 1 if sum([1 for c in x if c in string.ascii_letters]) >= 7 else 0,
    'b#digit >= 8': lambda x: 1 if sum([1 for c in x if c in string.digits]) >= 8 else 0,

    'b,': lambda x: 1 if ',' in x else 0,
    'b/': lambda x: 1 if '/' in x else 0,
    'b+': lambda x: 1 if '+' in x else 0,
    'b()': lambda x: 2 if ('(' in x and ')' in x) else 1 if ('(' in x or ')' in x) else 0,

    '1/length': lambda x: 0 if len(x) == 0 else 1/len(x),

    'bkeyword-address-beginning': lambda x: 0 if len(x) == 0 else 1 if x.split()[0] in addressBeginningTermSet else 0,
    'bStreet-Term': lambda x: containTerm(x, streetTermSet),
    'bWardDistrict-Term': lambda x: containTerm(x, wardDistrictTermSet),
    'bCity-Term': lambda x: containTerm(x, cityTermSet)
}


def getFeatureList(featureList):
    fls = []
    for _ in featureList:
        if (_[1]):
            if (type(_[0]).__name__ == 'list'):
                for __ in _[0]:
                    fls.append((__, _[1]))
            else:
                fls.append(_)
    return fls

def getFeature(texts, ttype):
    fls = getFeatureList(featureConfig[ttype])
    return np.asarray([extractFeature(preprocess(_), fls) for _ in texts])

def preprocess(text):
    # print(text.encode())
    # if (preprocessing_name['convert unicode to ascii']):
    for i in range(len(unic)):
        text = text.replace(unic[i], asi[i])

    # if (preprocessing_name['remove break line']):
    text = text.replace('\n', '')

    # if (preprocessing_name['convert to lower']):
    text = text.lower()

    # if (preprocessing_name['remove multiple spaces']):
    text = re.sub(' +', ' ', text)

    # if (preprocessing_name['trim "space" and ","']):
    text = text.strip(rm_preprocessed_punctuation)

    # if (preprocessing_name['space after punctuation']):
    for i in range(len(text) - 1):
        if (text[i] in string.punctuation and text[i + 1] != ' '):
            text = text[:i + 1] + ' ' + text[i + 1:]

    return text

def extractFeature(text, fl):
    return [fFeatures[fName](text) for (fName, fFun) in fl]
