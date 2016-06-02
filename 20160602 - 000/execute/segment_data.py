from libs.store_manage import loadModel, loadFullAddress, saveTemplateTestResult, saveHistogram
import libs.segment as sgm
from libs.models import clasify
from libs.features import getFeatureList, getFeature, preprocess
from config import featureConfig
from math import log
import numpy as np
import matplotlib.pyplot as plt

def _exec():
    _f = lambda x: x + \
                   getFeature(x, 'name')[0].tolist() + \
                   getFeature(x, 'address')[0].tolist() + \
                   getFeature(x, 'phone')[0].tolist()

    X, yName, yAddress, yPhone = loadFullAddress()
    models = loadModel()

    goodresults = []
    badresults = []
    for i, x in zip(range(len(X)), X):
        goodresults.append([])
        badresults.append([])
        templateList = sgm.segmentText(x)
        for j, tm in enumerate(templateList):
            label_0, score_0 = clasify(models, tm[0])
            label_1, score_1 = clasify(models, tm[1])
            label_2, score_2 = clasify(models, tm[2])
            totalScore = log(score_0.max()) + log(score_1.max()) + log(score_2.max())

            typleInfo = sorted(zip(tm, [label_0, label_1, label_2], [score_0, score_1, score_2]), key=lambda _: _[1])

            row = [i, j] + [_[0] for _ in typleInfo] + [_[1] for _ in typleInfo] + \
                  [typleInfo[0][2].max()] + [typleInfo[1][2].max()] + [typleInfo[2][2].max()] + [totalScore] + \
                  _f([preprocess(typleInfo[0][0])]) + typleInfo[0][2].tolist() + \
                  _f([preprocess(typleInfo[1][0])]) + typleInfo[1][2].tolist() + \
                  _f([preprocess(typleInfo[2][0])]) + typleInfo[2][2].tolist()

            if ([_[1] for _ in typleInfo] == list(range(3))):
                goodresults[i].append(row)
            else:
                badresults[i].append(row)

        goodresults[i] = sorted(goodresults[i], key=lambda x: x[11], reverse=True)

    acc, ranks, goodresults = checkSegmentResults((yName, yAddress, yPhone), goodresults)
    saveTemplateResults({'name': yName, 'address': yAddress, 'phone': yPhone}, acc, goodresults, badresults)
    saveHistogramResult(ranks)

    return acc

def saveTemplateResults(labels, acc, goodresults, badresults):
    _f = lambda l, prefix: [prefix + _[0] for _ in l]

    featureNames = _f(getFeatureList(featureConfig['name']), 'Name_') + \
                   _f(getFeatureList(featureConfig['address']), 'Address_') + \
                    _f(getFeatureList(featureConfig['phone']), 'Phone_')

    _score =  ['NameScore', 'AddressScore', 'PhoneScore', 'NothingScore']

    titleG = ['TestCase', 'Top', 'Name', 'Address', 'Phone', 'Corrected',
             'NameScore', 'AddressScore', 'PhoneScore', 'Score'] + \
            ['PrepName'] + featureNames + _score + \
            ['PrepAddress'] + featureNames + _score +\
            ['PrepPhone'] + featureNames + _score

    titleB = ['TestCase', 'Top', 'Term_0', 'Term_1', 'Term_2', 'Label_0', 'Label_1', 'Label_2',
              'Score_0', 'Score_1', 'Score_2'] + \
             ['PrepName'] + featureNames + _score + \
             ['PrepAddress'] + featureNames + _score + \
             ['PrepPhone'] + featureNames + _score

    data = {'GoodResults': [['Accuracy = ', acc], titleG], 'BadResults': [titleB]}

    for tc in goodresults:
        for i, row in enumerate(tc):
            data['GoodResults'] += [[row[0]] + [i] + row[2:6] + row[9:]]

    for tc in badresults:
        for i, row in enumerate(tc):
            data['BadResults'] += [[row[0]] + [i] + row[2:11] + row[12:]]


    saveTemplateTestResult(data)

    None

def checkSegmentResults(y, results):
    ranks = {0: 0}

    _f = lambda x: x.strip(' .,')

    for sgs, yName, yAddress, yPhone,  in zip(enumerate(results), y[0], y[1], y[2]):
        print('  -> Test Case #', sgs[0])
        if (len(sgs[1]) > 0):
            for idx, row in enumerate(sgs[1]):
                if (_f(row[2]) == _f(yName) and
                    _f(row[3]) == _f(yAddress) and
                    _f(row[4]) == _f(yPhone)):
                    ranks[idx] = 1 if (idx not in ranks) else ranks[idx] + 1
                    results[sgs[0]][idx] = row[:5] + [1] + row[5:]
                else:
                    results[sgs[0]][idx] = row[:5] + [0] + row[5:]

    return ranks[0]/len(results), ranks, results


def saveHistogramResult(_x, save=True, name='image'):
    if (sum(_x.values()) > 0):
        _ = []
        for key, value in _x.items():
            _ += [key] * value
        x = np.asarray(_)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # the histogram of the data
        n, bins, patches = ax.hist(x, 100)

        ax.set_xlabel('Top')
        ax.set_ylabel('Probability')
        ax.set_title(name)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, max(n))
        ax.grid(True)

        saveHistogram(plt)
        # if (save):
        #     plt.savefig(name + '.png')

