from libs.store_manage import loadModel, loadTermTest, saveTermTestResult
from libs.models import clasify
from libs.features import preprocess, getFeatureList, getFeature
from config import featureConfig

def _exec():
    termData = loadTermTest()
    models = loadModel()

    acc = 0

    for (x, l) in zip(termData['X'], termData['y']):
        label, prob = clasify(models, preprocess(x[0]))
        acc += 1 if label == l else 0

    acc = acc / termData['X'].shape[0]

    saveFileForReview(termData, models)

    return acc

def saveFileForReview(termData, models):

    _f = lambda l: [i[0] for i in l]

    ftName = ['Name_' + i for i in _f(getFeatureList(featureConfig['name']))]
    ftAddress = ['Address_' + i for i in _f(getFeatureList(featureConfig['address']))]
    ftPhone = ['Phone_' + i for i in _f(getFeatureList(featureConfig['phone']))]

    data = [['Text', 'Label', 'Predicted', 'Error', 'Preprocessed'] + \
            ['%ProbName', '%ProbNotName'] + ['%ProbAddress', '%ProbNotAddress'] + ['%ProbPhone', '%ProbNotPhone'] + \
            ['%PName', '%PAddress', '%PPhone', '%PNothing'] + ftName + ftAddress + ftPhone]
    for x, y in zip(termData['X'], termData['y']):
        prepx = preprocess(x[0])
        y_hat, prob = clasify(models, prepx)

        _ftName = getFeature([prepx], 'name')
        probName = models['name'].predict_proba(_ftName)[0].tolist()
        _ftAddress = getFeature([prepx], 'address')
        probAddress = models['address'].predict_proba(_ftAddress)[0].tolist()
        _ftPhone = getFeature([prepx], 'phone')
        probPhone = models['phone'].predict_proba(_ftPhone)[0].tolist()

        data.append([x[0], y, y_hat, 1 if y != y_hat else 0, prepx] + \
                    probName + probAddress + probPhone + prob.tolist() + \
                    _ftName[0].tolist() + _ftAddress[0].tolist() + _ftPhone[0].tolist())

    saveTermTestResult({'sheet_1': data})
