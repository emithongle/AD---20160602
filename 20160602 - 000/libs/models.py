from sknn.mlp import Classifier, Layer
from libs.features import *

def getModelType():
    return 'Neural-Network'

def buildModel(modelConfig):
    return Classifier(
        layers=[Layer(modelConfig['layers'][i][1], units=modelConfig['layers'][i][0])
                for i in range(len(modelConfig['layers']))],
        learning_rule=modelConfig['learning_rule'],
        learning_rate=modelConfig['learning_rate'],
        n_iter=modelConfig['n_iter']
    )

def getModelInfo(modelConfig):
    if (modelConfig['type'] == 'Neural-Network'):
        return str(len(modelConfig['layers'])) + ' layers: [' + \
                ', '.join([str(n_unit) + '-' + act_func for (n_unit, act_func) in modelConfig['layers']]) + \
                '], learning_rate: ' + str(modelConfig['learning_rate']) + ', learning_rule: ' + modelConfig['learning_rule'] + \
                ', n_iterator: ' + str(modelConfig['n_iter'])
    return ''

def clasify(clfs, text):
    def _f(clf, text, ttype):
        return clf.predict_proba(getFeature([text], ttype))

    probName = _f(clfs['name'], text, 'name')[0]
    probAddress = _f(clfs['address'], text, 'address')[0]
    probPhone = _f(clfs['phone'], text, 'phone')[0]

    proba = [
        probName[0] * probAddress[1] * probPhone[1],
        probName[1] * probAddress[0] * probPhone[1],
        probName[1] * probAddress[1] * probPhone[0],
        probName[1] * probAddress[1] * probPhone[1]
    ]

    # _ = np.exp(proba) / np.sum(np.exp(proba))
    _ = np.asarray(proba) / np.sum(proba)
    return max(range(len(_)), key=_.__getitem__), _

