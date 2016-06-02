from config import *
from libs.store import saveFile, loadFile, saveDataFrame, loadDataFrame
import numpy as np
import pandas as pd

def loadOriginData():

    name = np.array(loadFile(folders['data'], files['data']['name']))
    address = np.array(loadFile(folders['data'], files['data']['address']))
    phone = np.array(loadFile(folders['data'], files['data']['phone']))
    nothing = np.array(loadFile(folders['data'], files['data']['nothing']))

    _f = lambda x : x[1:]

    name[0], address[0], phone[0], nothing[0] = _f(name[0]), _f(address[0]), _f(phone[0]), _f(nothing[0])

    return (name, address, phone, nothing)

def saveTrainingTestingData(train, test):
    _f = lambda y, x: pd.DataFrame(np.asarray([y] + x.T.tolist()).T)

    saveDataFrame(_f(train['name']['y'], train['name']['X']), folders['feature'], files['feature']['name']['train'])
    saveDataFrame(_f(train['address']['y'], train['address']['X']), folders['feature'], files['feature']['address']['train'])
    saveDataFrame(_f(train['phone']['y'], train['phone']['X']), folders['feature'], files['feature']['phone']['train'])

    saveDataFrame(_f(test['name']['y'], test['name']['X']), folders['feature'], files['feature']['name']['test'])
    saveDataFrame(_f(test['address']['y'], test['address']['X']), folders['feature'], files['feature']['address']['test'])
    saveDataFrame(_f(test['phone']['y'], test['phone']['X']), folders['feature'], files['feature']['phone']['test'])

def loadTrainingTestingData():
    _f = lambda x: {'X': x[:, 1:].astype(float), 'y': x[:, 0].astype(int)}

    train = {
        'name': _f(loadDataFrame(folders['feature'], files['feature']['name']['train']).as_matrix()),
        'address': _f(loadDataFrame(folders['feature'], files['feature']['address']['train']).as_matrix()),
        'phone': _f(loadDataFrame(folders['feature'], files['feature']['phone']['train']).as_matrix())
    }

    test = {
        'name': _f(loadDataFrame(folders['feature'], files['feature']['name']['test']).as_matrix()),
        'address': _f(loadDataFrame(folders['feature'], files['feature']['address']['test']).as_matrix()),
        'phone': _f(loadDataFrame(folders['feature'], files['feature']['phone']['test']).as_matrix())
    }

    return train, test

def saveModel(models):
    saveFile(models['name'], folders['model'], timeManager.getTime() + '_' + files['model']['name'])
    saveFile(models['address'], folders['model'], timeManager.getTime() + '_' + files['model']['address'])
    saveFile(models['phone'], folders['model'], timeManager.getTime() + '_' + files['model']['phone'])

    updateModelConfig({
        'model-name': timeManager.getTime(),
        'name': model_configs['name-model'],
        'address': model_configs['address-model'],
        'phone': model_configs['phone-model'],
    })

def loadModel():
    return {
        'name': loadFile(folders['model'], timeManager.getTime() + '_' + files['model']['name']),
        'address': loadFile(folders['model'], timeManager.getTime() + '_' + files['model']['address']),
        'phone': loadFile(folders['model'], timeManager.getTime() + '_' + files['model']['phone']),
    }

def loadAllModel():
    None

def saveTermTestResult(data):
    saveFile(data, folders['result'], timeManager.getTime() + '_' + files['result']['term'])

def loadTermTest():
    _f = lambda x: {'X': x[:, 1:], 'y': x[:, 0].astype(int)}
    return _f(loadDataFrame(folders['data-test'], files['data-test']['term']).as_matrix())

def saveTemplateTestResult(data):
    saveFile(data, folders['result'], timeManager.getTime() + '_' + files['result']['segment'])

def loadFullAddress():
    _ = np.asarray(loadFile(folders['data-test'], files['data-test']['full-address']))
    yName, yAddress, yPhone = _[:, 0].tolist(), _[:, 1].tolist(), _[:, 2].tolist()
    X = _[:, 3].tolist()
    return X, yName, yAddress, yPhone

def updateLogFile(data):
    saveFile({'logs': loadFile(folders['log'], files['log']) + data}, folders['log'], files['log'])

def saveHistogram(data):
    saveFile(data, folders['result'], timeManager.getTime() + '_' + files['result']['histogram'])

def updateModelConfig(models):
    mc = loadFile(folders['model'], files['model']['model-config'])
    mc['name'][models['model-name']] = models['name']
    mc['address'][models['model-name']] = models['address']
    mc['phone'][models['model-name']] = models['phone']
    saveFile(mc, folders['model'], files['model']['model-config'])