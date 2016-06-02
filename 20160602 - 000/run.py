import execute.repair_data as pd
import execute.train_data as td
import execute.classify_data as cd
import execute.segment_data as sd
import libs.store_manage as sm
from config import *
from libs.features import getFeatureList
from libs.models import getModelType, getModelInfo
import datetime

ulogs = []

for i in range(nRun):
    print('=============================================')
    print('Test Case #', i)
    t = datetime.datetime.now()
    timeManager.setTime(str(t.date().strftime('%Y%m%d')) + '_' + str(t.time().strftime('%H%M%S')))

    print(' - Prepairing Data...')
    pd._exec()
    print(' - Training Data...')
    t_acc = td._exec()
    print(' - Classifying Data...')
    cd_acc = cd._exec()
    print(' - Segmenting Data...')
    sd_acc = sd._exec()

    featureName = [i[0] for i in getFeatureList(featureConfig['name'])]
    featureAddress = [i[0] for i in getFeatureList(featureConfig['address'])]
    featurePhone = [i[0] for i in getFeatureList(featureConfig['phone'])]


    ulogs.append([
        timeManager.getTime(),
        str(len(featureName)) + ' features: ' + ', '.join(featureName),
        str(len(featureAddress)) + ' features: ' + ', '.join(featureAddress),
        str(len(featurePhone)) + ' features: ' + ', '.join(featurePhone),
        model_configs['name-model']['type'],
        getModelInfo(model_configs['name-model']),
        t_acc[0],
        model_configs['address-model']['type'],
        getModelInfo(model_configs['address-model']),
        t_acc[1],
        model_configs['phone-model']['type'],
        getModelInfo(model_configs['phone-model']),
        t_acc[2],
        cd_acc,
        sd_acc
    ])


sm.updateLogFile(ulogs)