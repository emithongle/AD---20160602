import libs.store_manage as sm
import numpy as np
from sklearn.cross_validation import train_test_split
from config import testSize, classLabels
from libs.features import getFeature
from libs.store_manage import saveTrainingTestingData

def _exec():
    orgName, orgAddress, orgPhone, orgNothing = sm.loadOriginData()

    def mergeData(ls1, l1, ls2, l2):
        if (ls1.shape[0] < ls2.shape[0]):
            ls2 = ls2[np.random.randint(0, ls2.shape[0]-1, ls1.shape[0])]
        elif (ls1.shape[0] > ls2.shape[0]):
            ls1 = ls1[np.random.randint(0, ls1.shape[0] - 1, ls2.shape[0])]
        return np.append(ls1, ls2, axis=0), [l1] * len(ls1) + [l2] * len(ls2)

    # Split data for training & Testing

    X_trName, X_tName, y_trName, y_tName = train_test_split(orgName, [classLabels['name']] * orgName.shape[0])
    X_trAddress, X_tAddress, y_trAddress, y_tAddress= train_test_split(orgAddress,
                                                                       [classLabels['address']] * orgAddress.shape[0])
    X_trPhone, X_tPhone, y_trPhone, y_tPhone = train_test_split(orgPhone,
                                                                [classLabels['phone']] * orgPhone.shape[0])
    X_trNothing, X_tNothing, y_trNothing, y_tNothing= train_test_split(orgNothing,
                                                                       [classLabels['nothing']] * orgNothing.shape[0])

    # Training Data

    X4TrainNameText, y4TrainName =  mergeData(X_trName,
                       classLabels['name'],
                       np.asarray(X_trAddress.tolist() + X_trPhone.tolist() + X_trNothing.tolist()),
                       classLabels['nothing'])

    X4TrainAddressText, y4TrainAddress = mergeData(X_trAddress,
                       classLabels['address'],
                       np.asarray(X_trName.tolist() + X_trPhone.tolist() + X_trNothing.tolist()),
                       classLabels['nothing'])

    X4TrainPhoneText, y4TrainPhone = mergeData(X_trPhone,
                       classLabels['phone'],
                       np.asarray(X_trAddress.tolist() + X_trName.tolist() + X_trNothing.tolist()),
                       classLabels['nothing'])

    # Testing Data

    X4TestNameText, y4TestName = mergeData(X_tName,
                                           classLabels['name'],
                                           np.asarray(X_tAddress.tolist() + X_tPhone.tolist() + X_tNothing.tolist()),
                                           classLabels['nothing'])

    X4TestAddressText, y4TestAddress = mergeData(X_tAddress,
                                         classLabels['address'],
                                         np.asarray(X_tName.tolist() + X_tPhone.tolist() + X_tNothing.tolist()),
                                         classLabels['nothing'])

    X4TestPhoneText, y4TestPhone = mergeData(X_tPhone,
                                         classLabels['phone'],
                                         np.asarray(X_tAddress.tolist() + X_tName.tolist() + X_tNothing.tolist()),
                                         classLabels['nothing'])

    # Extract Features

    X4TrainNameFeature = getFeature(X4TrainNameText, 'name')
    X4TrainAddressFeature = getFeature(X4TrainAddressText, 'address')
    X4TrainPhoneFeature = getFeature(X4TrainPhoneText, 'phone')

    X4TestNameFeature = getFeature(X4TestNameText, 'name')
    X4TestAddressFeature = getFeature(X4TestAddressText, 'address')
    X4TestPhoneFeature = getFeature(X4TestPhoneText, 'phone')

    saveTrainingTestingData(
        {
            'name': {
                'X': X4TrainNameFeature,
                'y': y4TrainName
            },
            'address': {
                'X': X4TrainAddressFeature,
                'y': y4TrainAddress
            },
            'phone': {
                'X': X4TrainPhoneFeature,
                'y': y4TrainPhone
            }
        },
        {
            'name': {
                'X': X4TestNameFeature,
                'y': y4TestName
            },
            'address': {
                'X': X4TestAddressFeature,
                'y': y4TestAddress
            },
            'phone': {
                'X': X4TestPhoneFeature,
                'y': y4TestPhone
            }
        }
    )
