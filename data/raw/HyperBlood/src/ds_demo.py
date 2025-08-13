'''
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences https://www.iitis.pl

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
************************************************************************

HSI blood classification dataset by M. Romaszewski, P.Glomb, M. Cholewa, A. Sochan 
Institute of Theoretical and Applied Informatics, Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
Dataset DOI: 10.5281/zenodo.3984905

DEMO: classification of a dataset image

@author: mromaszewski@iitis.pl
'''
import numpy as np
import matplotlib.pyplot as plt
from ds_load import get_data,get_anno,get_rgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def get_Xy(data,anno):
    """
    return data as 2D arrays (useful e.g. for applying sklearn functions)
    
    Parameters:
    ---------------------
    data: data cube as nparray
    annotation: 2d annotation array
    
    Returns:
    -----------------------
    X: 2d array (no. pixels x no.bands)
    y: labels for pixels 
    """
    X = data.reshape(-1,data.shape[2])
    y = anno.reshape(-1)
    return X,y

def demo():
    """
    Demo: classification of an example HSI image 
    classifier: SVM(RBF), 2% of pixels used for training 
    """
    name = 'E(1)'
    data,wav = get_data(name)
    anno = get_anno(name)
    
    X_raw,y_raw = get_Xy(data,anno)
    X = X_raw[y_raw!=0]
    y = y_raw[y_raw!=0]
    
    train_index, test_index = next(StratifiedShuffleSplit(test_size=0.98,random_state=42).split(X, y))
    
    svc = SVC(kernel='rbf',gamma='scale',C=1.0)
    svc.fit(X[train_index],y[train_index])
    y_pred = svc.predict(X[test_index])
    acc = accuracy_score(y[test_index], y_pred)
    print ("accuracy score: {:0.2f}".format(acc))
    
    y_pred = svc.predict(X_raw)
    y_pred[y_raw==0]=0
    plt.subplot(1,3,1)
    rgb  = get_rgb(data,wav)
    plt.imshow(rgb,interpolation='nearest')
    plt.title("Ground truth")
    plt.subplot(1,3,2)
    plt.imshow(anno,interpolation='nearest')
    plt.title("Classes")
    plt.subplot(1,3,3)
    plt.imshow(y_pred.reshape(anno.shape),interpolation='nearest')
    plt.title("Prediction, accuracy: {:0.2f}".format(acc))
    plt.tight_layout()
    plt.show()
    plt.close()
    
        
if __name__ == '__main__':
    demo()
