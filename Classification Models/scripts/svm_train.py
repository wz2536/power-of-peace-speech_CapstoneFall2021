import argparse
import os
import pandas as pd
import numpy as np
import logging
import s3fs

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
# from sklearn.model_selection import GridSearchCV

import sklearn.externals
import joblib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default='s3://compressed-data-sample/train_embedding_yfirst.csv')

    args = parser.parse_args()

    s3 = s3fs.S3FileSystem(anon=False)
    train = pd.read_csv(s3.open(args.train, mode='rb'), header=None)

    # labels are in the first column
    X_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values

    # Setup SVM pipeline
    
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('pca', PCA()),
                     ('svc', SVC())])
#     param_grid = {
#         "pca__n_components": [int(i) for i in np.linspace(100, 768, num=8)],
#         "svc__kernel": ['linear', 'poly', 'rbf']
#     }
#     search = GridSearchCV(pipe, param_grid, n_jobs=-1)    
    pipe.fit(X_train, y_train)
#     print("Best parameter (CV score=%0.3f):" % search.best_score_)
    

    # Print the coefficients of the trained classifier, and save the coefficients
    model_location = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(pipe, model_location)
    logging.info(f'Stored trained model at {model_location}')