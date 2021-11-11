import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl
import xgboost as xgb
import s3fs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=4)
    parser.add_argument('--min_child_weight', type=int, default=6)
    parser.add_argument('--subsample', type=float, default=0.7)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--eval_metric', type=str, default='auc')
    parser.add_argument('--verbosity', type=int, default=1)
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--train', type=str, default='s3://compressed-data-sample/train_embedding_yfirst.csv')
    parser.add_argument('--validation', type=str, default='s3://compressed-data-sample/test_embedding_yfirst.csv')
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    args = parser.parse_args()
    
    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'verbosity': args.verbosity,
        'objective': args.objective,
        'eval_metric': args.eval_metric
    }

    # Transform to matrix, specify data columns and label columns
    s3 = s3fs.S3FileSystem(anon=False)
    train = pd.read_csv(s3.open(args.train, mode='rb'), header=None)
    dtrain = xgb.DMatrix(train.iloc[:, 1:], label=train.iloc[:, 0])
    
    if args.validation is not None:
        val =  pd.read_csv(s3.open(args.validation, mode='rb'), header=None)
        dval = xgb.DMatrix(val.iloc[:, 1:], label=val.iloc[:, 0])
    else:
        dval = None
    
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]
    
    bst = xgb.train(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        # callbacks=[xgb.callback.LearningRateScheduler(custom_rates)]
    )
    
    model_location = args.model_dir + '/xgboost-model'
    pkl.dump(bst, open(model_location, 'wb'))
    logging.info("Stored trained model at {}".format(model_location))