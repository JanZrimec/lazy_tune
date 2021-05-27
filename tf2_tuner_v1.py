'''
First min fucntional system = V1:
- parses results, retrives best hps after train
- retrains model with optimizing epochs
- saves retrained model, best hps, final metrics, and best model above (everything)
- is actually sufficient for most tasks, as boxcox lambda can be explored earlier, epchs are optimized after tuning, and only batch_size remains which likely doenst have a huge effect

# todo 
V2 - adds epoch, batch_size and dataset preprocessing:
- https://keras-team.github.io/keras-tuner/tutorials/subclass-tuner/
- https://github.com/keras-team/keras-tuner/issues/122

# config and execution:
-> set config.yaml
source activate dnagan
nohup python tf2_tuner_v1.py &

# checking results from eg. slurm output while executing
less slurm-74445.out | grep val_coef_det_k | grep -Po '(?<=val_coef_det_k:).*' | sort -n
'''

# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TerminateOnNaN
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from my_utils import (coef_det_k,
                     TrainValTensorBoard, 
                     TestCallback)
import tensorflow as tf
print(tf.__version__)

import random,functools
import hashlib,csv,os,re,traceback,inspect,argparse,importlib,yaml,sys,json
import kerastuner as kt
from io import StringIO 

import logging
logging.basicConfig(format='%(asctime)s-%(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)


def load_module(model_path):
    '''loads module containing models given path'''
    spec = importlib.util.spec_from_file_location('module',model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def parse_tuner_results(output_dir):
    fname = (os.path.join(output_dir, 'oracle.json'))
    with open(fname) as f:
        oracle = json.load(f)

    # traverse file structure, extract scores and tags from jsons
    dirs = [x for x in os.listdir(output_dir) if 'trial' in x]
    tmp = []
    for dir1 in dirs:
        fname = (os.path.join(output_dir, dir1, 'trial.json'))
        with open(fname) as f:
            tmp.append(json.load(f))

    df = pd.DataFrame([x['trial_id'] for x in tmp],columns=['trial_id'])
    df['score'] = [x['score'] for x in tmp]

    for key in tmp[0]['metrics']['metrics'].keys():
        df[key] = [x['metrics']['metrics'][key]['observations'][0]['value'][0] for x in tmp]

    return df.sort_values(by='score')

def get_best_model_results(output_dir):
    return parse_tuner_results(output_dir).reset_index(drop=True).iloc[0]

def load_best_model(best_dir):
    model = load_model(best_dir,
                    custom_objects={"coef_det_k": coef_det_k})
    model.summary()
    return model

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def main():
    '''main tuner function'''
    # read in params from file
    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    experiment = config['experiment_name']
    datasets = config['input_files']['datasets']
    models = config['input_files']['models']
    optimizer_iterations = config['params']['optimizer_iterations']
    replicate_seed = config['params']['replicate_seed']
    test_split = config['params']['test_split']
    validation_split = config['params']['validation_split']
    verbose = config['params']['verbose']
    tensorboard = config['params']['tensorboard']
    epochs = config['params']['epochs']
    min_delta = config['params']['min_delta']
    patience = config['params']['patience']
    mbatch = config['params']['mbatch']

    # generate output files dirs
    output_dir = experiment+'/'+'results'
    best_dir = experiment+'/'+'best_model'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    if tensorboard:
        tensorboard_dir = experiment+'/'+'tensorboard'
        os.makedirs(tensorboard_dir, exist_ok=True)

    MODEL = list(models.keys())[0]
    DATASET = list(datasets.keys())[0]
    REPLICATE_SEED = replicate_seed[0]
    BATCH_SIZE = mbatch[0]
    MIN_DELTA = min_delta[0]
    PATIENCE = patience[0]
    EPOCHS = epochs[0]

    logging.info("Config file read and param setup.", exc_info=True)
    
    # setting seeds
    os.environ['PYTHONHASHSEED'] = str(REPLICATE_SEED + 1)
    np.random.seed(REPLICATE_SEED + 2)
    random.seed(REPLICATE_SEED + 3)
    tf.random.set_seed(REPLICATE_SEED + 4)

    # load data
    module = load_module(models[MODEL])
    X_train, X_test, Y_train, Y_test = module.load_data(datasets[DATASET])
    logging.info("Load data.", exc_info=True)
    
    # callbacks
    tcb = TestCallback((X_test, Y_test))
    call_backs = [EarlyStopping(monitor='val_loss',min_delta=MIN_DELTA,patience=PATIENCE),
                  tcb,
                  TerminateOnNaN(), # failsafe
                 ]

    if tensorboard:
        call_backs.append(TrainValTensorBoard(log_dir=os.path.join(args.tensorboard_dir, file_name + hash_string),
                                              histogram_freq=10, write_grads=True))

    print(call_backs)    

    
    # tuner
    tuner = kt.BayesianOptimization(
        functools.partial(module.build_model,
                          input_shape=[X.shape[1:] for X in X_train]),
        objective='val_loss',
        max_trials=optimizer_iterations,
        seed=REPLICATE_SEED,
        directory=experiment,
        project_name='results')

    tuner.search_space_summary()

    logging.info("Setup tuner.", exc_info=True)    
    
    tuner.search(x=X_train,
             y=Y_train,
             validation_split=validation_split,
             batch_size=BATCH_SIZE,
             epochs=EPOCHS,
             shuffle=True,
             callbacks=call_backs
            )

    logging.info("Tuner run.", exc_info=True) 
    
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hps.get('learning_rate')

    # Build the model with the optimal hyperparameters and train it on the data
    model = module.build_model(best_hps,input_shape=(X_train.shape[1]))

    history = model.fit(x=X_train,
                        y=Y_train, 
                        validation_split=validation_split,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        shuffle=True,
                        callbacks=call_backs
                       )

    val_acc_per_epoch = history.history['loss']
    best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    model = module.build_model(best_hps,input_shape=(X_train.shape[1]))

    # Retrain the model
    history = model.fit(x=X_train,
                        y=Y_train, 
                        validation_split=validation_split,
                        batch_size=BATCH_SIZE,
                        epochs=best_epoch,
                        shuffle=True,
                        callbacks=call_backs
                       )

    # evaluate
    eval_result = model.evaluate(X_test, Y_test)
    print(str(['test_'+x for x in model.metrics_names])+':', eval_result)

    logging.info("Optimize/retrain best model.", exc_info=True) 
    
    
    # store results
    #tuner.results_summary()
    with Capturing() as output:
        tuner.results_summary()
    with open(experiment+'/best_model/results_summary.txt', 'w') as file:
        [file.writelines(row+'\n') for row in output]

    # best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(experiment+'/best_model')

    results = (pd.DataFrame(get_best_model_results(output_dir))
               .append(pd.DataFrame(best_model.evaluate(X_test, Y_test),
                                    index=['test_'+x for x in best_model.metrics_names]))
              )
    results.to_csv(experiment+'/best_model/results.csv',header=False)

    df_hps = pd.DataFrame(best_hps.values.values(),index=best_hps.values.keys())
    df_hps.to_csv(experiment+'/best_model/best_hps.csv',header=False)

    # best model retrained and epoch optimized
    model.save(experiment+'/best_model_optimized')

    results = (pd.DataFrame([history.history[key][-1] for key in history.history.keys()],
                          index=list(history.history.keys())) #.transpose()
               .append(pd.DataFrame(eval_result,index=['test_'+x for x in model.metrics_names]))
              )
    results.to_csv(experiment+'/best_model_optimized/results.csv',header=False)

    df_hps = pd.DataFrame(best_hps.values.values(),index=best_hps.values.keys()).transpose()
    df_hps['epochs'] = best_epoch
    df_hps.transpose().to_csv(experiment+'/best_model_optimized/best_hps.csv',header=False)

    logging.info("Save results.", exc_info=True) 
    
    
if __name__ == '__main__':
    main()
    