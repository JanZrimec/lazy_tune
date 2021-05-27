# lazy_tuner

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
