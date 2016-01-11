
Command to run the program:

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32 nohup python /home/fujie/keras_experiment/incremental_bo/spearmint/spearmint/main.py --driver=local --method=RandomChooser --method-args=noiseless=0 --polling-time=20 --max-concurrent=1 -w --port=50000 --max-finished-jobs=100 --mode=get_hyper_from_output /home/fujie/keras_experiment/random/hyperopt_experiment_9/spear_config.pb &

```