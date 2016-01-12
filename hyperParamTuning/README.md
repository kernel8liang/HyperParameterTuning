1. path needed to added into .bashrc file to run the keras spearmint experiment:

export HYPEROPT_PATH='/home/jie/docker_folder/random_keras/hyperopt_experiment_1'
export EXPERI_PROJECT_PATH='.../hyper_parameter_tuning'

2. run the main.py in spearmint/spearmint/main.py using following command:

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32 nohup python
/Users/yumengyin/Desktop/hyper_parameter_tuning/hyperParamTuning/HyperParamTuning.py
--driver=local --method=RandomChooser --method-args=noiseless=0 --polling-time=20
--max-concurrent=1 -w --port=50000 --max-finished-jobs=100 --mode=generate .../spear_config.pb &

```
3. spear_config.pb and hyperyaml.yaml need to modified accordingly to adjust the hyperparameter that user want to tune



THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 nohup python /home/jie/d3/fujie/copy1/hyper_parameter_tuning/hyperParamTuning/HyperParamTuning.py --driver=local --method=RandomChooser --method-args=noiseless=0 --polling-time=20 --max-concurrent=1 -w --port=50000 --max-finished-jobs=100 --mode=generate /home/jie/d3/fujie/copy1/hyper_parameter_tuning/hyperParamTuning/experiment/1/11/example/spear_config.pb &


python /Users/yumengyin/Desktop/hyper_parameter_tuning/hyperParamTuning/HyperParamTuning.py --driver=local --method=RandomChooser --method-args=noiseless=0 --polling-time=20 --max-concurrent=1 -w --port=50000 --max-finished-jobs=100 --mode=generate /Users/yumengyin/Desktop/hyper_parameter_tuning/hyperParamTuning/experiment/1/11/example/spear_config.pb
