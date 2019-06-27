<Target Tracking parameters>


### SINGLE Target,  q=0.01###
python run_tracking.py --init_mean 1.0 --init_sd 10 --batch_size 64 --target_update_freq 50  --learning_rate 0.001 --buffer_size 1000 --nb_warmup_steps 100 --nb_epoch_steps 100 --nb_train_steps 5000 --env TargetTracking-v1 --noise 1.0 --num_layers 3 --num_units 128 --render 0 --record 0 --log_dir results/target_tracking/empty/ADFQ-eg --map empty --seed 0

python run_tracking.py --init_mean 1.0 --init_sd 10 --batch_size 64 --target_update_freq 50  --learning_rate 0.001 --buffer_size 1000 --nb_warmup_steps 100 --nb_epoch_steps 100 --nb_train_steps 5000 --env TargetTracking-v1 --noise 1.0 --num_layers 3 --num_units 128 --render 0 --record 0 --log_dir results/target_tracking/empty/ADFQ-ts --act_policy bayesian --scope deepadfq-ts --map empty --seed 0

python run_tracking.py --batch_size 64 --target_update_freq 50  --learning_rate 0.001 --buffer_size 1000 --nb_warmup_steps 100 --nb_epoch_steps 100 --nb_train_steps 5000 --env TargetTracking-v1 --num_layers 3 --num_units 128 --render 0 --record 0 --log_dir ../../results/target_tracking/empty/DQN --map empty --seed 0 

python run_tracking.py --batch_size 64 --target_update_freq 50  --learning_rate 0.001 --buffer_size 1000 --nb_warmup_steps 100 --nb_epoch_steps 100 --nb_train_steps 5000 --env TargetTracking-v1 --num_layers 3 --num_units 128 --render 0 --record 0 --log_dir ../../results/target_tracking/empty/DDQN --map empty --double_q 1 --scope deepq-d --seed 0 


### SINGLE Target,  q=0.01 obstacle02###
python run_tracking.py --init_mean 1.0 --init_sd 10 --batch_size 64 --target_update_freq 50  --learning_rate 0.001 --buffer_size 1000 --nb_warmup_steps 100 --nb_epoch_steps 100 --nb_train_steps 5000 --env TargetTracking-v1 --noise 1.0 --num_layers 3 --num_units 128 --render 0 --record 0 --log_dir results/target_tracking/obstacles02/ADFQ-eg --map obstacles02 --seed 0

python run_tracking.py --init_mean 1.0 --init_sd 10 --batch_size 64 --target_update_freq 50  --learning_rate 0.001 --buffer_size 1000 --nb_warmup_steps 100 --nb_epoch_steps 500 --nb_train_steps 3000 --env TargetTracking-v1 --noise 1.0 --num_layers 3 --num_units 128 --render 1 --record 1 --log_dir results/target_tracking/ros/obstacles02_single --act_policy bayesian --scope deepadfq-ts --map obstacles02 --seed 0

python run_tracking.py --batch_size 64 --target_update_freq 50  --learning_rate 0.001 --buffer_size 1000 --nb_warmup_steps 100 --nb_epoch_steps 100 --nb_train_steps 5000 --env TargetTracking-v1 --num_layers 3 --num_units 128 --render 0 --record 0 --log_dir ../../results/target_tracking/obstacles02/DQN --map obstacles02 --seed 0 

python run_tracking.py --batch_size 64 --target_update_freq 50  --learning_rate 0.001 --buffer_size 1000 --nb_warmup_steps 100 --nb_epoch_steps 100 --nb_train_steps 5000 --env TargetTracking-v1 --num_layers 3 --num_units 128 --render 0 --record 0 --log_dir ../../results/target_tracking/obstacles02/DDQN --map obstacles02 --double_q 1 --scope deepq-d --seed 0 


python run_tracking.py --mode test --env TargetTracking-info0 --render 1 --log_dir results/target_tracking

### SINGLE Target, q=0.01, obstacle02, IMAGE_BASED###
python run_tracking.py --init_mean 1.0 --init_sd 30 --batch_size 64 --target_update_freq 50  --learning_rate 0.001 --buffer_size 1000 --nb_warmup_steps 500 --nb_epoch_steps 500 --nb_train_steps 10000 --env TargetTracking-v5 --noise 1.0 --num_layers 3 --num_units 128 --render 0 --record 0 --log_dir results/target_tracking/ --act_policy bayesian --scope deepadfq-ts --map obstacles02 --seed 0 --im_size 50

MULTI
### This works for target_num = 2, v1###
python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 300 --batch_size 64 --target_update_freq 100 --init_mean 1.0 --init_sd 30 --learning_rate 0.0005 --buffer_size 2000 --nb_train_steps 9000 --log_dir results/target_tracking/multi/target2/ADFQ-eg --nb_targets 2 --env TargetTracking-vMulti --noise 0.01 --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 300 --batch_size 64 --target_update_freq 100 --init_mean 1.0 --init_sd 30 --learning_rate 0.0005 --buffer_size 2000 --nb_train_steps 9000 --log_dir results/target_tracking/multi/target2/ADFQ-ts --nb_targets 2 --env TargetTracking-vMulti --noise 0.01 --act_policy bayesian --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 300 --batch_size 64 --target_update_freq 100 --learning_rate 0.0005 --buffer_size 2000 --nb_train_steps 9000 --log_dir ../../results/target_tracking/multi/target2/DQN --nb_targets 2 --env TargetTracking-vMulti --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --scope deepadfq-ts --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 300 --batch_size 64 --target_update_freq 100 --learning_rate 0.0005 --buffer_size 2000 --nb_train_steps 9000 --log_dir ../../results/target_tracking/multi/target2/DDQN --nb_targets 2 --env TargetTracking-vMulti --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --double_q 1 --scope deepq-d --double_q 1 --seed 0 

### This works for target_num = 3, v1###
python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 300 --batch_size 64 --target_update_freq 100 --init_mean 1.0 --init_sd 30 --learning_rate 0.0005 --buffer_size 2000 --nb_train_steps 9000 --log_dir results/target_tracking/multi/target3/ADFQ-eg --nb_targets 3 --env TargetTracking-vMulti --noise 0.01 --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 300 --batch_size 64 --target_update_freq 100 --init_mean 1.0 --init_sd 30 --learning_rate 0.0005 --buffer_size 2000 --nb_train_steps 9000 --log_dir results/target_tracking/multi/target3/ADFQ-ts --nb_targets 3 --env TargetTracking-vMulti --noise 0.01 --act_policy bayesian --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 300 --batch_size 64 --target_update_freq 100 --learning_rate 0.0005 --buffer_size 2000 --nb_train_steps 9000 --log_dir ../../results/target_tracking/multi/target3/DQN --nb_targets 3 --env TargetTracking-vMulti --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --scope deepadfq-ts --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 300 --batch_size 64 --target_update_freq 100 --learning_rate 0.0005 --buffer_size 2000 --nb_train_steps 9000 --log_dir ../../results/target_tracking/multi/target3/DDQN --nb_targets 3 --env TargetTracking-vMulti --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --double_q 1 --scope deepq-d --double_q 1 --seed 0 


###
python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 600 --batch_size 64 --target_update_freq 100 --init_mean 1.0 --init_sd 30 --learning_rate 0.0005 --buffer_size 2000 --nb_train_steps 9000 --log_dir results/target_tracking/ros/target3 --nb_targets 3 --env TargetTracking-vMulti --noise 0.01 --act_policy bayesian --num_layers 3 --num_units 256 --render 0 --record 0 --map obstacles02 --seed 0


### For final experiment? target num = 2, q=0.001###
python run_anytime_planner.py --env TargetTracking-info1 --nb_targets 2 --map emptySmall --log_dir results/target_tracking/multi/target2/baseline --repeat 10

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 150 --batch_size 64 --target_update_freq 50 --init_mean 1.0 --init_sd 30 --learning_rate 0.0001 --buffer_size 2000 --nb_train_steps 7500 --env TargetTracking-info1 --nb_targets 2 --noise 0.001 --log_dir results/target_tracking/multi/target2/ADFQ-eg --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 150 --batch_size 64 --target_update_freq 50 --init_mean 1.0 --init_sd 30 --learning_rate 0.0001 --buffer_size 2000 --nb_train_steps 7500 --env TargetTracking-info1 --nb_targets 2 --noise 0.001 --log_dir results/target_tracking/multi/target2/ADFQ-ts --act_policy bayesian --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --scope deepadfq-ts --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 150 --batch_size 64 --target_update_freq 50 --learning_rate 0.0001 --buffer_size 2000 --nb_train_steps 7500 --env TargetTracking-info1 --nb_targets 2 --log_dir ../../results/target_tracking/multi/target2/DQN --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 150 --batch_size 64 --target_update_freq 50 --learning_rate 0.0001 --buffer_size 2000 --nb_train_steps 7500 --env TargetTracking-info1 --nb_targets 2 --log_dir ../../results/target_tracking/multi/target2/DDQN --double_q 1 --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --scope deepq-d --seed 0

<Test>
python run_tracking.py --mode test --env TargetTracking-vMulti --log_dir results/target_tracking/multi/TargetTracking-vMulti_02271007 --render 1 --map emptySmall

### This works for target_num = 2  q=0.001 (InfoPlanner Binding)###
python run_tracking.py --nb_warmup_steps 200 --nb_epoch_steps 500 --batch_size 64 --target_update_freq 100 --init_mean 1.0 --init_sd 30 --learning_rate 0.0005 --buffer_size 2000 --nb_train_steps 10000 --log_dir results/target_tracking/multi/target2 --env TargetTracking-info1 --noise 0.01 --act_policy bayesian --num_layers 3 --num_units 256 --render 1 --record 0 --map emptySmall --seed 1 --nb_targets 3


### This works for target_num = 3?###
python run_tracking.py --nb_warmup_steps 200 --nb_epoch_steps 500 --batch_size 64 --target_update_freq 100 --init_mean 1.0 --init_sd 30 --learning_rate 0.0001 --buffer_size 2000 --nb_train_steps 10000 --log_dir results/target_tracking/multi --env TargetTracking-vMulti --noise 0.01 --seed 1 --act_policy bayesian --num_layers 3 --num_units 300 --render 0 --record 0 --map emptySmall

### Static target num = 3, q=0.001 (large unit size is not always better!)### 
python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 150 --batch_size 64 --target_update_freq 50 --init_mean 1.0 --init_sd 30 --learning_rate 0.0001 --buffer_size 2000 --nb_train_steps 7500 --env TargetTracking-info1 --nb_targets 3 --noise 0.001 --log_dir results/target_tracking/multi/target3/ADFQ-eg --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 150 --batch_size 64 --target_update_freq 50 --init_mean 1.0 --init_sd 30 --learning_rate 0.0001 --buffer_size 2000 --nb_train_steps 7500 --env TargetTracking-info1 --nb_targets 3 --noise 0.001 --log_dir results/target_tracking/multi/target3/ADFQ-ts --act_policy bayesian --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --scope deepadfq-ts --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 150 --batch_size 64 --target_update_freq 50 --learning_rate 0.0001 --buffer_size 2000 --nb_train_steps 7500 --env TargetTracking-info1 --nb_targets 3 --log_dir ../../results/target_tracking/multi/target3/DQN --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --seed 0

python run_tracking.py --nb_warmup_steps 150 --nb_epoch_steps 150 --batch_size 64 --target_update_freq 50 --learning_rate 0.0001 --buffer_size 2000 --nb_train_steps 7500 --env TargetTracking-info1 --nb_targets 3 --log_dir ../../results/target_tracking/multi/target3/DDQN --double_q 1 --num_layers 3 --num_units 256 --render 0 --record 0 --map emptySmall --scope deepq-d --seed 0



