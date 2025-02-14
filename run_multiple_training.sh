python inspect_model.py --epochs 100 --n_steps 200 --batch_size 5000 --Np 512 --Ng 4096 --box_width 2.2 > experiments/logs/o1.txt 2>&1;
python inspect_model.py --epochs 100 --n_steps 200 --batch_size 5000 --Np 256 --Ng 4096 --box_width 2.2 > experiments/logs/o4.txt 2>&1; # less Np
# python inspect_model.py --epochs 100 --n_steps 200 --batch_size 5000 --Np 128 --Ng 4096 --box_width 2.2 > experiments/logs/o5.txt 2>&1; # less Np


# python inspect_model.py --epochs 100 --n_steps 200 --batch_size 5000 --Np 512 --Ng 2048 --box_width 2.2 > experiments/logs/o2.txt 2>&1; # less Ng
# python inspect_model.py --epochs 100 --n_steps 200 --batch_size 5000 --Np 512 --Ng 1024 --box_width 2.2 > experiments/logs/o3.txt 2>&1; # less Ng


python inspect_model.py --epochs 100 --n_steps 200 --batch_size 5000 --Np 512 --Ng 4096 --box_width 0.635 > experiments/logs/o6.txt 2>&1; # smaller box, pc_ref is scaled, velocity is scaled


# try scaling the number of place cells instead of the velocity of the agent,
# so that in a sequence the agent traverses the same number of place cells
# python inspect_model.py --epochs 100 --n_steps 200 --batch_size 5000 --Np 43 --Ng 4096 --box_width 0.635 --original_velocity > experiments/logs/o9.txt 2>&1; # if squared relationship


# python inspect_model.py --epochs 100 --n_steps 200 --batch_size 5000 --Np 512 --Ng 4096 --box_width 2.2 --override_pc_ref 0.2 > experiments/logs/o7.txt 2>&1; # bigger pc ref
# python inspect_model.py --epochs 100 --n_steps 200 --batch_size 5000 --Np 512 --Ng 4096 --box_width 2.2 --override_pc_ref 0.07 > experiments/logs/o8.txt 2>&1; # smaller pc ref



########### RIAB

# python inspect_model_riab.py --n_exp 500 --epochs 2 --n_steps 0 --batch_size 5000 --Np 512 --Ng 4096 --box_width 0.635 > experiments/logs/1.txt 2>&1; # smaller box, pc_ref is scaled, velocity is scaled
# python inspect_model_riab.py --n_exp 1000 --epochs 2 --n_steps 0 --batch_size 5000 --Np 512 --Ng 4096 --box_width 0.635 > experiments/logs/2.txt 2>&1; # smaller box, pc_ref is scaled, velocity is scaled

# python inspect_model_riab.py --n_exp 500 --epochs 2 --n_steps 0 --batch_size 5000 --Np 512 --Ng 1024 --box_width 0.635 > experiments/logs/3.txt 2>&1; # less experiments/logs, less Ng

python inspect_model_riab.py --n_exp 100 --epochs 5 --n_steps 0 --batch_size 5000 --Np 512 --Ng 8192 --box_width 0.635 > experiments/logs/4.txt 2>&1; # less experiments/logs, less Ng
