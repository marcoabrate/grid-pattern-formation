import platform, socket
if 'mac' in platform.platform():
    BASE_DIR = "/Users/marco/phd/vr_to_pc/"
    DATA_DIR = "/Users/marco/phd/data/vr_to_pc"
elif 'calcolatore' in socket.gethostname():
    BASE_DIR = "/home/marco/phd/vr_to_pc/"
    DATA_DIR = "/home/marco/phd/data/vr_to_pc"
else:
    BASE_DIR = "/home/marco/vr_to_pc/"
    DATA_DIR = "/media/data/marco/vr_to_pc"

import sys
sys.path.append(BASE_DIR)


import numpy as np
import torch

from tqdm import tqdm
from matplotlib import pyplot as plt



# If GPUs available, select which to train on
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[4]:


from utils import generate_run_ID, load_trained_weights
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer


# In[5]:


# Training options and hyperparameters
class Options:
    pass
options = Options()

options.save_dir = 'experiments'
options.n_steps = 100 # 100_000     # number of training steps
options.batch_size = 5_000 # 200     # number of trajectories per batch
options.sequence_length = 20  # number of steps in trajectory
options.learning_rate = 1e-4  # gradient descent learning rate
options.Np = 512              # number of place cells
options.Ng = 4096             # number of grid cells
options.place_cell_rf = np.round(0.17*(0.635/2.2), 5) # 0.12  # width of place cell center tuning curve (m)
options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
options.RNN_type = 'RNN'      # RNN or LSTM
options.activation = 'relu'   # recurrent nonlinearity
options.weight_decay = 1e-4   # strength of weight decay on recurrent weights
options.DoG = True            # use difference of gaussians tuning curves
options.periodic = False      # trajectories with periodic boundary conditions
options.box_width = 0.635 # 2.2       # width of training environment
options.box_height = 0.635 # 2.2      # height of training environment

# additional options which were not given, but necessary
options.device = 'cuda' if torch.cuda.is_available() else 'cpu'

options.run_ID = generate_run_ID(options)


# In[6]:


print("original total number of datapoints")
print(f"{100_000 * 200 * options.sequence_length * 1_000:,f}") # where 1_000 is the number of epochs


# In[7]:


print("current total number of datapoints (w/ similar results)")
print(f"{options.n_steps * options.batch_size * options.sequence_length * 100:,f}") # where 100 is the number of epochs


# In[8]:


# If you've trained with these params before, will restore trained model
place_cells = PlaceCells(options)
model = RNN(options, place_cells).cuda()
trajectory_generator = TrajectoryGenerator(options, place_cells)
trainer = Trainer(options, model, trajectory_generator)

# # Load example weights stored on github
# weight_dir = options.save_dir + '/example_trained_weights.npy'
# load_trained_weights(model, trainer, weight_dir)


# # My Data

# In[9]:


import os
import numpy as np

load_dirs = [
    os.path.join(
        DATA_DIR, 'box', 'run',
        f"exp_dim0.635_fps50_s3600_seed{s:04d}"
    ) for s in range(1, 1_000+1)
]

thetas, positions, velocities, rot_velocities =\
    [], [], [], []
print("\n[*] Loading simulations from:")
for ld in load_dirs:
    print(ld)
    # load riab data
    thetas.append(np.expand_dims(
        np.load(os.path.join(ld, "riab_simulation/thetas.npy")).astype(np.float32),
        axis=-1
    ))
    positions.append(np.load(os.path.join(ld, "riab_simulation/positions.npy")).astype(np.float32))
    velocities.append(np.load(os.path.join(ld, "riab_simulation/velocities.npy")).astype(np.float32))
    rot_velocities.append(np.expand_dims(
        np.load(os.path.join(ld, "riab_simulation/rot_velocities.npy")).astype(np.float32),
        axis=-1
    ))


# In[10]:


vel = np.stack(velocities, axis=0)
pos = np.stack(positions, axis=0)
hd = np.stack(thetas, axis=0)
rot_vel = np.stack(rot_velocities, axis=0)

print(vel.shape, pos.shape, hd.shape, rot_vel.shape)


# In[11]:


positions = []
thetas = []

subsample = 10
# for idx in range(subsample):
#     positions.append(pos[:, idx::subsample, ...])
#     thetas.append(hd[:, idx::subsample, ...])
# pos = np.concatenate(positions)
# hd = np.concatenate(thetas)

pos = pos[:, ::subsample, ...]
hd = hd[:, ::subsample, ...]

print(pos.shape, hd.shape)


# In[13]:


from simulation.riab_simulation.utils import calculate_rot_velocity

rot_vel = np.apply_along_axis(
    calculate_rot_velocity,
    axis=1,
    arr=hd
)

vel = np.concatenate([
    np.diff(pos, axis=1),
    np.zeros((pos.shape[0], 1, 2))
], axis=1).astype(np.float32)

import torch

class WindowedPredictionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        velocity, rot_velocity, positions,
        window_size
    ):
        self.n_exp = len(velocity)
        self.windows_in_exp = velocity.shape[1] - window_size - 1

        self.velocity = torch.from_numpy(velocity)
        # self.rot_velocity = torch.from_numpy(rot_velocity)

        self.positions = torch.from_numpy(positions)
        self.positions -= (options.box_width / 2)

        self.window_size = window_size

    def __getitem__(self, index):
        # Ensure that the index is within the range of the dataset
        if not (0 <= index < len(self)):
            raise ValueError("Index out of range")
        
        exp = index // self.windows_in_exp
        index_in_exp = index % self.windows_in_exp

        vel = self.velocity[exp]
        # rot_vel = self.rot_velocity[exp]
        pos = self.positions[exp]

        window_slice = (index_in_exp, index_in_exp + self.window_size)        
        # v = torch.concatenate(
        #     [vel[window_slice[0]:window_slice[1], ...],
        #     rot_vel[window_slice[0]:window_slice[1], ...]],
        #     axis=-1
        # )
        vel = vel[window_slice[0]:window_slice[1], ...]
        init_pos = pos[window_slice[0], ...][None, ...]
        pos = pos[window_slice[0]+1:window_slice[1]+1, ...]

        return vel, pos, init_pos

    def __len__(self):
        return self.n_exp * self.windows_in_exp


# In[18]:


dataloader_train = torch.utils.data.DataLoader(
    WindowedPredictionDataset(
        vel[:-1], rot_vel[:-1], pos[:-1], window_size=options.sequence_length
    ),
    shuffle=True, batch_size=options.batch_size
)

print("\tDataloader length:", len(dataloader_train))
print()
for i, batch in enumerate(dataloader_train):
    if i == 0:
        v, p, ip = batch

        print(f"\tBATCH {i}")
        print('\t', v.shape, v.dtype)
        print('\t', p.shape, p.dtype)
        print('\t', ip.shape, ip.dtype)
        print()

        break
    
print()
print("total number of datapoints (w/ rat in a box)")
print(f"{len(dataloader_train) * options.batch_size * options.sequence_length:,f}")


# In[19]:


dataloader_test = torch.utils.data.DataLoader(
    WindowedPredictionDataset(
        vel[-1:], rot_vel[-1:], pos[-1:], window_size=options.sequence_length
    ),
    shuffle=True, batch_size=options.batch_size
)


# # Task statistics

# In[20]:


# Plot a few sample trajectories
# inputs, pos, pc_outputs = trajectory_generator.get_test_batch()
inputs, pos, pc_outputs = trajectory_generator.get_test_batch_mine(dataloader_test)
us = place_cells.us.cpu()
pos = pos.cpu()

plt.figure(figsize=(5,5))
plt.scatter(us[:,0], us[:,1], c='lightgrey', label='Place cell centers')
for i in range(5):
    plt.plot(pos[:,i,0],pos[:,i,1], label='Simulated trajectory', c='C1')
    if i==0:
        plt.legend();

plt.savefig(os.path.join(trainer.ckpt_dir, 'simulated_trajectory.png'))
plt.close()

# In[21]:


# Plot a few place cell outputs
pc_outputs = pc_outputs.reshape(-1, options.Np).detach().cpu()
pc = place_cells.grid_pc(pc_outputs[::100], res=100)

plt.figure(figsize=(16,2))
for i in range(8):
    plt.subplot(1,8,i+1)
    plt.imshow(pc[i], cmap='jet')
    plt.axis('off')
        
plt.suptitle('Place cell outputs', fontsize=16)
plt.savefig(os.path.join(trainer.ckpt_dir, 'place_cells.png'))
plt.close()

# trainer.train(n_epochs=100, n_steps=options.n_steps, save=True)
trainer.train_mine(n_epochs=100, dl=dataloader_train, save=True)

plt.figure(figsize=(12,3))
plt.subplot(121)
plt.plot(trainer.err, c='black')

plt.title('Decoding error (m)'); plt.xlabel('train step')
plt.subplot(122)
plt.plot(trainer.loss, c='black');
plt.title('Loss'); plt.xlabel('train step');

plt.savefig(os.path.join(trainer.ckpt_dir, 'training_curves.png'))
plt.close()


