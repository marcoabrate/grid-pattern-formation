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

from matplotlib import pyplot as plt



# If GPUs available, select which to train on
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"


from utils import generate_run_ID
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer
import argparse
import cv2

def read_video_files_lq(video_dir, frame_dim, sigma=2, remove_bg=False):
    files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]

    print(f'{video_dir}: {len(files)} files')
    files = sorted(files)
    
    video_vector = np.zeros((len(files), 1, frame_dim[1], frame_dim[0]), dtype=np.float32)

    for idx, f in enumerate(files):
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if remove_bg:
            im = im[20:,:]
        im = cv2.GaussianBlur(im, ksize=(0,0), sigmaX=sigma)
        im = cv2.resize(im, frame_dim, interpolation=cv2.INTER_AREA)
        video_vector[idx, 0, ...] = (im / 255)
    
    print(f'\tShape {video_vector.shape}')
    
    return video_vector

def main(args):
    # Training options and hyperparameters
    class Options:
        pass
    options = Options()

    original_pc_ref = 0.12
    original_box_width = 2.2

    options.behaviour = args.behaviour
    options.n_exp = args.n_exp
    options.epochs = args.epochs # number of epochs
    options.n_steps = args.n_steps # number of training steps
    options.batch_size = args.batch_size # number of trajectories per batch
    options.Np = args.Np # number of place cells
    options.Ng = args.Ng # number of grid cells
    options.box_width = args.box_width # width of training environment
    options.box_height = options.box_width # height of training environment
    options.learning_rate = args.learning_rate # gradient descent learning rate
    options.weight_decay = args.weight_decay # strength of weight decay on recurrent weights

    options.save_dir = 'experiments'
    options.sequence_length = 20  # number of steps in trajectory
    options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
    options.RNN_type = 'RNN'      # RNN or LSTM
    options.activation = 'relu'   # recurrent nonlinearity
    options.DoG = True            # use difference of gaussians tuning curves
    options.periodic = False      # trajectories with periodic boundary conditions

    adjust_scale = options.box_width / original_box_width

    if args.override_pc_ref is None:
        options.place_cell_rf = original_pc_ref * adjust_scale # width of place cell center tuning curve (m)
    else:
        options.place_cell_rf = args.override_pc_ref
    
    if args.original_velocity:
        options.velocity = 0.13 * 2 * np.pi # 0.817 forward velocity rayleigh dist scale (m/sec)
    else:
        options.velocity = 0.13 * 2 * np.pi * adjust_scale

    print(f"place cell rf: {options.place_cell_rf}")
    print(f"velocity: {options.velocity}")
    print()

    # additional options which were not given, but necessary
    options.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    options.run_ID = generate_run_ID(options, is_riab=True, is_video=True)

    print("original total number of datapoints")
    print(f"{100_000 * 200 * options.sequence_length * 1_000:,f}") # where 1_000 is the number of epochs

    # If you've trained with these params before, will restore trained model
    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).cuda()
    trajectory_generator = TrajectoryGenerator(options, place_cells)
    trainer = Trainer(options, model, trajectory_generator)


    # # My Data

    load_dirs = [
        os.path.join(
            DATA_DIR, 'box', args.behaviour,
            f"exp_dim0.635_fps17_s3600_seed{s:03d}"
        ) for s in range(100, 100+args.n_exp)
    ]

    videos, thetas, positions, velocities, rot_velocities =\
        [], [], [], [], []
    args.frame_dim = [128, 64]
    print("\n[*] Loading simulations from:")
    for i, ld in enumerate(load_dirs):
        print(ld)
        # load riab data
        video_lq = read_video_files_lq(
            os.path.join(ld, 'box_messy'), [args.frame_dim[0]//4, args.frame_dim[1]//4]
        )
        videos.append(video_lq.reshape(video_lq.shape[0], -1))
        
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

    vid = np.stack(videos, axis=0)
    vel = np.stack(velocities, axis=0)
    pos = np.stack(positions, axis=0)
    hd = np.stack(thetas, axis=0)
    rot_vel = np.stack(rot_velocities, axis=0)
    print(vid.shape, vel.shape, pos.shape, hd.shape, rot_vel.shape)
    print()

    class WindowedPredictionDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            video, velocity, rot_velocity, positions,
            window_size
        ):
            self.n_exp = len(velocity)
            self.windows_in_exp = velocity.shape[1] - window_size - 1

            self.video = torch.from_numpy(video)
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

            vid = self.video[exp]
            vel = self.velocity[exp]
            # rot_vel = self.rot_velocity[exp]
            pos = self.positions[exp]

            window_slice = (index_in_exp, index_in_exp + self.window_size)        
            # v = torch.concatenate(
            #     [vel[window_slice[0]:window_slice[1], ...],
            #     rot_vel[window_slice[0]:window_slice[1], ...]],
            #     axis=-1
            # )
            vid = vid[window_slice[0]:window_slice[1], ...]
            vel = vel[window_slice[0]:window_slice[1], ...]
            init_pos = pos[window_slice[0], ...][None, ...]
            pos = pos[window_slice[0]+1:window_slice[1]+1, ...]

            return vid, vel, pos, init_pos

        def __len__(self):
            return self.n_exp * self.windows_in_exp

    dataloader_train = torch.utils.data.DataLoader(
        WindowedPredictionDataset(
            vid[:-1],
            vel[:-1],
            rot_vel[:-1],
            pos[:-1],
            window_size=options.sequence_length
        ),
        shuffle=True, batch_size=options.batch_size
    )

    print("\tDataloader length:", len(dataloader_train))
    print()

    print("current total number of datapoints (w/ similar results)")
    print(f"{len(dataloader_train) * options.batch_size * options.sequence_length * args.epochs:,f}") # where 100 is the number of epochs
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
        
    print("total number of datapoints (w/ rat in a box)")
    print(f"{len(dataloader_train) * options.batch_size * options.sequence_length:,f}")
    print()

    dataloader_test = torch.utils.data.DataLoader(
        WindowedPredictionDataset(
            vid[-1:, :((vel.shape[1]//500)-1)*500],
            vel[-1:, :((vel.shape[1]//500)-1)*500],
            rot_vel[-1:, :((vel.shape[1]//500)-1)*500],
            pos[-1:, :((vel.shape[1]//500)-1)*500],
            window_size=options.sequence_length
        ),
        shuffle=True, batch_size=500
    )


    # Plot a few sample trajectories
    inputs, pos, pc_outputs = trajectory_generator.get_test_batch_mine(dataloader_test)
    us = place_cells.us.cpu()
    pos = pos.cpu()

    plt.figure(figsize=(5,5))
    plt.scatter(us[:,0], us[:,1], c='lightgrey', label='Place cell centers')
    for i in range(10):
        plt.plot(pos[:,i,0],pos[:,i,1], label='Simulated trajectory', c='C1')
        if i==0:
            plt.legend();

    plt.savefig(os.path.join(trainer.ckpt_dir, 'simulated_trajectory.png'))
    plt.close()

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

    trainer.train_mine(n_epochs=args.epochs, dl=dataloader_train, save=True)

    print('done training')
    torch.cuda.empty_cache()

    plt.figure(figsize=(12,3))
    plt.subplot(121)
    plt.plot(trainer.err, c='black')

    plt.title('Decoding error (m)'); plt.xlabel('train step')
    plt.subplot(122)
    plt.plot(trainer.loss, c='black');
    plt.title('Loss'); plt.xlabel('train step');

    plt.savefig(os.path.join(trainer.ckpt_dir, 'training_curves.png'))
    plt.close()

    inputs, pos, pc_outputs = trajectory_generator.get_single_test_batch_mine(dataloader_test)
    pos = pos.cpu()
    pred_pos = place_cells.get_nearest_cell_pos(model.predict(inputs)).cpu()
    us = place_cells.us.cpu()

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    for i in range(5):
        plt.plot(pos[:,i,0], pos[:,i,1], c='black', label='True position', linewidth=2)
        plt.plot(pred_pos[:,i,0], pred_pos[:,i,1], '.-',
                c='C1', label='Decoded position')
        if i==0:
            plt.legend()
    plt.scatter(us[:,0], us[:,1], s=20, alpha=0.5, c='lightgrey')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-options.box_width/2,options.box_width/2])
    plt.ylim([-options.box_height/2,options.box_height/2]);
    plt.savefig(os.path.join(trainer.ckpt_dir, 'decoded_pos.png'))
    plt.close()

    # Visualize predicted place cell outputs
    inputs, pos, pc_outputs = trajectory_generator.get_single_test_batch_mine(dataloader_test)
    preds = model.predict(inputs)
    preds = preds.reshape(-1, options.Np).detach().cpu()
    pc_outputs = model.softmax(pc_outputs).reshape(-1, options.Np).cpu()
    pc_pred = place_cells.grid_pc(preds[:100])
    pc = place_cells.grid_pc(pc_outputs[:100])

    plt.figure(figsize=(16,4))
    for i in range(8):
        plt.subplot(2,8,i+9)
        plt.imshow(pc_pred[2*i], cmap='jet')
        if i==0:
            plt.ylabel('Predicted')
        plt.axis('off')
        
    for i in range(8):
        plt.subplot(2,8,i+1)
        plt.imshow(pc[2*i], cmap='jet', interpolation='gaussian')
        if i==0:
            plt.ylabel('True')
        plt.axis('off')
        
    plt.suptitle('Place cell outputs', fontsize=16)
    plt.savefig(os.path.join(trainer.ckpt_dir, 'decoded_pc.png'))
    plt.close()

    # ckpt_path = "experiments/riab_500_box_0635_epochs_2_steps_0_seq_20_batch_5000_g_1024_p_512_rf_003463636363636363_lr_00001_weight_decay_00001/epoch_2.pth"
    # model.load_state_dict(torch.load(ckpt_path))
    # model.eval()

    # grid scores calculation
    print('grid scores calculation')    
    from visualize import compute_ratemaps
    n_avg = 100
    lo_res = 30
    _, rate_map_lores, _, _ = compute_ratemaps(
        model,
        trajectory_generator,
        options,
        res=lo_res,
        n_avg=n_avg,
        Ng=options.Ng,
        riab=True,
        dl=dataloader_test
    )

    from scores import GridScorer

    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    box_width=options.box_width
    box_height=options.box_height
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(lo_res, coord_range, masks_parameters)

    scores = []
    for idx, rm in enumerate(rate_map_lores):
        if idx % 500 == 0 : print(f"Computing scores for trajectory {idx}/{rate_map_lores.shape[0]}")
        
        scores.append(
            scorer.get_scores(rm.reshape(lo_res, lo_res))
        )
        
    score_60 = np.array([s[0] for s in scores])
    score_90 = np.array([s[1] for s in scores])
    max_60_mask = np.array([s[2] for s in scores])
    max_90_mask = np.array([s[3] for s in scores])
    max_60_ind = np.array([s[5] for s in scores])

    np.save(os.path.join(trainer.ckpt_dir, 'score_60.npy'), score_60)
    np.save(os.path.join(trainer.ckpt_dir, 'score_90.npy'), score_90)
    np.save(os.path.join(trainer.ckpt_dir, 'max_60_mask.npy'), max_60_mask)
    np.save(os.path.join(trainer.ckpt_dir, 'max_90_mask.npy'), max_90_mask)
    np.save(os.path.join(trainer.ckpt_dir, 'max_60_ind.npy'), max_60_ind)


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--behaviour', type=str, default='cluster3', help="Behaviour from which to load data")

    argparser.add_argument('--n_exp', type=int, default=3, help="Number of experiments to load") # was 100_000
    argparser.add_argument('--epochs', type=int, default=150, help="Number of epochs") # was 100_000
    argparser.add_argument('--n_steps', type=int, default=0, help="Number of training steps") # was 100_000
    argparser.add_argument('--batch_size', type=int, default=1_000, help="Number of trajectories per batch") # was 200

    argparser.add_argument('--override_pc_ref', type=float, default=None, help="Width of place cell center tuning curve (m)")
    argparser.add_argument('--original_velocity', action=argparse.BooleanOptionalAction)

    argparser.add_argument('--Np', type=int, default=512, help="Number of place cells")
    argparser.add_argument('--Ng', type=int, default=6144, help="Number of grid cells")

    argparser.add_argument('--box_width', type=float, default=0.635, help="Width of training environment")

    argparser.add_argument('--learning_rate', type=float, default=1e-4, help="Gradient descent learning rate")
    argparser.add_argument('--weight_decay', type=float, default=1e-4, help="Strength of weight decay on recurrent weights")

    args = argparser.parse_args()

    main(args)
