import numpy as np
import torch

from matplotlib import pyplot as plt
import argparse

# If GPUs available, select which to train on
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from utils import generate_run_ID
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer


def main(args):
    # Training options and hyperparameters
    class Options:
        pass
    options = Options()

    original_pc_ref = 0.12
    original_box_width = 2.2

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

    options.run_ID = generate_run_ID(options)

    print("original total number of datapoints")
    print(f"{100_000 * 200 * options.sequence_length * 1_000:,f}") # where 1_000 is the number of epochs
    print("current total number of datapoints")
    print(f"{options.n_steps * options.batch_size * options.sequence_length * args.epochs:,f}")
    print()

    # If you've trained with these params before, will restore trained model
    place_cells = PlaceCells(options)
    model = RNN(options, place_cells).cuda()
    trajectory_generator = TrajectoryGenerator(options, place_cells)
    trainer = Trainer(options, model, trajectory_generator)

    # # Load example weights stored on github
    # weight_dir = options.save_dir + '/example_trained_weights.npy'
    # load_trained_weights(model, trainer, weight_dir)


    # # Task statistics

    # Plot a few sample trajectories
    inputs, pos, pc_outputs = trajectory_generator.get_test_batch()
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


    trainer.train(n_epochs=args.epochs, n_steps=options.n_steps, save=True)
    print('training done')    

    plt.figure(figsize=(12,3))
    plt.subplot(121)
    plt.plot(trainer.err, c='black')

    plt.title('Decoding error (m)'); plt.xlabel('train step')
    plt.subplot(122)
    plt.plot(trainer.loss, c='black');
    plt.title('Loss'); plt.xlabel('train step');

    plt.savefig(os.path.join(trainer.ckpt_dir, 'training_curves.png'))
    plt.close()

    inputs, pos, pc_outputs = trajectory_generator.get_test_batch()
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
    plt.ylim([-options.box_height/2,options.box_height/2])
    plt.savefig(os.path.join(trainer.ckpt_dir, 'decoded_pos.png'))
    plt.close()


    # Visualize predicted place cell outputs
    inputs, pos, pc_outputs = trajectory_generator.get_test_batch()
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
        Ng=options.Ng
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
    argparser.add_argument('--epochs', type=int, default=100, help="Number of epochs") # was 100_000
    argparser.add_argument('--n_steps', type=int, default=100, help="Number of training steps") # was 100_000
    argparser.add_argument('--batch_size', type=int, default=5_000, help="Number of trajectories per batch") # was 200

    argparser.add_argument('--override_pc_ref', type=float, default=None, help="Width of place cell center tuning curve (m)")
    argparser.add_argument('--original_velocity', action=argparse.BooleanOptionalAction)

    argparser.add_argument('--Np', type=int, default=512, help="Number of place cells")
    argparser.add_argument('--Ng', type=int, default=4096, help="Number of grid cells")

    argparser.add_argument('--box_width', type=float, default=2.2, help="Width of training environment")

    argparser.add_argument('--learning_rate', type=float, default=1e-4, help="Gradient descent learning rate")
    argparser.add_argument('--weight_decay', type=float, default=1e-4, help="Strength of weight decay on recurrent weights")
    args = argparser.parse_args()

    main(args)