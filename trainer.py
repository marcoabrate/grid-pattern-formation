# -*- coding: utf-8 -*-
import torch
import numpy as np

from visualize import save_ratemaps_mine, save_ratemaps
import os


class Trainer(object):
    def __init__(self, options, model, trajectory_generator, restore=True):
        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID)
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path))
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")
            print("Saving to: {}".format(self.ckpt_dir))

    def train_step(self, inputs, pc_outputs, pos):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        self.model.zero_grad()

        loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        loss.backward()
        self.optimizer.step()

        return loss.item(), err.item()

    def train(self, n_epochs: int = 1000, n_steps=10, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # tbar = tqdm(range(n_steps), leave=False)
        for epoch_idx in range(1, n_epochs+1):
            for step_idx in range(n_steps):
                inputs, pc_outputs, pos = next(gen)
                loss, err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)

                if step_idx%50 == 0:
                    print('Epoch: {}/{}. Step {}/{}. Loss: {}. Err: {}cm'.format(
                        epoch_idx, n_epochs, step_idx, n_steps,
                        np.round(loss, 2), np.round(100 * err, 2)))

            if (epoch_idx%10==0) and save:
                # Save a picture of rate maps
                save_ratemaps(self.model, self.trajectory_generator,
                              self.options, step=epoch_idx)
            if (epoch_idx%25==0) and save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)


    def train_mine(self, n_epochs, dl_train, dl_test, start_epoch=1, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        dl_len = len(dl_train)

        # tbar = tqdm(range(n_steps), leave=False)
        for epoch_idx in range(start_epoch, start_epoch+n_epochs):
            for step_idx, batch in enumerate(dl_train):
                image, vel, pos, init_pos = batch
                (inputs, pos, pc_outputs) = self.trajectory_generator.next_mine(image, vel, pos, init_pos)

                loss, err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)

                if step_idx%10 == 0:
                    print('Epoch: {}/{}. Batch {}/{}. Loss: {}. Err: {}cm'.format(
                        epoch_idx, n_epochs, step_idx, dl_len,
                        np.round(loss, 2), np.round(100 * err, 2)))
            print(flush=True)
            if (epoch_idx%15==0) and save:
            # if save:
                print('Saving rate maps snapshot')
                save_ratemaps_mine(
                    self.model, self.trajectory_generator,
                    self.options, step=dl_len*(epoch_idx-1)+step_idx+1,
                    dl_test=dl_test
                )
            if (epoch_idx%25==0) and save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
