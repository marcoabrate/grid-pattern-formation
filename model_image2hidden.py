# -*- coding: utf-8 -*-
import torch

class RNNCell(torch.nn.Module):
    def __init__(self, Ng):
        super(RNNCell, self).__init__()

        self.vel2hidden = torch.nn.Linear(2, Ng, bias=False)
        self.hidden2hidden = torch.nn.Linear(Ng, Ng, bias=False)

        self.relu = torch.nn.ReLU()

    def forward(self, vel, hidden):
        vel_enc = self.vel2hidden(vel)
        hidden_enc = self.hidden2hidden(hidden)
        return self.relu(vel_enc + hidden_enc)

class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.image_loss_weight = options.image_loss_weight
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells
        self.device = options.device

        # Input weights
        self.encoder_pc = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.encoder_image = torch.nn.Linear(512, self.Ng, bias=False)
        
        self.rnn_cell = RNNCell(self.Ng)

        # Linear read-out weights
        self.decoder_pc = torch.nn.Linear(self.Ng, self.Np, bias=False)
        self.decoder_image = torch.nn.Linear(self.Ng, 512, bias=False)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()

        self.loss_fn_image = torch.nn.L1Loss(reduction='none')

    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: image, vel: [sequence_length, batch_size, 2]. init_actv: [batch_size, Np]

        Returns: 
            g: Batch of grid cell activations with shape [sequence_length, batch_size, Ng].
        '''
        image, vel, init_actv = inputs
        init_state = (
            self.encoder_pc(init_actv) +
            self.encoder_image(image[0, ...])
        )[None]

        output = torch.zeros(vel.shape[0], vel.shape[1], self.Ng).to(self.device)

        sequence_length = vel.shape[0]

        # loop over time
        h_out = init_state
        for t in range(sequence_length):
            vel_t = vel[t,...]
            h_out = self.rnn_cell(vel_t, h_out)
            output[t,...] = h_out

        return output
    

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: image, vel: [sequence_length, batch_size, 2]. init_actv: [batch_size, Np]

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [sequence_length, batch_size, Np].
        '''
        g = self.g(inputs)

        place_preds = self.decoder_pc(g)
        image_preds = self.sigmoid(
            self.decoder_image(g)
        )
        
        return place_preds, image_preds


    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        image = inputs[0]
        y = pc_outputs

        place_preds, image_preds = self.predict(inputs)
        yhat = self.softmax(place_preds)

        # image prediction loss
        loss = self.image_loss_weight * self.loss_fn_image(image_preds, image[1:, ...]).sum(-1).mean()
        
        # place cells prediction loss
        loss -= (y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.rnn_cell.hidden2hidden.weight**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(place_preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err