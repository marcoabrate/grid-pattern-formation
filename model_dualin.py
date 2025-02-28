# -*- coding: utf-8 -*-
import torch

class RNNCell(torch.nn.Module):
    def __init__(self, Ng):
        super(RNNCell, self).__init__()

        self.image2hidden = torch.nn.Linear(512, Ng, bias=False)
        self.vel2hidden = torch.nn.Linear(2, Ng, bias=False)
        self.hidden2hidden = torch.nn.Linear(Ng, Ng, bias=False)

        self.relu = torch.nn.ReLU()

    def forward(self, image, vel, hidden):
        image_enc = self.image2hidden(image)
        vel_enc = self.vel2hidden(vel)
        hidden_enc = self.hidden2hidden(hidden)
        return self.relu(image_enc + vel_enc + hidden_enc)

class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells
        self.device = options.device

        # Input weights
        self.encoder_pc = torch.nn.Linear(self.Np, self.Ng, bias=False)
        
        self.rnn_cell = RNNCell(self.Ng)

        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        
        self.softmax = torch.nn.Softmax(dim=-1)

    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: image, vel: [sequence_length, batch_size, 2]. init_actv: [batch_size, Np]

        Returns: 
            g: Batch of grid cell activations with shape [sequence_length, batch_size, Ng].
        '''
        image, vel, init_actv = inputs
        init_state = self.encoder_pc(init_actv)[None]

        output = torch.zeros(image.shape[0], image.shape[1], self.n_hidden).to(self.device)

        sequence_length = image.shape[0]

        # loop over time
        h_out = init_state
        for t in range(sequence_length):
            image_t = image[t,...]
            vel_t = vel[t,...]
            h_out = self.rnn_cell(image_t, vel_t, h_out)
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
        place_preds = self.decoder(self.g(inputs))
        
        return place_preds


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
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(self.predict(inputs))
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err