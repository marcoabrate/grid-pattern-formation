# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy
import os

from spatial_fields import detect_fields
from scipy import ndimage

def get_field_centre(field):
    # Replace NaN with 0 for calculation
    field_filled = np.nan_to_num(field, nan=0.0)
    
    # Calculate center of mass using the actual values as weights
    center = ndimage.center_of_mass(field_filled)
    
    return [int(c) for c in center]

def rate_maps_field_detection(rate_maps:np.array, rate_maps_1:np.array, rate_maps_2:np.array):
    """
    Detect the fields for the rate maps.

    Args:
    rate_maps (np.array): (n_units, n_samples_pos, n_samples_pos) Array of rate maps.
    rate_maps_1 (np.array): (n_units, n_samples_pos, n_samples_pos) Array of neurons' activities for the first trajectory.
    rate_maps_2 (np.array): (n_units, n_samples_pos, n_samples_pos) Array of neurons' activities for the seconds trajectory.

    Returns:
    (np.array): (n_units, 1) The number of fields for each rate map.
    (List[List[[np.array]]): The list of fields for each rate map.
    """
    rate_maps = np.nan_to_num(rate_maps, copy=True)
    rate_maps_1 = np.nan_to_num(rate_maps_1, copy=True)
    rate_maps_2 = np.nan_to_num(rate_maps_2, copy=True)

    n = rate_maps.shape[-1]

    rate_maps_min = np.moveaxis(
        np.tile(np.min(rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
    )
    rate_maps_max = np.moveaxis(
        np.tile(np.max(rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
    )
    rate_maps_norm = np.divide(
        (rate_maps - rate_maps_min),
        (rate_maps_max - rate_maps_min),
        where=(rate_maps_max - rate_maps_min)!=0,
        out=rate_maps
    )

    rm_fields = []

    params = {
        'base_threshold': 0.1,
        'threshold_step': 0.05,
        'primary_filter_kwargs': {
            'min_bins': 10, 'min_peak_value': 0.5
        },
        'secondary_filter_kwargs': {
            'min_stability': 0.25, 'max_relative_bins': 0.5,
            'stability_kwargs': {'min_included_value': 0.01, 'min_bins': 5}
        }
    }
    for rm, rm1, rm2 in zip(rate_maps_norm, rate_maps_1, rate_maps_2):
        fields = detect_fields(
            rm, (rm1, rm2), **params
        )
        rm_fields.append(fields)

    n_fields = np.array([len(f) for f in rm_fields])

    return n_fields, rm_fields

class PlaceCells(object):

    def __init__(self, options, us=None):
        self.Np = options.Np
        self.sigma = options.place_cell_rf
        self.surround_scale = options.surround_scale
        self.box_width = options.box_width
        self.box_height = options.box_height
        self.is_periodic = options.periodic
        self.DoG = options.DoG
        self.device = options.device
        self.softmax = torch.nn.Softmax(dim=-1)

        # NOT 0-1 NORMALIZED BY DEFAULT
        self.rate_maps = np.load(
            os.path.join(options.hidden_units_dir, 'place', 'rate_maps.npy')
        )

        n = self.rate_maps.shape[-1]
        rate_maps_min = np.moveaxis(
            np.tile(np.min(self.rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
        )
        rate_maps_max = np.moveaxis(
            np.tile(np.max(self.rate_maps, axis=(1,2)), (n, n, 1)), -1, 0
        )
        self.rate_maps = np.divide(
            (self.rate_maps - rate_maps_min),
            (rate_maps_max - rate_maps_min),
            where=(rate_maps_max - rate_maps_min)!=0,
            out=self.rate_maps
        )

        self.n_bins = 25
        self.bin_intervals = torch.linspace(
            -self.box_width/2, self.box_width/2, self.n_bins+1, dtype=torch.float32
        ).to(self.device)

        rm_half1 = np.load(
            os.path.join(options.hidden_units_dir, 'place', 'rm_half1.npy')
        )
        rm_half2 = np.load(
            os.path.join(options.hidden_units_dir, 'place', 'rm_half2.npy')
        )
        _, rm_fields = rate_maps_field_detection(self.rate_maps, rm_half1, rm_half2)
        fields_centres = []
        for fields in rm_fields:
            if fields:
                fields_dim = [np.isnan(f).sum() for f in fields]
                fields_centres.append(get_field_centre(fields[np.argmin(fields_dim)]))
            else: fields_centres.append([0, 0])
        fc = torch.tensor(fields_centres, dtype=torch.int32)
        bin_centres = (self.bin_intervals[:-1] + self.bin_intervals[1:]) /2

        self.us = bin_centres[fc].to(self.device)

        self.rate_maps = torch.tensor(self.rate_maps, dtype=torch.float32).to(self.device)

    def get_activation(self, pos):
        '''
        Get place cell activations for a given position.

        Args:
            pos: 2d position of shape [sequence_length, batch_size, 2].

        Returns:
            outputs: Place cell activations with shape [sequence_length, batch_size, Np].
        '''
        pos_digitized = torch.bucketize(pos, self.bin_intervals) - 1
        outputs = self.rate_maps[..., pos_digitized[..., 0], pos_digitized[..., 1]]

        outputs = torch.moveaxis(outputs, 0, -1)

        return self.softmax(outputs)

    def get_nearest_cell_pos(self, activation, k=3):
        '''
        Decode position using centers of k maximally active place cells.
        
        Args: 
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        '''
        _, idxs = torch.topk(activation, k=k)
        pred_pos = self.us[idxs].mean(-2)
        return pred_pos

    def grid_pc(self, pc_outputs, res=32):
        ''' Interpolate place cell outputs onto a grid'''
        coordsx = np.linspace(-self.box_width/2, self.box_width/2, res)
        coordsy = np.linspace(-self.box_height/2, self.box_height/2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Convert to numpy
        pc_outputs = pc_outputs.reshape(-1, self.Np)
        
        T = pc_outputs.shape[0] #T vs transpose? What is T? (dim's?)
        pc = np.zeros([T, res, res])
        for i in range(len(pc_outputs)):
            gridval = scipy.interpolate.griddata(self.us.cpu(), pc_outputs[i], grid)
            pc[i] = gridval.reshape([res, res])
        
        return pc

    def compute_covariance(self, res=30):
        '''Compute spatial covariance matrix of place cell outputs'''
        pos = np.array(np.meshgrid(np.linspace(-self.box_width/2, self.box_width/2, res),
                         np.linspace(-self.box_height/2, self.box_height/2, res))).T

        pos = torch.tensor(pos)

        # Put on GPU if available
        pos = pos.to(self.device)

        #Maybe specify dimensions here again?
        pc_outputs = self.get_activation(pos).reshape(-1,self.Np).cpu()

        C = pc_outputs@pc_outputs.T
        Csquare = C.reshape(res,res,res,res)

        Cmean = np.zeros([res,res])
        for i in range(res):
            for j in range(res):
                Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)
                
        Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)

        return Cmean
