from . import neural_utils
import time
from datetime import datetime
import math
import numpy as np

from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
device = torch.device("cuda:0")


UPDATE_FREQ = 1024
NL_UPDATE_FREQ = UPDATE_FREQ*128


class StdMeanTracker():
    def __init__(self):
        self.N = 0
        self.std = 0
        self.mean = 0

    def add(self, batch_out):
        bN = batch_out.view(-1).size(0)
        mean = batch_out.mean().item()
        std = batch_out.std().item()

        self.std = math.sqrt(
            (self.std**2*(self.N-1) + std**2*(bN-1)) / (self.N + bN - 1))
        self.mean = (self.mean*self.N + mean*bN) / (self.N + bN)
        self.N = self.N + bN


WIN = torch.FloatTensor([0, 0, 1]).to(device)
TIE = torch.FloatTensor([0, 1, 0]).to(device)
LOSS = torch.FloatTensor([1, 0, 0]).to(device)


def value2categorical(value):
    value = value.view(-1)
    valueCategorical = torch.zeros((value.size(0), 3)).to(device)
    valueCategorical[value == 1] = WIN
    valueCategorical[value == 0] = TIE
    valueCategorical[value == -1] = LOSS
    return valueCategorical


class WriteHelper():
    def __init__(self, writer):
        self.writer = writer
        self.value_histogram_idx = 0
        self.value_out_history = []
        self.white_mask_history = []
        self.graph_added = False

    def write_to_tensorboard(self, tag, batch_idx, batch_size, n_batches, nnet, state, value, policy, value_out, policy_out, value_loss, policy_loss, total_loss, optimizer=None):
        if not self.graph_added:
            self.writer.add_graph(nnet, state)
        white_mask = neural_utils.get_where_state_white_mask(state)
        black_mask = ~white_mask
        v_non_zero = (value != 0)

        self.writer.add_pr_curve(
            f'{tag}/ValuePR/', (value[v_non_zero]+1)/2, (value_out[v_non_zero]+1)/2, batch_idx)
        whole_batch = np.arange(batch_size)
        pol_argmax = policy_out.flatten(1).argmax(1)
        policy_labels = torch.zeros(batch_size).to(device)
        policy_preds = torch.zeros(batch_size).to(device)
        policy_labels[:] = policy.view(batch_size, -1)[whole_batch,pol_argmax]
        policy_preds[:] = policy_out.view(batch_size, -1)[whole_batch,pol_argmax]
        self.writer.add_pr_curve(
            f'{tag}/PolicyPR/', policy_labels, policy_preds, batch_idx)

        p_correct = policy.flatten(1).argmax(1) == policy_out.flatten(1).argmax(1)
        v_correct = ((value == -1) & (value_out < 0)) | ((value == 1) & (value_out > 0))
        p_loss = policy_loss.item()
        v_loss = value_loss.item()
        total_loss = total_loss.item()
        lrs = [group['lr']
               for group in optimizer.param_groups] if optimizer else []
        self.writer.add_scalars(f'{tag}/ValueAccuracy/', {
            'value': v_correct.sum().item()/v_non_zero.sum().item(),
            'valueWhite': (v_correct & white_mask).sum().item()/(v_non_zero & white_mask).sum().item(),
            'valueBlack': (v_correct & black_mask).sum().item()/(v_non_zero & black_mask).sum().item(),
        }, batch_idx)
        self.writer.add_scalars(f'{tag}/PolicyAccuracy/', {
            'policy': p_correct.sum().item()/policy.size(0),
            'policyWhite': (p_correct & white_mask).sum().item()/white_mask.sum().item(),
            'policyBlack': (p_correct & black_mask).sum().item()/black_mask.sum().item(),
        }, batch_idx)
        self.writer.add_scalars(f'{tag}/Loss/', {
            'policy': p_loss,
            'value': v_loss,
            'total': total_loss,
        }, batch_idx)
        self.writer.add_scalars(f'{tag}/LossValue/', {
            'value': v_loss,
        }, batch_idx)
        if lrs:
            self.writer.add_scalars(f'{tag}/LRs/', {
                **{f'{i}': lr for i, lr in enumerate(lrs)},
            }, batch_idx)

        # histogram for every ~100k samples
        self.value_out_history.append(value_out.cpu())
        self.white_mask_history.append(white_mask)
        next_batch_value_histogram_idx = (batch_idx+1)*batch_size//(1024*128)
        if self.value_histogram_idx != next_batch_value_histogram_idx:
            histogram = torch.stack(self.value_out_history).flatten()
            historical_white_mask = torch.stack(
                self.white_mask_history).flatten()
            self.writer.add_histogram(
                'value_output', histogram, self.value_histogram_idx)
            self.writer.add_histogram(
                'value_output_when_white', histogram[historical_white_mask], self.value_histogram_idx)
            self.writer.add_histogram(
                'value_output_when_black', histogram[~historical_white_mask], self.value_histogram_idx)
            self.value_histogram_idx = next_batch_value_histogram_idx
            self.value_out_history = []
            self.white_mask_history = []


def fit_epoch(nnet, optimizer, scheduler, policy_criterion, value_criterion,
              POLICY_WEIGHT, VALUE_WEIGHT, train_dl, val_dl, writer, skip_validation=False, epoch=0):
    batch_size = train_dl.batch_size
    t0 = datetime.now()
    nnet.train()
    write_helper = WriteHelper(writer)
    # Scale losses for metrics so they are the same regardless of reduction
    vs = (lambda loss: loss/batch_size) if value_criterion.reduction == 'sum' else (lambda loss: loss)
    ps = (lambda loss: loss/batch_size) if policy_criterion.reduction == 'sum' else (lambda loss: loss)
    for i, batch in enumerate(train_dl):
        # Transfer to GPU
        state, policy, value = batch['state'].to(device), batch['policy'].to(device), \
            batch['value'].to(device)
        # value = value2categorical(value)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        policy_out, value_out = nnet(state)
        policy_loss = policy_criterion(policy_out, policy) * POLICY_WEIGHT
        # value_loss  = (value_criterion(value_out * 25, value * 25) / 25).sum() * VALUE_WEIGHT
        value_loss = value_criterion(value_out, value) * VALUE_WEIGHT
        total_loss = policy_loss + value_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        write_helper.write_to_tensorboard(f'train_{epoch+1}', i, train_dl.batch_size, len(
            train_dl), nnet, state, value, policy, value_out, policy_out, vs(value_loss), ps(policy_loss), ps(policy_loss) + vs(value_loss), optimizer)
        print(f"Train Epoch {epoch+1}: {100.*i/len(train_dl):.2f}% {str(datetime.now() - t0).split('.')[0]}",
              end='\r\n' if i+1 == len(train_dl) else '\r')

    if skip_validation:
        return None

    # Validation
    nnet.eval()
    acc_total_loss = 0
    write_helper = WriteHelper(writer)
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(val_dl):
            # Transfer to GPU
            state, policy, value = batch['state'].to(device), batch['policy'].to(device), \
                batch['value'].to(device)
            value = value2categorical(value)
            policy_out, value_out = nnet(state)
            policy_loss = policy_criterion(policy_out, policy) * POLICY_WEIGHT
            value_loss = value_criterion(value_out, value) * VALUE_WEIGHT
            total_loss = policy_loss + value_loss
            acc_total_loss += total_loss.item()

            write_helper.write_to_tensorboard(f'test_{epoch+1}', i, val_dl.batch_size, len(
                val_dl), nnet, state, value, policy, value_out, policy_out, vs(value_loss), ps(policy_loss), (ps(policy_loss) + vs(value_loss))/2, optimizer=None)
            print(f"Test Epoch {epoch+1}: {100.*i/len(val_dl):.2f}% {str(datetime.now() - t0).split('.')[0]}",
                  end='\r\n' if i+1 == len(val_dl) else '\r')

    return acc_total_loss/len(val_dl)  # avg validation loss
