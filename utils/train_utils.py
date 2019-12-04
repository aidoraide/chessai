import time
import numpy as np

from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
device = torch.device("cuda:0")

UPDATE_FREQ = 1024
NL_UPDATE_FREQ = UPDATE_FREQ*128

N_BUCKETS = 3
bucket = lambda v: int((v+1)/2*(1-1e-6)*N_BUCKETS)

WIN = torch.FloatTensor([0, 0, 1]).to(device)
TIE = torch.FloatTensor([0, 1, 0]).to(device)
LOSS = torch.FloatTensor([1, 0, 0]).to(device)
def value2categorical(value):
    value = value.view(-1)
    valueCategorical = torch.zeros((value.size(0), 3)).to(device)
    valueCategorical[value ==  1] = WIN
    valueCategorical[value ==  0] = TIE
    valueCategorical[value == -1] = LOSS
    return valueCategorical

def fit_epoch(nnet, optimizer, scheduler, policy_criterion, value_criterion,
              POLICY_WEIGHT, VALUE_WEIGHT, train_dl, val_dl, skip_validation=False, epoch=0):
    running_policy_loss, running_value_loss, p_correct, p_total, v_correct, v_total, t0 = 0., 0., 0, 0, 0, 0, time.time()
    v_counter, v_exp = Counter(), Counter()
    nnet.train()
    for i, batch in enumerate(train_dl):
        # Transfer to GPU
        state, policy, value = batch['state'].to(device), batch['policy'].to(device), \
            batch['value'].to(device)
        value = value2categorical(value)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        policy_out, value_out = nnet(state)
        policy_loss = policy_criterion(policy_out, policy) * POLICY_WEIGHT
        # value_loss  = (value_criterion(value_out * 25, value * 25) / 25).sum() * VALUE_WEIGHT
        value_loss  = value_criterion(value_out, value) * VALUE_WEIGHT
        total_loss = policy_loss + value_loss
        total_loss.backward()
        optimizer.step()

        # print statistics
        running_policy_loss += policy_loss.item()
        running_value_loss  += value_loss.item()
        p_correct += sum(policy.flatten(1).argmax(1) == policy_out.flatten(1).argmax(1)).item()
        p_total += policy.size(0)
        # v_correct += (((value < 0) & (value_out < 0)) | ((value > 0) & (value_out > 0))).sum().item()
        # v_total += value.size(0)
        # v_counter.update([bucket(v.item()) for v in value_out])
        # v_exp.update([bucket(v.item()) for v in value])
        v_correct += sum(value.argmax(1) == value_out.argmax(1)).item()
        v_total += value.size(0)
        v_counter.update([v.item() for v in value_out.argmax(1)])
        v_exp.update([v.item() for v in value.argmax(1)])
        nl = ((i*train_dl.batch_size) % NL_UPDATE_FREQ == 0 or i == len(train_dl)) and i != 0
        if (i*train_dl.batch_size) % UPDATE_FREQ == 0:    # print every UPDATE_FREQ mini-batches
            # vcs = [f'{v_counter[b]/v_total:.3f}' for b in range(N_BUCKETS)]
            # vexps = [f'{v_exp[b]/v_total:.3f}' for b in range(N_BUCKETS)]
            vcs = [f'{v_counter[b]/v_total:.3f}' for b in range(3)]
            vexps = [f'{v_exp[b]/v_total:.3f}' for b in range(3)]
            print('[%d,%7d,%7.2f%%] (%6.2fs) acc[p: %.2f%% v: %.2f%%] loss[p: %.3e v: %.3e total: %.3e] lrs: %s   vout_dist %s vexp_dist %s' %
                  (epoch + 1, i, 100.*i/len(train_dl), time.time() - t0, p_correct*100./p_total, v_correct/v_total*100, 
                   running_policy_loss / p_total, running_value_loss / v_total,
                   (running_policy_loss + running_value_loss) / p_total, scheduler.get_lr(), ' '.join(vcs), 
                   ' '.join(vexps)),
                  end='\r\n' if nl else '\r')

        if nl:
            running_policy_loss, running_value_loss, p_correct, p_total, v_correct, v_total, t0 = 0., 0., 0, 0, 0, 0, time.time()
            v_counter, v_exp = Counter(), Counter()

        # scheduler.step()

    if skip_validation:
        return None

    # Validation
    nnet.eval()
    correct, val_correct, total, value_loss_total, policy_loss_total = 0, 0, 0, 0.0, 0.0
    # values, value_exp = [], []
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(val_dl):
            # Transfer to GPU
            state, policy, value = batch['state'].to(device), batch['policy'].to(device), \
                batch['value'].to(device)
            value = value2categorical(value)
            policy_out, value_out = nnet(state)
            policy_loss = policy_criterion(policy_out, policy) * POLICY_WEIGHT
            value_loss  = value_criterion(value_out, value).mean() * VALUE_WEIGHT

            correct += sum(policy.flatten(1).argmax(1) == policy_out.flatten(1).argmax(1)).item()
            val_correct += sum(value.argmax(1) == value_out.argmax(1)).item()
            # val_correct += (((value < 0) & (value_out < 0)) | ((value > 0) & (value_out > 0))).sum().item()
            total += policy.size()[0]
            value_loss_total += value_loss.item()
            policy_loss_total += policy_loss.item()

            # values.extend([v.argmax().item()-1 for v in value_out])
            # value_exp.extend([v.argmax().item()-1 for v in value])

    acc = 100.*correct/total
    val_acc = 100.*val_correct/total
    avg_value_loss = value_loss_total/len(val_dl)
    avg_policy_loss = policy_loss_total/len(val_dl)
    avg_loss = avg_value_loss + avg_policy_loss

    # matplotlib.rcParams['figure.figsize'] = [15, 10]

    # for i in range(4):
    #     filt = [-1, 0, 1] if i == 0 else [[0], [-1], [1]][i-1]
    #     ax = plt.subplot(2, 2, i+1)
    #     ax.hist([v for v, exp in zip(values, value_exp) if exp in filt], bins=200, color='blue', edgecolor='black')
    #     ax.set_title(f'Histogram of predicted values in {filt}', size=20)
    #     ax.set_xlabel('Predicted value (win=1, loss=-1, tie=0)', size=17)
    #     ax.set_ylabel('Frequency', size=17)

    # plt.tight_layout()
    # plt.show()
    print(f'VALIDATION RESULTS: p_acc: {acc:.2f}%, v_acc: {val_acc:2f}%, total loss: {avg_loss:.3e}, policy loss: {avg_policy_loss:.3e}, value_loss: {avg_value_loss:.3e}' + (' '*300))
    print()

    return avg_loss
