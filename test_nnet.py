from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model import load_model
from utils import data_utils
from utils.constants import device
from utils.train_utils import WriteHelper

if __name__ == '__main__':
    BATCH_SIZE = 2048
    nnet = load_model()
    _, __, test_dl = data_utils.get_split_dataloaders(
        BATCH_SIZE, data_utils.get_lichess_dataframe)

    VALUE_WEIGHT = 1
    POLICY_WEIGHT = 1
    LOSS_REDUCTION = 'sum'
    policy_criterion = nn.BCELoss(reduction=LOSS_REDUCTION)
    value_criterion = nn.MSELoss(reduction=LOSS_REDUCTION)
    # Validation
    nnet.eval()
    acc_total_loss = 0
    writer = SummaryWriter('runs/Dec09_12-05-42_DESKTOP-1USGMOH')
    write_helper = WriteHelper(writer)
    t0 = datetime.now()
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(test_dl):
            # Transfer to GPU
            state, policy, value = batch['state'].to(device), batch['policy'].to(device), \
                batch['value'].to(device)
            # value = value2categorical(value)
            policy_out, value_out = nnet(state)
            policy_loss = policy_criterion(policy_out, policy) * POLICY_WEIGHT
            value_loss = value_criterion(value_out, value) * VALUE_WEIGHT
            total_loss = policy_loss + value_loss
            acc_total_loss += total_loss.item()

            write_helper.write_to_tensorboard(f'ultimate_test', i, test_dl.batch_size, len(
                test_dl), nnet, state, value, policy, value_out, policy_out, value_loss/BATCH_SIZE, policy_loss/BATCH_SIZE, total_loss/BATCH_SIZE, optimizer=None)
            print(f"Testing: {100.*(i+1)/len(test_dl):.2f}% {str(datetime.now() - t0).split('.')[0]}",
                  end='\r\n' if i+1 == len(test_dl) else '\r')
