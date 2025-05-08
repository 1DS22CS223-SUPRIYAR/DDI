import copy
from sklearn import datasets
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from models import Drugram
from collator import *
torch.manual_seed(2)
np.random.seed(3)
from configs import Model_config
from DDI.backend.dataset import Dataset

# Check if CUDA or MPS is available
use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()

# Set the device to GPU, MPS, or CPU based on availability
if use_cuda:
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif use_mps:
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

def test(data_set, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for _, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
            p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
            label) in enumerate(tqdm(data_set)):

        score = model(d_node.to(device), d_attn_bias.to(device), d_spatial_pos.to(device),
                      d_in_degree.to(device), d_out_degree.to(device), d_edge_input.to(device),
                      p_node.to(device), p_attn_bias.to(device), p_spatial_pos.to(device),
                      p_in_degree.to(device), p_out_degree.to(device), p_edge_input.to(device))

        label = Variable(torch.from_numpy(np.array(label-1)).long()).to(device)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(score, label)
        loss_accumulate += loss
        count += 1

        outputs = score.argmax(dim=1).detach().cpu().numpy() + 1
        label_ids = label.to('cpu').numpy() + 1

        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + outputs.flatten().tolist()

    loss = loss_accumulate / count

    accuracy = accuracy_score(y_label, y_pred)
    micro_precision = precision_score(y_label, y_pred, average='micro')
    micro_recall = recall_score(y_label, y_pred, average='micro')
    micro_f1 = f1_score(y_label, y_pred, average='micro')

    macro_precision = precision_score(y_label, y_pred, average='macro')
    macro_recall = recall_score(y_label, y_pred, average='macro')
    macro_f1 = f1_score(y_label, y_pred, average='macro')

    return accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss.item()


def main():
    config = Model_config()
    print(config)

    loss_history = []

    # model = torch.load('./save_model/best_model.pth')
    model = Drugram(**config)

    # Move the model to the correct device (GPU, MPS, or CPU)
    model = model.to(device)

    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, dim=0)

    params = {'batch_size': config['batch_size'],
              'shuffle': True,
              'num_workers': config['num_workers'],
              'drop_last': True,
              'collate_fn': collator}

    train_data = pd.read_csv('dataset/train.csv')
    val_data = pd.read_csv('dataset/val.csv')
    test_data = pd.read_csv('dataset/test.csv')

    training_set = Dataset(train_data.index.values, train_data.Label.values, train_data)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(val_data.index.values, val_data.Label.values, val_data)
    validation_generator = data.DataLoader(validation_set, **params)

    testing_set = Dataset(test_data.index.values, test_data.Label.values, test_data)
    testing_generator = data.DataLoader(testing_set, **params)

    max_auc = 0
    model_max = copy.deepcopy(model)

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'], eta_min=args.min_lr)

    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(config['epochs']):
        model.train()
        for i, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
                p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
                label) in enumerate(tqdm(training_generator)):

            score = model(d_node.to(device), d_attn_bias.to(device), d_spatial_pos.to(device),
                          d_in_degree.to(device), d_out_degree.to(device), d_edge_input.to(device),
                          p_node.to(device), p_attn_bias.to(device), p_spatial_pos.to(device),
                          p_in_degree.to(device), p_out_degree.to(device), p_edge_input.to(device))

            label = Variable(torch.from_numpy(np.array(label-1)).long()).to(device)
            loss_fct = torch.nn.CrossEntropyLoss()

            loss = loss_fct(score, label)
            loss_history.append(loss)

            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step()

            if (i % 1000 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        with torch.set_grad_enabled(False):
            accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss = test(validation_generator, model)
            print("[Validation metrics]: loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
                loss, accuracy, macro_precision, macro_recall, macro_f1))

            if accuracy > max_auc:
               # torch.save(model, 'save_model/' + str(accuracy) + '_model.pth')
                torch.save(model, 'save_model/best_model.pth')
                model_max = copy.deepcopy(model)
                max_auc = accuracy
                print("*" * 30 + " save best model " + "*" * 30)

        torch.cuda.empty_cache()

    print('\n--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss  = test(testing_generator, model_max)
            print("[Testing metrics]: loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
                loss, accuracy, macro_precision, macro_recall, macro_f1))
    except:
        print('testing failed')
    return model_max, loss_history


if __name__ == '__main__':
    main()
