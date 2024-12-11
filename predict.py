"""import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
torch.manual_seed(2)
np.random.seed(3)
from argparse import ArgumentParser
from dataset import Dataset
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collator import collator

use_cuda = torch.cuda.is_available()
device = torch.device("cpu")

parser = ArgumentParser(description='Drugram Prediction.')
parser.add_argument('-b', '--batch-size', default=16, type=int,metavar='N')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
            p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
            label) in enumerate(tqdm(data_generator)):

        score = model(d_node.cuda(), d_attn_bias.cuda(), d_spatial_pos.cuda(),
                      d_in_degree.cuda(), d_out_degree.cuda(), d_edge_input.cuda(),p_node.cuda(),
                      p_attn_bias.cuda(), p_spatial_pos.cuda(), p_in_degree.cuda(),
                      p_out_degree.cuda(), p_edge_input.cuda())
       
        label = Variable(torch.from_numpy(np.array(label-1)).long()).cuda()
        
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

    print("[Validation metrics]: loss:{:.4f} mean_accuracy:{:.4f} micro_precision:{:.4f} micro_recall:{:.4f} micro_f1:{:.4f} macro_precision:{:.4f} macro_recall:{:.4f} macro_f1:{:.4f}".format(
        loss, accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1))


def main():
    args = parser.parse_args()

    model = torch.load('save_model/best_model.pth', map_location=torch.device("cpu"))

    model = model.to(device)


    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': args.workers,
              'drop_last': False,
              'collate_fn': collator}

    df_test = pd.read_csv('dataset/test.csv')


    testing_set = Dataset(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    print('--- Go for Predicting ---')
    with torch.set_grad_enabled(False):
        test(testing_generator, model)

    torch.cuda.empty_cache()

#main()
#print("Done!")

if __name__ == "__main__":
    main()
    print("Done!")
    """

"""import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
torch.manual_seed(2)
np.random.seed(3)
from argparse import ArgumentParser
from dataset import Dataset
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collator import collator

# Dynamically set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = ArgumentParser(description='Drugram Prediction.')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')

def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
            p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
            label) in enumerate(tqdm(data_generator)):

        # Send inputs to device
        score = model(
            d_node.to(device), d_attn_bias.to(device), d_spatial_pos.to(device),
            d_in_degree.to(device), d_out_degree.to(device), d_edge_input.to(device),
            p_node.to(device), p_attn_bias.to(device), p_spatial_pos.to(device),
            p_in_degree.to(device), p_out_degree.to(device), p_edge_input.to(device)
        )
       
        label = Variable(torch.from_numpy(np.array(label - 1)).long()).to(device)
        
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

    print("[Validation metrics]: loss:{:.4f} mean_accuracy:{:.4f} micro_precision:{:.4f} micro_recall:{:.4f} micro_f1:{:.4f} macro_precision:{:.4f} macro_recall:{:.4f} macro_f1:{:.4f}".format(
        loss, accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1))

def main():
    args = parser.parse_args()

    # Load model and send to device
    model = torch.load('save_model/best_model.pth', map_location=device)
    model = model.to(device)

    params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': args.workers,
        'drop_last': False,
        'collate_fn': collator
    }

    df_test = pd.read_csv('dataset/test.csv')

    testing_set = Dataset(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    print('--- Go for Predicting ---')
    with torch.set_grad_enabled(False):
        test(testing_generator, model)

    # Clear CUDA cache if using GPU
    if use_cuda:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    print("Done!")"""


import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
torch.manual_seed(2)
np.random.seed(3)
from argparse import ArgumentParser
from dataset import Dataset
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collator import collator

# Check if MPS (Apple GPU) is available
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS (Metal Performance Shaders) backend is not available. Please ensure you are running on an Apple Silicon device with PyTorch configured for MPS.")

device = torch.device("mps")

parser = ArgumentParser(description='Drugram Prediction.')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')

def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
            p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
            label) in enumerate(tqdm(data_generator)):

        # Send inputs to MPS device (Apple GPU)
        score = model(
            d_node.to(device), d_attn_bias.to(device), d_spatial_pos.to(device),
            d_in_degree.to(device), d_out_degree.to(device), d_edge_input.to(device),
            p_node.to(device), p_attn_bias.to(device), p_spatial_pos.to(device),
            p_in_degree.to(device), p_out_degree.to(device), p_edge_input.to(device)
        )
       
        label = Variable(torch.from_numpy(np.array(label - 1)).long()).to(device)
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(score, label)
        loss_accumulate += loss
        count += 1

        outputs = score.argmax(dim=1).detach().cpu().numpy() + 1
        label_ids = label.to('cpu').numpy() + 1

        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + outputs.flatten().tolist()
    
    #Saving Output
    df_results = pd.DataFrame({
        'Label': y_label,
        'Prediction': y_pred
    }) 
    output_file = "predictions.csv"
    df_results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
     
    #Metrics Calculation
    loss = loss_accumulate / count

    accuracy = accuracy_score(y_label, y_pred)
    micro_precision = precision_score(y_label, y_pred, average='micro')
    micro_recall = recall_score(y_label, y_pred, average='micro')
    micro_f1 = f1_score(y_label, y_pred, average='micro')

    macro_precision = precision_score(y_label, y_pred, average='macro')
    macro_recall = recall_score(y_label, y_pred, average='macro')
    macro_f1 = f1_score(y_label, y_pred, average='macro')

    print("[Validation metrics]: loss:{:.4f} mean_accuracy:{:.4f} micro_precision:{:.4f} micro_recall:{:.4f} micro_f1:{:.4f} macro_precision:{:.4f} macro_recall:{:.4f} macro_f1:{:.4f}".format(
        loss, accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1))

def main():
    args = parser.parse_args()

    # Load model and send it to MPS device
    model = torch.load('save_model/best_model.pth', map_location=torch.device("mps"))
    model = model.to(device)

    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': args.workers,
              'drop_last': False,
              'collate_fn': collator}

    df_test = pd.read_csv('dataset/test.csv')

    testing_set = Dataset(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    print('--- Go for Predicting ---')
    with torch.set_grad_enabled(False):
        test(testing_generator, model)

    torch.mps.empty_cache()

if __name__ == "__main__":
    main()
    print("Done!")


