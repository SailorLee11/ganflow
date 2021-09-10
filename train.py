"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function
from torch.utils.data import DataLoader,TensorDataset
from options import Options
from lib.data import load_data
from lib.model import Ganomaly
import torch
from load_data.preprocessing import main_process_unsw
##
def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    # LOAD DATA
    dataloader,testload = load_data(opt)
    ##
    #data, x_test, y_test = main_process(100)

    #feature = data.shape[1]
    #opt_add.features = feature
    # data = (data.astype(np.float32) - 127.5) / 127.5
    # X_normal = data.values.reshape(data.shape[0], 44)
    # print(X_normal.shape)
    #X_normal = data

    #X_normal = torch.FloatTensor(X_normal)
    # print(X_normal.shape)
    # X_normal = torch.unsqueeze(X_normal, -2)
    # print(X_normal.shape)
    #X_normal_data = TensorDataset(X_normal)
    #train_loader = DataLoader(dataset=X_normal_data,
    #                          batch_size = 64,
    #                          shuffle=True)


    # LOAD MODEL
    model = Ganomaly(opt, dataloader)
    #model.malware_eva(testload)
    ##
    # TRAIN MODEL
    model.train(testload)

if __name__ == '__main__':
    train()
