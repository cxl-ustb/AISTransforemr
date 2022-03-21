import argparse
from dataread import data_train,data_test
import torch
import numpy as np
from transformer.Models import Transformer
import os
from torch import optim
from transformer.Optim import ScheduledOptim
from torch.nn import functional as F
import random
from matplotlib import pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'

def DrawTrajectory(tra_pred,tra_true):
    tra_pred[:,:,0]=tra_pred[:,:,0]*0.00063212+110.12347
    tra_true[:,:,0]=tra_true[:,:,0]*0.00063212+110.12347
    tra_pred[:,:,1]=tra_pred[:,:,1]*0.000989464+20.023834
    tra_true[:,:,1]=tra_true[:,:,1]*0.000989464+20.023834
    idx=random.randrange(0,tra_true.shape[0])
    plt.figure(figsize=(9,6),dpi=150)
    pred=tra_pred[idx,:,:].cpu().detach().numpy()
    true=tra_true[idx,:,:].cpu().detach().numpy()
    np.savetxt('pred_true.txt',np.vstack((pred,true)))
    print("A track includes a total of {0} detection points,and their longtitude and latitude differences are".format(pred.shape[0]))
    for i in range(pred.shape[0]):
        print("{0}:({1} degrees,{2} degrees)".format(i+1,abs(pred[i,0]-true[i,0]),abs(pred[i,1]-true[i,1])))
    print('\n')
    plt.plot(pred[:,0],pred[:,1], "r-o")
    plt.plot(true[:,0],true[:,1], "b-*")
    plt.show()

def cal_performance(tra_pred,tra_true):
    return F.mse_loss(tra_pred,tra_true)


def train(model,data, optimizer, device, opt):
    model.train()
    for epoch_i in range(opt.epoch):
        total_loss = 0
        desc = ' - (Training)  '
        optimizer.zero_grad()
        tra_pred = model(input_data=data)
        # backward and update parameters
        loss = cal_performance(tra_pred, data[:, 1:, :])
        loss.backward()
        optimizer.step_and_update_lr()
        total_loss += loss.item()
        if epoch_i % 100 == 0:
            DrawTrajectory(tra_pred, data[:, 1:, :])

    torch.save(model,'model.pt')
def test(model,data,device):
    desc = ' - (Training)  '
    print(desc)
    tra_pred = model(input_data=data)
    DrawTrajectory(tra_pred, data[:, 1:, :])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=1000)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)

    opt = parser.parse_args()
    opt.d_word_vec = opt.d_model
    device="cuda:0"
    data=torch.from_numpy(np.array(data_train)).to(device).to(torch.float32)
    transformer = Transformer(
        500,
        500,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
    ).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(
        model=transformer,
        data=data,
        optimizer=optimizer,
        device=device,
        opt=opt
    )
    data=torch.from_numpy(np.array(data_test)).to(device).to(torch.float32)
    model=torch.load('model.pt')
    test(
        model=model,
        data=data,
        device=device
    )







