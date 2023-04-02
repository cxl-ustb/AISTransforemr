import pickle
from torch.utils.data import Dataset
import numpy as np
import torch

f=open('./data/413719000/413719000.pkl','rb')
data=pickle.load(f)
data=data[data['速度']>=1].values
data_train_raw, data_test_raw = [],[]
i=0
while i<1400:
    data_train_raw.append(data[i:i+11])
    i+=10
i=1400

while i<2600:
    data_test_raw.append(data[i:i+11])
    i+=10
pass

class ShipTrajData(Dataset):
    def __init__(self, data): # x,y,v,theta
        self.data = torch.from_numpy(np.array(data)).to(torch.float32)
    def __getitem__(self, index):
        return self.data[index, :, :]
    def __len__(self):
        return self.data.size(0)
        # return data.size(0)
