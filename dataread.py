import pickle

f=open('./data/413719000/413719000.pkl','rb')
data=pickle.load(f)
data=data[data['速度']>=1].values
data_train,data_test=[],[]
i=0
while i<1400:
    data_train.append(data[i:i+11])
    i+=10
i=1400
while i<2600:
    data_test.append(data[i:i+11])
    i+=10
pass