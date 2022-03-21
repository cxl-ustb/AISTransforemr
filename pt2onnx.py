import torch.onnx
import netron
import torch
model=torch.load('model.pt').to("cuda:0")
d=torch.randn(15,99,4).to("cuda:0")
torch.onnx.export(model,d,'model.onnx',opset_version=10)
netron.start('model.onnx')