import torch.onnx
import netron
import torch
model=torch.load('model.pt').to("cpu")
# d=torch.randn(15,99,4).to("cpu")
d=torch.randn(140,11,4).to("cpu")
torch.onnx.export(model,d,'model.onnx',opset_version=10)
netron.start('model.onnx')