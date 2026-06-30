import torch
from tirex import load_model, ForecastModel

model: ForecastModel = load_model("NX-AI/TiRex")
data = torch.rand((5, 128))  # Sample Data (5 time series with length 128)
quantiles, mean = model.forecast(context=data, prediction_length=64)
print(f'Quantiles: {quantiles} - Mean: {mean}')