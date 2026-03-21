import torch
import numpy as np


def predict(model, input_data, device):
    """
    Predict the class of a new ECG signal.

    Args:
    - model (torch.nn.Module): Trained model.
    - input_data (torch.Tensor): Input ECG data (should be preprocessed and in tensor format).
    - device (torch.device): Device to perform computation (CPU or GPU).

    Returns:
    - prediction (numpy.ndarray): Predicted classes (binary for multi-label classification).
    """
    model.eval()  # Set the model to evaluation mode
    input_data = input_data.to(device)  # Ensure data is on the correct device

    with torch.no_grad():
        outputs = model(input_data)  # Get the model's raw output (logits)
        preds = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid and threshold for multi-label classification

    return preds.cpu().numpy()  # Convert the result to numpy for further usage