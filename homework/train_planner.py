"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data

def masked_mse(pred, target, mask):
  """
  pred: (B, 3, 2)
  target: (B, 3, 2)
  mask: (B, 3)
  """
  mask = mask.unsqueeze(-1).float()
  diff = (pred - target) * mask
  return (diff ** 2).mean()

def train(model_name, batch_size, lr, epochs, device):
  train_loader = load_data(
    "drive_data/train", 
    transform_pipeline = "default", 
    return_dataloader = True, 
    batch_size = batch_size, 
    shuffle = True, 
    num_workers = 2,
  )

  model = load_model(model_name)
  model = model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr = lr)

  for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
      batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

      if model_name in ["mlp_planner", "transformer_planner"]:
        pred = model(
          track_left = batch["track_left"]
          track_right = batch["track_right"],
        )
      else:
        pred = model(image = batch["image"])

      loss = masked_mse(pred, batch["waypoints"], batch["waypoints_mask"])

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} Loss: {avg_loss:.4f}")

  path = save_model(model)
  print(f"Saved model to {path}")





