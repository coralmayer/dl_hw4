"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")

import torch
from torch._higher_order_ops.flex_attention import trace_flex_attention_backward
import torch.nn as nn
from torch.optim import Adam

from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric

def masked_mse(pred, target, mask):
  """
  pred: (B, 3, 2)
  target: (B, 3, 2)
  mask: (B, 3)
  """
  mask = mask.unsqueeze(-1).float()
  diff = (pred - target) * mask

  lateral_diff = diff[..., 0]
  lon_diff = diff[..., 1]
  return (2.0 * lateral_diff**2 + 1.0 * lon_diff**2).mean()

def get_loaders(transform_pipeline):
  train_loader = load_data(
      "drive_data/train", 
      transform_pipeline = transform_pipeline, 
      shuffle = True, 
      batch_size = 64,
    )
  val_loader = load_data(
    "drive_data/val", 
    transform_pipeline = transform_pipeline, 
    shuffle = False,       
    batch_size = 64,
  )
  return train_loader, val_loader


def train_planner():
  device = "cuda" if torch.cuda.is_available() else "cpu"

  for model_name in ["mlp_planner", "transformer_planner", "cnn_planner"]:
    if model_name in ["mlp_planner", "transformer_planner"]:
      transform_pipeline = "state_only"
    else:
      transform_pipeline = "default"

    if model_name == "mlp_planner":
      num_epochs = 20
    elif model_name == "transformer_planner":
      num_epochs = 60
    else:
      num_epochs = 60
    
    model = load_model(model_name).to(device)
    optimizer = Adam(model.parameters(), lr = 1e-3)

    best_lat = float("inf")

    for epoch in range(num_epochs):
      train_loader, val_loader = get_loaders(transform_pipeline)

      model.train()
      metric = PlannerMetric()

      for batch in train_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if model_name in ["mlp_planner", "transformer_planner"]:
          pred = model(track_left=batch["track_left"], track_right=batch["track_right"])
        else:
          pred = model(image=batch["image"])

        loss = masked_mse(pred, batch["waypoints"], batch["waypoints_mask"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.add(pred, batch["waypoints"], batch["waypoints_mask"])

      print(f"Epoch {epoch+1} Train:", metric.compute())

      model.eval()
      metric = PlannerMetric()

      with torch.inference_mode():
        for batch in val_loader:
          batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
          
          if model_name in ["mlp_planner", "transformer_planner"]:
            pred = model(track_left=batch["track_left"], track_right=batch["track_right"])
          else:
            pred = model(image=batch["image"])

          metric.add(pred, batch["waypoints"], batch["waypoints_mask"])

      val_metrics = metric.compute()
      print(f"Epoch {epoch+1} Val:", val_metrics)

      if val_metrics["lateral_error"] < best_lat:
        best_lat = val_metrics["lateral_error"]
        save_model(model)
        print(f"Saved BEST {model_name} (lat={best_lat:.3f})")

    print(f"Saved {model_name}\n")

if __name__ == "__main__":
    train_planner()

