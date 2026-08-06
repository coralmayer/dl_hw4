from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.functional import pad

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 4
        output_dim = n_waypoints * 2

        self.net = nn.Sequential(
          nn.Linear(input_dim, 128),
          nn.ReLU(inplace=True), 
          nn.Linear(128, 128),
          nn.ReLU(inplace=True), 
          nn.Linear(128, output_dim),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        
        x = torch.cat([track_left, track_right], dim=1)

        x = x.reshape(b, -1)
        out = self.net(x)
        return out.view(-1, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.d_model = d_model

        self.input_proj = nn.Sequential(
          nn.Linear(4, d_model),
          nn.ReLU(inplace=True),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, n_track, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
          d_model=d_model,
          nhead=4, 
          dim_feedforward=128,
          batch_first=True,
          activation="relu"
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.head = nn.Sequential(
          nn.Linear(d_model, 64), 
          nn.ReLU(inplace=True),
          nn.Linear(64, 2),
        )

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """

        x = torch.cat([track_left, track_right], dim=1)

        x = self.input_proj(x)

        x = x + self.pos_embed

        x = self.encoder(x)

        out = self.head(x)

        return out[:, :self.n_waypoints, :]


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
          nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
          nn.ReLU(),
          nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
          nn.ReLU(),
        )

        self.flatten_dim = 128 * 6 * 8

        self.head = nn.Sequential(
          nn.Linear(self.flatten_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 128), 
          nn.ReLU(),
          nn.Linear(128, n_waypoints * 2),
        )

        self.n_waypoints = n_waypoints

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = self.encoder(image)
        x = x.view(x.size(0), -1)
        out = self.head(x)
        return out.view(-1, self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
