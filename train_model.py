"""CLI entry point: `python train_model.py` trains and saves the XGBoost model."""
from model.train import train_model

if __name__ == "__main__":
    train_model()
