"""
Trains a PyTorch image classification model using device-agnostic code.

Example usage:
  python train.py --learning_rate 0.003 --batch_size 64 --num_epochs 20
"""
import argparse

import torch
from torchvision import transforms

import data_setup, engine, model_builder, utils

def get_args():
  parser = argparse.ArgumentParser(description="Train a TinyVGG model on image folders.")
  parser.add_argument("--train_dir", type=str, default="data/pizza_steak_sushi/train",
                      help="directory with training images (one sub-folder per class)")
  parser.add_argument("--test_dir", type=str, default="data/pizza_steak_sushi/test",
                      help="directory with testing images (one sub-folder per class)")
  parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="learning rate for the Adam optimizer")
  parser.add_argument("--batch_size", type=int, default=32,
                      help="number of samples per batch")
  parser.add_argument("--num_epochs", type=int, default=5,
                      help="number of epochs to train for")
  parser.add_argument("--hidden_units", type=int, default=10,
                      help="number of hidden units in the TinyVGG layers")
  parser.add_argument("--model_name", type=str, default="05_going_modular_script_mode_tinyvgg_model.pth",
                      help="filename to save the trained model under models/")
  return parser.parse_args()

def main():
  args = get_args()
  print(f"[INFO] Training a model for {args.num_epochs} epochs with batch size {args.batch_size}, "
        f"{args.hidden_units} hidden units and a learning rate of {args.learning_rate}")
  print(f"[INFO] Training data file: {args.train_dir}")
  print(f"[INFO] Testing data file: {args.test_dir}")

  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=args.train_dir,
      test_dir=args.test_dir,
      transform=data_transform,
      batch_size=args.batch_size
  )

  # Create model with help from model_builder.py
  torch.manual_seed(42)
  model = model_builder.TinyVGG(
      input_shape=3,
      hidden_units=args.hidden_units,
      output_shape=len(class_names)
  ).to(device)

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=args.learning_rate)

  # Start training with help from engine.py
  engine.train(model=model,
               train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=args.num_epochs,
               device=device)

  # Save the model with help from utils.py
  utils.save_model(model=model,
                   target_dir="models",
                   model_name=args.model_name)

# The guard is required on Windows/macOS: DataLoader workers re-import this file
if __name__ == "__main__":
  main()
