"""
Predicts the class of a target image with a trained TinyVGG model.

Example usage:
  python predict.py --image data/pizza_steak_sushi/test/sushi/175783.jpg
"""
import argparse

import torch
import torchvision
from torchvision import transforms

import model_builder

def get_args():
  parser = argparse.ArgumentParser(description="Predict on a target image with a trained TinyVGG model.")
  parser.add_argument("--image", type=str, required=True,
                      help="filepath of the target image to predict on")
  parser.add_argument("--model_path", type=str, default="models/05_going_modular_script_mode_tinyvgg_model.pth",
                      help="filepath of the trained model state_dict to load")
  parser.add_argument("--class_names", type=str, nargs="+", default=["pizza", "steak", "sushi"],
                      help="class names in the same order as the model was trained on")
  return parser.parse_args()

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
  """Loads a TinyVGG state_dict, reading the model sizes from the saved weights."""
  state_dict = torch.load(model_path, map_location=device)
  # First conv weight has shape [hidden_units, input_channels, 3, 3]
  hidden_units = state_dict["conv_block_1.0.weight"].shape[0]
  # Last linear weight has shape [output_shape, hidden_units*13*13]
  output_shape = state_dict["classifier.1.weight"].shape[0]
  model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=hidden_units,
                                output_shape=output_shape).to(device)
  model.load_state_dict(state_dict)
  print(f"[INFO] Loaded model from {model_path} ({hidden_units} hidden units)")
  return model

def main():
  args = get_args()
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model = load_model(args.model_path, device)

  # Load the image, turn it into float values between 0 and 1 and resize it like the training data
  image = torchvision.io.read_image(args.image).type(torch.float32) / 255.
  transform = transforms.Resize(size=(64, 64))
  image = transform(image)

  # Predict on the image (add a batch dimension first)
  model.eval()
  with torch.inference_mode():
    pred_logits = model(image.unsqueeze(dim=0).to(device))
  pred_prob = torch.softmax(pred_logits, dim=1)
  pred_label = torch.argmax(pred_prob, dim=1).item()

  print(f"[INFO] Predicting on {args.image}")
  print(f"[INFO] Pred class: {args.class_names[pred_label]}, Pred prob: {pred_prob.max():.3f}")

if __name__ == "__main__":
  main()
