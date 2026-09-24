"""Gradio demo app for the Oxford-IIIT Pet breed classifier.

Run locally with: python app.py
Deploy to Hugging Face Spaces by uploading this folder (app.py, model.py,
09_effnetb0_oxford_pets.pth, class_names.txt, requirements.txt, examples/) as a
"Gradio" Space.
"""
import os
from timeit import default_timer as timer

import gradio as gr
import torch

from model import create_effnetb0

# Setup class names
with open("class_names.txt", "r") as f:
  class_names = [line.strip() for line in f.readlines()]

# Create model and load saved weights
model, transforms = create_effnetb0(num_classes=len(class_names))
model.load_state_dict(torch.load(f="09_effnetb0_oxford_pets.pth",
                                 map_location=torch.device("cpu")))


def predict(img):
  start_time = timer()

  img = transforms(img).unsqueeze(0)

  model.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(model(img), dim=1)

  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
  pred_time = round(timer() - start_time, 4)
  return pred_labels_and_probs, pred_time


title = "FoodVision Pets 🐶🐱"
description = "An EfficientNetB0 feature extractor that classifies photos into one of 37 cat/dog breeds (Oxford-IIIT Pet dataset)."
article = "Created as part of exercise 7, notebook 09 (PyTorch Model Deployment), Learn PyTorch for Deep Learning course."

example_list = [["examples/" + example] for example in os.listdir("examples")] if os.path.exists("examples") else []

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"),
                            gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

if __name__ == "__main__":
  demo.launch()
