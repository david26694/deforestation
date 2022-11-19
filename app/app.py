from pathlib import Path

import gradio as gr
import numpy as np
from fastai.vision.all import *


def label(file_name):
    return train_labels[file_name.replace(".jpg", "")]

config = {
    "labels": [
        "Plantation (0)",
        "Grassland (1)",
        "Smallholder Agriculture (2)",
    ],
    "size": 256,
}



learn = load_learner("model.pkl")

def classify_image(input):
    _, _, prediction = learn.predict(input)
    outputs = {label: float(prediction[i]) for i, label in enumerate(config["labels"])}
    # Get argmax
    argmax_label = config["labels"][np.argmax(prediction)]
    return argmax_label, round(outputs[argmax_label], 3) * 100


gr.Interface(
    fn=classify_image, 
    inputs=gr.inputs.Image(shape=(config["size"], config["size"])),
    outputs=[
        gr.outputs.Textbox(label="Output of the model"),
        gr.outputs.Textbox(label="Probability (0 - 100)")
    ],
    examples=[str(x) for x in Path("./").glob("*.png")],
    flagging_options=["Correct label", "Incorrect label"],
    allow_flagging="manual",
).launch()
