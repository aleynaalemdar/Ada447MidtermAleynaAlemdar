import gradio as gr
from fastai.vision.all import *

learn = load_learner("export.pkl")

labels = learn.dls.vocab

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(fn=classify_image,
                    inputs=gr.Image(type="pil"),
                    outputs=gr.Label(num_top_classes=3),
                    title="CIFAR-10 Image Classifier")

demo.launch()
