import torch
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import AutoModelForImageClassification, AutoProcessor
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageTk
import tkinter as tk
import os

model_name = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

def predict_image(image_tensor):
    """
    Predict the class of an image tensor using the ResNet-50 model.
    """
    inputs = processor(images=to_pil_image(image_tensor), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    labels = model.config.id2label
    predicted_label = labels[predicted_class]
    return predicted_label

def load_image_from_path(image_path):
    """Load an image from a path and display predictions."""
    original_image = read_image(image_path)

    # Apply resize transformation
    resize_transform = Resize((300, 300))
    original_image = resize_transform(original_image)

    # Predict the image
    original_prediction = predict_image(original_image)

    # Display the original image and its prediction
    original_img = ImageTk.PhotoImage(to_pil_image(original_image))
    original_label.config(
        image=original_img,
        text=f"Original: {original_prediction}",
        compound=tk.BOTTOM
    )
    original_label.image = original_img

def create_default_image():
    """Create a default background image with a border."""
    default_image = Image.new("RGB", (300, 300), (179, 225, 187))
    return ImageTk.PhotoImage(default_image)

def display_images_from_folder(folder_path):
    """Display images from the folder as clickable thumbnails."""
    for widget in dataset_frame.winfo_children():
        widget.destroy()

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            img = img.resize((80, 80))
            img_thumb = ImageTk.PhotoImage(img)

            thumbnail_frame = tk.Frame(dataset_frame, bg="#c2f6fa")
            thumbnail_frame.pack(side=tk.TOP, padx=20, pady=5)

            button = tk.Button(thumbnail_frame, image=img_thumb, command=lambda path=image_path: load_image_from_path(path), bg="#c2f6fa", borderwidth=0)
            button.image = img_thumb
            button.pack()

            label = tk.Label(thumbnail_frame, text=filename, wraplength=100, anchor="center", bg="#c2f6fa")
            label.pack()

root = tk.Tk()
root.title("Image Classification")
root.configure(bg="#e1edf2")

window_width = 450
window_height = 500
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width / 2) - (window_width / 2))
y_cordinate = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

# Create default background images for all labels
default_img = create_default_image()

# Original Image label
original_label = tk.Label(root, text="Original Image", font=("Arial", 10), fg="black", compound=tk.BOTTOM, image=default_img, bg="#60b1ae")
original_label.pack(side=tk.LEFT, padx=10, pady=10)

# Scrollable frame for dataset
container = tk.Frame(root, bg="#c2f6fa")
container.pack(side=tk.RIGHT, padx=0, pady=0, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(container, bg="#c2f6fa")
scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview, bg="#c2f6fa")
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

dataset_frame = tk.Frame(canvas, bg="#c2f6fa")
canvas.create_window((0, 0), window=dataset_frame, anchor="nw")

def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

dataset_frame.bind("<Configure>", on_frame_configure)

images_folder_path = "images"
if os.path.exists(images_folder_path):
    display_images_from_folder(images_folder_path)
else:
    print(f"The folder '{images_folder_path}' does not exist.")

# Bind mouse wheel scrolling
def on_mouse_wheel(event):
    canvas.yview_scroll(-1 * int(event.delta / 120), "units")

canvas.bind_all("<MouseWheel>", on_mouse_wheel)

root.mainloop()
