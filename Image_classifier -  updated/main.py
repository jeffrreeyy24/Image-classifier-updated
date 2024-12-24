import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur, Compose, Resize, ToTensor
from transformers import AutoModelForImageClassification, AutoProcessor
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
import os
import numpy as np
import cv2
import random
from io import BytesIO
from PIL import Image

model_name = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)


def compute_image_entropy(image_tensor):
    """
    Compute the entropy of an image as a measure of its complexity.
    """
    image_tensor = image_tensor.detach()  # Detach tensor from the computation graph
    image_np = image_tensor.numpy().transpose(1, 2, 0)  # Convert to HWC format
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist /= hist.sum()  # Normalize the histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Avoid log(0)
    return entropy


def adjust_blur(image_tensor, use_lighter_blur=True):
    """
    Adjust the Gaussian blur's sigma based on image entropy.
    We use a lighter blur for better retention of original features.
    """
    entropy = compute_image_entropy(image_tensor)
    if use_lighter_blur:
        sigma = (1.0, 1.5)  # Lighter blur for better preservation
    else:
        sigma = (1.5, 2.5)  # Default blur for higher entropy
    blur = GaussianBlur(kernel_size=(3, 3), sigma=sigma)
    return blur(image_tensor)


def adjust_epsilon(image_tensor):
    """
    Adjust the epsilon based on image entropy or complexity.
    """
    entropy = compute_image_entropy(image_tensor)
    if entropy < 5:
        epsilon = 0.1  # Small epsilon for low entropy
    else:
        epsilon = 0.5  # Larger epsilon for high entropy
    return epsilon


def jpeg_compress(image_tensor, quality=75):
    """
    Apply JPEG compression as an additional defense.
    """
    pil_image = to_pil_image(image_tensor)
    with BytesIO() as byte_io:
        pil_image.save(byte_io, format="JPEG", quality=quality)
        byte_io.seek(0)
        # Re-open the byte_io stream as a PIL image before applying ToTensor()
        compressed_image = Image.open(byte_io)
        compressed_image.load()  # Ensure the image is loaded from byte_io
    return ToTensor()(compressed_image)



def predict_image(image_tensor, sanitize=False):
    """
    Predict the class of an image tensor using the ResNet-50 model with optional sanitization.
    """
    if sanitize:
        image_tensor = sanitize_input(image_tensor)

    inputs = processor(images=to_pil_image(image_tensor), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    labels = model.config.id2label
    predicted_label = labels[predicted_class]
    return predicted_label


def generate_adversarial_example(image_tensor):
    """
    Generate an adversarial example using FGSM with dynamic epsilon.
    """
    epsilon = adjust_epsilon(image_tensor)  # Adjust epsilon based on image complexity
    inputs = processor(images=to_pil_image(image_tensor), return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    pixel_values.requires_grad = True

    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    target_label = logits.argmax(-1)
    loss = F.cross_entropy(logits, target_label)

    model.zero_grad()
    loss.backward()

    sign_data_grad = pixel_values.grad.sign()
    adversarial_tensor = pixel_values + epsilon * sign_data_grad
    adversarial_tensor = adversarial_tensor.clamp(0, 1)
    return adversarial_tensor[0]


def sanitize_input(image_tensor, preserve_details=True):
    """
    Apply input sanitization with dynamically adjusted Gaussian blur and JPEG compression.
    The 'preserve_details' flag controls the aggressiveness of the sanitization.
    """
    image_tensor = adjust_blur(image_tensor, use_lighter_blur=preserve_details)
    image_tensor = jpeg_compress(image_tensor)  # Apply JPEG compression
    return image_tensor


def evaluate_defense(image_tensor):
    """
    Evaluate the defense mechanism by predicting original, adversarial, and sanitized images.
    """
    original_prediction = predict_image(image_tensor)

    adversarial_tensor = generate_adversarial_example(image_tensor)
    adversarial_prediction = predict_image(adversarial_tensor)

    sanitized_prediction = predict_image(adversarial_tensor, sanitize=True)

    return original_prediction, adversarial_prediction, sanitized_prediction


def load_image_from_path(image_path):
    """Load an image from a path and display predictions for original, adversarial, and sanitized versions."""
    original_image = read_image(image_path)

    resize_transform = Resize((300, 300)) 
  # Create a resize transform
    original_image = resize_transform(original_image)  

    # Evaluate defense
    original_prediction, adversarial_prediction, sanitized_prediction = evaluate_defense(original_image)

    # Display the original image and its prediction
    original_img = ImageTk.PhotoImage(to_pil_image(original_image))
    original_label.config(
        image=original_img,
        text=f"Original: {original_prediction}",
        compound=tk.BOTTOM
    )
    original_label.image = original_img

    # Generate and resize the adversarial image
    adversarial_image = generate_adversarial_example(original_image)
    adversarial_image_resized = resize_transform(adversarial_image) 
    adversarial_img = ImageTk.PhotoImage(to_pil_image(adversarial_image_resized))
    adversarial_label.config(
        image=adversarial_img,
        text=f"Adversarial: {adversarial_prediction}",
        compound=tk.BOTTOM
    )
    adversarial_label.image = adversarial_img

    # Generate and resize the sanitized image
    sanitized_image = sanitize_input(adversarial_image)
    sanitized_image_resized = resize_transform(sanitized_image)  # Resize to default size
    sanitized_img = ImageTk.PhotoImage(to_pil_image(sanitized_image_resized))
    sanitized_label.config(
        image=sanitized_img,
        text=f"Sanitized: {sanitized_prediction}",
        compound=tk.BOTTOM
    )
    sanitized_label.image = sanitized_img


def create_default_image():
    """Create a default background image with a border."""
    default_image = Image.new("RGB", (300, 300), (179, 225, 187))  
    draw = ImageDraw.Draw(default_image)
    border_color = "white"
    border_width = 2
    draw.rectangle(
        [border_width, border_width, default_image.width - border_width, default_image.height - border_width],
        outline=border_color,
        width=border_width
    )
    return ImageTk.PhotoImage(default_image)


def display_images_from_folder(folder_path):
    """Display images from the folder as clickable thumbnails."""
    # Clear the current thumbnails
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

window_width = 1125
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

# Adversarial Image label
adversarial_label = tk.Label(root, text="Adversarial Image", font=("Arial", 10), fg="black", compound=tk.BOTTOM, image=default_img, bg="#e3a3ac")
adversarial_label.pack(side=tk.LEFT, padx=10, pady=10)

# Sanitized Image label
sanitized_label = tk.Label(root, text="Sanitized Image", font=("Arial", 10), fg="black", compound=tk.BOTTOM, image=default_img, bg="#89c2ad")
sanitized_label.pack(side=tk.LEFT, padx=10, pady=10)

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

root.mainloop()# new code 12-24-24
