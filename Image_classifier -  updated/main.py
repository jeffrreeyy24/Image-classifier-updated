import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import GaussianBlur, Compose, Resize, ToTensor
from transformers import AutoModelForImageClassification, AutoProcessor
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
import os
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

model_name = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

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


def adversarial_training_step(model, inputs, labels, epsilon=0.1):
    """
    Perform one step of adversarial training.
    """
    pixel_values = inputs["pixel_values"]
    pixel_values.requires_grad = True  # Allow gradients for FGSM

    # Forward pass to compute logits
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits

    # Compute the loss and gradients
    loss = criterion(logits, labels)
    model.zero_grad()
    loss.backward()

    # Generate adversarial examples using FGSM
    sign_data_grad = pixel_values.grad.sign()
    adversarial_tensor = pixel_values + epsilon * sign_data_grad
    adversarial_tensor = adversarial_tensor.clamp(0, 1)  # Clip the adversarial image to valid range

    # Perform a second forward pass on adversarial example
    outputs_adv = model(pixel_values=adversarial_tensor)
    logits_adv = outputs_adv.logits
    loss_adv = criterion(logits_adv, labels)

    # Combine the losses (clean and adversarial)
    total_loss = (loss + loss_adv) / 2

    # Backpropagate and update weights
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def train_model_with_adversarial_examples(model, dataloader, epochs=5, epsilon=0.1):
    """
    Train the model using adversarial training.
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs = batch['input']
            labels = batch['label']

            # Perform adversarial training step
            loss = adversarial_training_step(model, inputs, labels, epsilon)
            running_loss += loss

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")


def evaluate_defense(image_tensor):
    """
    Evaluate the defense mechanism by predicting original, adversarial, and sanitized images.
    Return predictions and their probabilities.
    """
    inputs = processor(images=to_pil_image(image_tensor), return_tensors="pt")
    
    # Original prediction
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    original_prob, original_class = torch.max(probs, dim=-1)
    original_label = model.config.id2label[original_class.item()]

    # Adversarial prediction
    adversarial_tensor = generate_adversarial_example(image_tensor)
    adversarial_inputs = processor(images=to_pil_image(adversarial_tensor), return_tensors="pt")
    with torch.no_grad():
        outputs_adv = model(**adversarial_inputs)
    logits_adv = outputs_adv.logits
    probs_adv = torch.softmax(logits_adv, dim=-1)
    adv_prob, adv_class = torch.max(probs_adv, dim=-1)
    adv_label = model.config.id2label[adv_class.item()]

    # Sanitized prediction
    sanitized_tensor = sanitize_input(adversarial_tensor)
    sanitized_inputs = processor(images=to_pil_image(sanitized_tensor), return_tensors="pt")
    with torch.no_grad():
        outputs_sanitized = model(**sanitized_inputs)
    logits_sanitized = outputs_sanitized.logits
    probs_sanitized = torch.softmax(logits_sanitized, dim=-1)
    sanitized_prob, sanitized_class = torch.max(probs_sanitized, dim=-1)
    sanitized_label = model.config.id2label[sanitized_class.item()]

    return {
        "original": {"label": original_label, "probability": original_prob.item()},
        "adversarial": {"label": adv_label, "probability": adv_prob.item()},
        "sanitized": {"label": sanitized_label, "probability": sanitized_prob.item()},
    }



def update_graph_frame(predictions):
    """
    Update the graph frame with a bar chart displaying prediction probabilities and percentage numbers.
    """
    # Clear the graph frame
    for widget in graph_frame.winfo_children():
        widget.destroy()

    # Prepare data for the bar chart
    labels = ["Original", "Adversarial", "Sanitized"]
    probabilities = [
        predictions["original"]["probability"] * 100,
        predictions["adversarial"]["probability"] * 100,
        predictions["sanitized"]["probability"] * 100,
    ]

    # Create the Matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, probabilities, color=["blue", "red", "green"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy Percentage")
    ax.set_title("Prediction Accuracy")

    # Display percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',  # Display percentage with 1 decimal place
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # Position the label at the top of the bar
                    xytext=(0, 5),  # Offset the label slightly above the bar
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color="black")

    # Embed the Matplotlib figure in the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)




def load_image_from_path(image_path):
    """Load an image from a path, evaluate predictions, and update the UI."""
    original_image = read_image(image_path)
    resize_transform = Resize((300, 300))
    original_image = resize_transform(original_image)

    # Evaluate defense and update labels
    predictions = evaluate_defense(original_image)

    # Update the graph frame with the new predictions
    update_graph_frame(predictions)

    # Resize and update the images as 300x300
    original_img = ImageTk.PhotoImage(to_pil_image(original_image))
    original_label.config(
        image=original_img,
        text=f"Original: {predictions['original']['label']}",
        compound=tk.BOTTOM
    )
    original_label.image = original_img

    adversarial_image = generate_adversarial_example(original_image)
    adversarial_image_resized = resize_transform(adversarial_image)  # Resize adversarial image to 300x300
    adversarial_img = ImageTk.PhotoImage(to_pil_image(adversarial_image_resized))
    adversarial_label.config(
        image=adversarial_img,
        text=f"Adversarial: {predictions['adversarial']['label']}",
        compound=tk.BOTTOM
    )
    adversarial_label.image = adversarial_img

    sanitized_image = sanitize_input(adversarial_image)
    sanitized_image_resized = resize_transform(sanitized_image)  # Resize sanitized image to 300x300
    sanitized_img = ImageTk.PhotoImage(to_pil_image(sanitized_image_resized))
    sanitized_label.config(
        image=sanitized_img,
        text=f"Sanitized: {predictions['sanitized']['label']}",
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

#dataset
def display_images_from_folder(folder_path):
    """Display images from the folder as clickable thumbnails."""
    # Clear the current thumbnails
    for widget in dataset_frame.winfo_children():
        widget.destroy()

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            img = img.resize((100, 100))
            img_thumb = ImageTk.PhotoImage(img)

            thumbnail_frame = tk.Frame(dataset_frame, bg="#c2f6fa")
            thumbnail_frame.pack(side=tk.TOP, padx=20, pady=5, anchor="w")  # Align left for a vertical stack

            # Button for the image thumbnail
            button = tk.Button(
                thumbnail_frame,
                image=img_thumb,
                command=lambda path=image_path: load_image_from_path(path),
                bg="#c2f6fa",
                borderwidth=0
            )
            button.image = img_thumb
            button.grid(row=0, column=0, padx=10)

            # Label for the filename, placed to the right of the button
            label = tk.Label(
                thumbnail_frame,
                text=filename,
                wraplength=100,
                anchor="w",  # Left-align the text
                bg="#c2f6fa"
            )
            label.grid(row=0, column=1, padx=10)



root = tk.Tk()
root.title("Image Classification")
root.configure(bg="#e1edf2")
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

# Create default background images for all labels
default_img = create_default_image()

# Original Image label
original_label = tk.Label(root, text="Original Image", font=("Arial", 10), fg="black", compound=tk.BOTTOM, image=default_img, bg="#60b1ae")
original_label.pack(side=tk.LEFT, anchor="n", padx=10, pady=10)

# Adversarial Image label
adversarial_label = tk.Label(root, text="Adversarial Image", font=("Arial", 10), fg="black", compound=tk.BOTTOM, image=default_img, bg="#e3a3ac")
adversarial_label.pack(side=tk.LEFT, anchor="n", padx=10, pady=10)

# Sanitized Image label
sanitized_label = tk.Label(root, text="Sanitized Image", font=("Arial", 10), fg="black", compound=tk.BOTTOM, image=default_img, bg="#89c2ad")
sanitized_label.pack(side=tk.LEFT, anchor="n", padx=10, pady=10)


#graph frame
# Create a frame for the graph
graph_frame = tk.Frame(root, bg="#c5de6f", relief="groove", bd=2)
graph_frame.place(relx=0.1, rely=0.5, relwidth=0.5, relheight=0.4)

# Scrollable frame for dataset
container = tk.Frame(root, bg="#c2f6fa")
container.pack(side=tk.RIGHT, padx=20, pady=0, fill=tk.BOTH, expand=True)

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

root.mainloop()#12/27/24
