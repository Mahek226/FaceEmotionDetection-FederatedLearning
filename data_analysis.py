import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision import transforms
from PIL import Image
from task import download_kaggle_dataset, SafeImageFolder  # ✅ Safe loader


# -----------------------------
# Main Data Analysis
# -----------------------------
if __name__ == "__main__":
    # 1. Load dataset safely
    dataset_path = download_kaggle_dataset()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = SafeImageFolder(root=dataset_path, transform=transform)

    class_names = dataset.classes
    labels = [label for _, label in dataset if label != -1]  # ✅ skip dummy images
    labels = np.array(labels)

    # -----------------------------
    # 📊 Class Distribution
    # -----------------------------
    class_counts = pd.Series(labels).value_counts().sort_index()
    class_counts.index = [class_names[i] for i in class_counts.index]

    # Bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.title("Class Distribution (Bar Plot)")
    plt.show()

    # Line plot
    plt.figure(figsize=(8, 5))
    plt.plot(class_counts.index, class_counts.values, marker="o", linestyle="-", color="blue")
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.title("Class Distribution (Line Plot)")
    plt.grid(True)
    plt.show()

    # Pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(class_counts.values, labels=class_counts.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("tab10"))
    plt.title("Class Distribution (Pie Chart)")
    plt.show()

    # Heatmap of counts
    plt.figure(figsize=(6, 6))
    sns.heatmap(class_counts.values.reshape(-1, 1),
                annot=True, fmt="d", cmap="YlGnBu",
                yticklabels=class_counts.index, xticklabels=["Count"])
    plt.title("Class Distribution Heatmap")
    plt.ylabel("Classes")
    plt.xlabel("Image Count")
    plt.show()

    # -----------------------------
    # 📏 Image Size Distribution
    # -----------------------------
    widths, heights = [], []
    for path, _ in dataset.samples:
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue

    # Line plot for image widths and heights
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(widths)), widths, label="Width", color="blue", alpha=0.7)
    plt.plot(range(len(heights)), heights, label="Height", color="red", alpha=0.7)
    plt.legend()
    plt.title("Image Dimensions Across Dataset")
    plt.xlabel("Image Index")
    plt.ylabel("Pixels")
    plt.show()

    # Histogram of image widths & heights
    plt.figure(figsize=(8, 5))
    plt.hist(widths, bins=30, color="blue", alpha=0.6, label="Widths")
    plt.hist(heights, bins=30, color="red", alpha=0.6, label="Heights")
    plt.legend()
    plt.title("Image Dimension Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.show()

    # Scatter plot of Width vs Height
    plt.figure(figsize=(7, 6))
    plt.scatter(widths, heights, alpha=0.4, color="purple")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("Scatter Plot of Image Dimensions")
    plt.grid(True)
    plt.show()

    # KDE (density) plot of image dimensions
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=widths, y=heights, fill=True, cmap="Blues", thresh=0.05)
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("KDE Density of Image Dimensions")
    plt.show()

    # Boxplots for image width and height
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=[widths, heights])
    plt.xticks([0, 1], ["Widths", "Heights"])
    plt.title("Boxplot of Image Dimensions")
    plt.show()

    # -----------------------------
    # 📊 Extra: Correlation Heatmap of dimensions
    # -----------------------------
    df_dims = pd.DataFrame({"Width": widths, "Height": heights})
    corr = df_dims.corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Image Dimensions")
    plt.show()

    print("📊 Extended data analysis complete! All graphs generated.")
