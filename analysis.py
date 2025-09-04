import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torchvision import transforms
from task import Net, download_kaggle_dataset, test, SafeImageFolder  # ✅ use SafeImageFolder


# -----------------------------
# Load all saved models
# -----------------------------
def load_all_models(num_classes, model_dir="."):
    models = {}
    for fname in os.listdir(model_dir):
        if fname.startswith("saved_model_round") and fname.endswith(".pth"):
            round_num = int(fname.replace("saved_model_round", "").replace(".pth", ""))
            model_path = os.path.join(model_dir, fname)

            model = Net(num_classes=num_classes)
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()
            models[round_num] = model
            print(f"✅ Loaded {fname}")
    return dict(sorted(models.items()))


# -----------------------------
# Collect predictions for metrics
# -----------------------------
def get_predictions(model, loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            # ✅ skip dummy images with label -1
            mask = labels != -1
            if mask.sum() == 0:
                continue

            images, labels = images[mask], labels[mask]
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


# -----------------------------
# Main Analysis
# -----------------------------
if __name__ == "__main__":
    # 1. Load dataset safely
    dataset_path = download_kaggle_dataset()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = SafeImageFolder(root=dataset_path, transform=transform)  # ✅ safe loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes
    num_classes = len(class_names)

    # 2. Load models
    models = load_all_models(num_classes, model_dir=".")

    # 3. Evaluate all models
    results = {"round": [], "accuracy": [], "loss": []}
    for round_num, model in models.items():
        loss, acc = test(model, test_loader)
        results["round"].append(round_num)
        results["accuracy"].append(acc)
        results["loss"].append(loss)
        print(f"Round {round_num}: Accuracy={acc:.4f}, Loss={loss:.4f}")

    # -----------------------------
    # 📊 Plot 1: Accuracy vs Round
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(results["round"], results["accuracy"], marker="o", label="Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Model Accuracy per Round")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_per_round.png")
    plt.show()

    # -----------------------------
    # 📊 Plot 2: Loss vs Round
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(results["round"], results["loss"], marker="s", color="red", label="Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Federated Model Loss per Round")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_per_round.png")
    plt.show()

    # -----------------------------
    # 📊 Plot 3: Accuracy & Loss on same plot
    # -----------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(results["round"], results["accuracy"], marker="o", color="blue", label="Accuracy")
    ax2.plot(results["round"], results["loss"], marker="s", color="red", label="Loss")

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy", color="blue")
    ax2.set_ylabel("Loss", color="red")
    plt.title("Accuracy & Loss per Round")
    plt.savefig("accuracy_loss_combined.png")
    plt.show()

    # -----------------------------
    # 📊 Confusion Matrix (last model)
    # -----------------------------
    last_round = max(models.keys())
    y_true, y_pred, y_probs = get_predictions(models[last_round], test_loader, class_names)

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - Round {last_round}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # -----------------------------
    # 📊 Per-class accuracy
    # -----------------------------
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_names, y=class_accuracy)
    plt.ylabel("Accuracy")
    plt.title(f"Per-class Accuracy - Round {last_round}")
    plt.savefig("per_class_accuracy.png")
    plt.show()

    # -----------------------------
    # 📊 Precision, Recall, F1
    # -----------------------------
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]

    for metric in metrics:
        scores = [report[cls][metric] for cls in class_names]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=class_names, y=scores)
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} per Class - Round {last_round}")
        plt.savefig(f"{metric}_per_class.png")
        plt.show()

    # -----------------------------
    # 📊 ROC Curves (multi-class One-vs-Rest)
    # -----------------------------
    if len(np.unique(y_true)) > 2:  # only if multi-class
        plt.figure(figsize=(8, 6))
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves - Round {last_round}")
        plt.legend()
        plt.savefig("roc_curves.png")
        plt.show()

    print("📊 All analysis graphs generated and saved!")
