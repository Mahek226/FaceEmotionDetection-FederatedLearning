# fed/client.py

from task import Net, get_weights, set_weights, train, test, load_data
import flwr as fl
from flwr.client import NumPyClient
import torch

print("[DEBUG] Top of client.py reached")

class FlowerClient(NumPyClient):
    def __init__(self):
        print("[DEBUG] Initializing FlowerClient")
        self.trainloader, self.valloader, self.testloader = load_data("downloaded_faces")
        
        # FIX: Safely get number of classes
        try:
            num_classes = len(self.trainloader.dataset.dataset.classes)
        except Exception as e:
            print(f"[ERROR] Could not determine number of classes: {e}")
            num_classes = 14  # fallback default, change if needed

        self.net = Net(num_classes=num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_weights(self.net)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader, epochs=1, device=self.device)
        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient().to_client())
