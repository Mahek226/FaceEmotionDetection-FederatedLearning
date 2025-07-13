import flwr as fl
import torch
from task import Net, get_weights, set_weights, train, test, get_dataloaders
import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net(num_classes=CONFIG["training"]["num_classes"])
        self.trainloader, self.valloader = get_dataloaders()

    def get_parameters(self, config): return get_weights(self.model)
    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        self.model = train(self.model, self.trainloader, epochs=CONFIG["training"]["epochs"])
        return get_weights(self.model), len(self.trainloader.dataset), {}
    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, acc = test(self.model, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}

if __name__ == "__main__":
    fl.client.start_client(server_address=CONFIG["server"]["address"], client=FlowerClient().to_client())
