import flwr as fl
from task import Net, get_weights, set_weights, test, get_dataloaders
import torch
import yaml

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

def get_evaluate_fn():
    model = Net(num_classes=CONFIG["training"]["num_classes"])
    _, val_loader = get_dataloaders()
    def evaluate(server_round, parameters, config):
        set_weights(model, parameters)
        loss, acc = test(model, val_loader)
        return loss, {"accuracy": acc}
    return evaluate

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        evaluate_fn=get_evaluate_fn(),
        initial_parameters=fl.common.ndarrays_to_parameters(get_weights(Net(num_classes=CONFIG["training"]["num_classes"])))
    )

    fl.server.start_server(
        server_address=CONFIG["server"]["address"],
        config=fl.server.ServerConfig(num_rounds=CONFIG["server"]["num_rounds"]),
        strategy=strategy,
    )
