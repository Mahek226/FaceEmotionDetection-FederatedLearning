# fed_server.py
import flwr as fl
from task import Net, set_weights, test, get_weights, load_data

def get_evaluate_fn():
    def evaluate(server_round, parameters, config):  # ← fix the args
        print(f"[SERVER] Evaluating initial model at round {server_round}")
        model = Net(num_classes=14)  # Make sure this matches your dataset
        set_weights(model, parameters)
        _, val_loader, _ = load_data("downloaded_faces")
        loss, acc = test(model, val_loader, device="cpu")
        return loss, {"accuracy": acc}
    return evaluate

def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        evaluate_fn=get_evaluate_fn(),
        initial_parameters=fl.common.ndarrays_to_parameters(get_weights(Net(num_classes=14))),
    )
    print("[SERVER] Starting Flower server...")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
