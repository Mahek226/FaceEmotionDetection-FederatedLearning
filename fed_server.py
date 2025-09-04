# import flwr as fl
# from task import Net, get_weights, set_weights, test, get_dataloaders
# import torch
# import yaml

# with open("config.yaml", "r") as f:
#     CONFIG = yaml.safe_load(f)

# def get_evaluate_fn():
#     model = Net(num_classes=CONFIG["training"]["num_classes"])
#     _, val_loader = get_dataloaders()
#     def evaluate(server_round, parameters, config):
#         set_weights(model, parameters)
#         loss, acc = test(model, val_loader)
#         return loss, {"accuracy": acc}
#     return evaluate

# if __name__ == "__main__":
#     strategy = fl.server.strategy.FedAvg(
#         fraction_fit=1.0,
#         min_fit_clients=1,
#         min_available_clients=1,
#         evaluate_fn=get_evaluate_fn(),
#         initial_parameters=fl.common.ndarrays_to_parameters(get_weights(Net(num_classes=CONFIG["training"]["num_classes"])))
#     )

#     fl.server.start_server(
#         server_address=CONFIG["server"]["address"],
#         config=fl.server.ServerConfig(num_rounds=CONFIG["server"]["num_rounds"]),
#         strategy=strategy,
#     )


import flwr as fl
import torch
import yaml
from task import Net, get_weights, set_weights, test, get_dataloaders, save_model


# -----------------------------
# Load config
# -----------------------------
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


# -----------------------------
# Evaluation Function
# -----------------------------
def get_evaluate_fn(num_classes):
    # Recreate model
    model = Net(num_classes=num_classes)
    _, val_loader, _, _ = get_dataloaders()

    def evaluate(server_round, parameters, config):
        set_weights(model, parameters)
        loss, acc = test(model, val_loader)
        print(f"[Server] Round {server_round} - Val Loss: {loss:.4f}, Val Acc: {acc:.4f}")

        # ✅ Save model every round (optional)
        save_model(model, f"saved_model_round{server_round}.pth")

        return loss, {"accuracy": acc}

    return evaluate


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # ✅ Load data once on server to detect number of classes
    _, _, _, num_classes = get_dataloaders()

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        evaluate_fn=get_evaluate_fn(num_classes),
        initial_parameters=fl.common.ndarrays_to_parameters(
            get_weights(Net(num_classes=num_classes))
        ),
    )

    # Start Flower server
    history = fl.server.start_server(
        server_address=CONFIG["server"]["address"],
        config=fl.server.ServerConfig(num_rounds=CONFIG["server"]["num_rounds"]),
        strategy=strategy,
    )

    # ✅ Final save after training finishes
    global_model = Net(num_classes=num_classes)
    final_params = strategy.initial_parameters
    if history and history.global_parameters:
        final_params = history.global_parameters
    set_weights(global_model, fl.common.parameters_to_ndarrays(final_params))
    save_model(global_model, "saved_model.pth")
