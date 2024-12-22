from datasets import load_dataset
import pickle
import matplotlib.pyplot as plt

ds = load_dataset("NikolaiZhdanov/historical-checkers-games")
MAX_LENGTH = 400


if __name__ == "__main__":
    dataset = []

    for item in ds["train"]:
        moves = item["moves"][:MAX_LENGTH]
        while len(moves) < MAX_LENGTH:
            moves += "P"
        moves = moves.replace("P", "<pad>")
        dataset.append(moves)

    with open("dataset.pkl", "wb") as file:
        pickle.dump(dataset, file)
