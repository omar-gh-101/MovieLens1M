from typing import List
from dataset import *
from model import *

import matplotlib.pyplot as plt

train_loader,test_loader = get_loaders(train_batch_size,test_batch_size)

def train() -> None:
    """
    trains the model while keeping track of losses for evaluation
    plots the training and validation losses at the end
    """
    model.train()

    training_losses = []
    validation_losses = []
    count = []

    initial_loss = validation()

    training_losses.append(initial_loss)
    validation_losses.append(initial_loss)
    count.append(0)

    for epoch in range(epochs):
        training_loss = 0
        for x,y in train_loader:
            x = get_embedded_input(x,user_embeddings,movie_embeddings,genre_embeddings)

            prediction = model(x)

            loss = criterion(prediction,y.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            training_loss += loss.item()

        scheduler.step()

        training_loss /= len(train_loader)

        validation_loss = validation()

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        count.append(epoch + 1)

        print(f"Epoch ({epoch + 1}/{epochs}) Training Loss: {training_loss} Validation Loss: {validation_loss}")

    plot(validation_losses, training_losses, count)

def validation() -> float:
    """
    validates the model by calculating and returning the mean validation loss
    """
    model.eval()

    validation_loss = 0

    with torch.no_grad():
        for x,y in test_loader:
            x = get_embedded_input(x,user_embeddings,movie_embeddings,genre_embeddings)
            prediction = model(x)

            loss = criterion(prediction,y.unsqueeze(1))
            validation_loss += loss.item()

        validation_loss /= len(test_loader)

    return validation_loss

def save() -> None:
    """
    saves the model
    """
    torch.save(model.state_dict(), "../results/model.pth")

def load() -> None:
    """
    loads the saved model
    """
    model.load_state_dict(torch.load("../results/model.pth"))

def plot(validation_losses:List[float], training_losses:List[float],count:List[int]) -> None:
    """
    plots the training and validation losses
    """
    plt.plot(count,validation_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title("Validation Graph")
    plt.show()
    plt.plot(count,training_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Graph")
    plt.show()

if __name__ == "__main__":
    train()
    save()
