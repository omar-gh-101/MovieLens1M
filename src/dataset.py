import warnings
import csv

import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data():
    """
    loads data from data.csv file
    """
    try:
        data = []

        with open('../data/data.csv',mode='r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                data.append(list(map(int,row)))

        return data
    except FileNotFoundError:
        warnings.warn("File not found")

def get_loaders(train_batch_size,test_batch_size):
    """
    set-ups and returns the training and testing loaders
    so that they can be used in training
    """
    raw_data = load_data()

    input_data = [[row[0],row[1],row[2],row[3],row[4],row[5]] for row in raw_data]
    output_data = [row[6] for row in raw_data]

    training_percentage = 0.8

    training_input = input_data[:int(len(input_data)*training_percentage)]
    training_output = output_data[:int(len(input_data)*training_percentage)]
    testing_input = input_data[int(len(input_data)*training_percentage):]
    testing_output = output_data[int(len(input_data)*training_percentage):]

    training_input_tensor = torch.tensor(training_input, dtype=torch.long)
    training_output_tensor = torch.tensor(training_output, dtype=torch.float)
    testing_input_tensor = torch.tensor(testing_input, dtype=torch.long)
    testing_output_tensor = torch.tensor(testing_output, dtype=torch.float)

    training_dataset = TensorDataset(training_input_tensor, training_output_tensor)
    testing_dataset = TensorDataset(testing_input_tensor, testing_output_tensor)

    training_loader = DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True)
    testing_loader = DataLoader(testing_dataset, batch_size=test_batch_size, shuffle=True)

    return training_loader, testing_loader

def get_embedded_input(input,user_embeddings,movie_embeddings,genre_embeddings):
    """
    Applies embeddings to categorical features in input.

    features:
        - movie
        - user
        - genre
    """
    users_ids = input[:,0]
    user_embedding = user_embeddings(users_ids)

    genders = input[:,1].unsqueeze(1)
    ages = input[:,2].unsqueeze(1)
    occupations = input[:,3].unsqueeze(1)

    movies_ids = input[:,4]
    movie_embedding = movie_embeddings(movies_ids)
    genres = input[:,5]
    genres_embedding = genre_embeddings(genres)

    features = torch.cat(
        [user_embedding,genders,ages,occupations,movie_embedding,genres_embedding],
        dim=1
    )

    return features


