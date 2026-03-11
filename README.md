MovieLens 1M Recommendation System

A movie rating prediction model built using deep learning on the MovieLens 1M dataset. This project demonstrates the practical application of embeddings, feature engineering, and MLP architectures for collaborative filtering.

Table of Contents

Project Overview
Features
Dataset
Project Structure
Installation
Usage
Results
Future Improvements
License

Project Overview
This project implements a Movie Recommendation System that predicts user ratings for movies based on user features (age, gender, occupation) and movie features (genre).

Key goals:
Practice deep learning fundamentals: embeddings, regularization, loss computation
Build a well-structured, reproducible project ready for GitHub
Track and visualize training and validation losses

Features
User, Movie, and Genre Embeddings: Encodes categorical features for input to the neural network
MLP Architecture: 3-layer fully connected network with ReLU activations and dropout
Training & Validation Loops: Tracks losses per epoch
Save/Load Model: Easy checkpointing
Loss Visualization: Plots training and validation curves

Dataset
MovieLens 1M dataset:
1,000,209 ratings from 6,045 users on 3,900+ movies
CSV files provided: users.csv, movies.csv, ratings.csv
Combined into a single dataset (data.csv) using the utils/unpack.py script

Project Structure
MovieLens1M/
│
├── src/
│   ├── dataset.py         # Loads data, creates DataLoaders, applies embeddings
│   ├── model.py           # Defines MLP and embeddings
│   └── train.py           # Training, validation, plotting, save/load
│
├── utils/
│   └── unpack.py          # Combines raw CSV files into one clean dataset
│
├── data/                  # Raw CSV files + combined data.csv
│   ├── users.csv
│   ├── movies.csv
│   ├── ratings.csv
│   └── data.csv
│
├── results/
│   ├── model.pth          # Saved trained model
│   ├── training_loss.png
│   └── validation_loss.png
│
└── requirements.txt       # Python dependencies

Installation
Clone the repository:

git clone https://github.com/yourusername/MovieLens1M.git
cd MovieLens1M

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

Install dependencies:
pip install -r requirements.txt
Note: If you want GPU support for PyTorch, follow PyTorch installation instructions
 to match your CUDA version.

Usage

Prepare the dataset (combines raw CSVs into data.csv):
python src/utils/unpack.py

Train the model:
python src/train.py

Trains the MLP on the MovieLens 1M dataset

Saves the model to results/model.pth

Plots training and validation loss curves

Load the trained model:
from src.train import load, model
load()

Predict ratings:
Use get_embedded_input from dataset.py to embed new samples
Feed through the model for prediction

Results

Training loss typically drops from ~17 → ~1.0 within a few epochs
Validation loss stabilizes around ~1.1, showing good convergence
Saved plots: training_loss.png, validation_loss.png

Future Improvements

Experiment with larger embeddings or deeper networks
Add user/movie biases or additional features (timestamps, tags)
Implement GPU acceleration for faster training
Try different loss functions (e.g., RMSE, Huber)
