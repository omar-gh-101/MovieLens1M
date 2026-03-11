import torch
import torch.nn as nn

"""
MLP for MovieLens rating prediction
- Input: user + movie + genre embeddings + numeric features
- Layers: 128 -> 64 -> 1
- Dropout: 0.2 after first layer
- Optimizer: Adam with weight decay 1e-5
- Scheduler: CosineAnnealingLR
"""

movie_embeddings = nn.Embedding(3955,32)
user_embeddings = nn.Embedding(6045,32)
genre_embeddings = nn.Embedding(305,16)

input_size = user_embeddings.embedding_dim + movie_embeddings.embedding_dim + genre_embeddings.embedding_dim + 3

epochs = 20

train_batch_size = 512
test_batch_size = 256

model = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(
    model.parameters(),lr=1e-4,weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

criterion = nn.MSELoss()



