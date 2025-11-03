import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden = self.activation(self.W1(input_vector))

        # [to fill] obtain output layer representation
        output = self.W2(hidden)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)

        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument("--output_file", default="ffnn_predictions.txt", help="file to save predictions")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)

    train_losses = []
    val_accuracies = []

    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / (N // minibatch_size)
        train_losses.append(avg_train_loss)

        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))


        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16 
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
        print("Validation completed for epoch {}".format(epoch + 1))
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left Y-axis: Training Loss
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, marker='o', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Right Y-axis: Validation Accuracy
    ax2 = ax1.twinx()  # create a second y-axis
    color = 'tab:orange'
    ax2.set_ylabel('Validation Accuracy', color=color)
    ax2.plot(epochs, val_accuracies, color=color, marker='s', label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Highlight best epoch (highest validation accuracy)
    best_epoch = int(np.argmax(val_accuracies)) + 1
    best_acc = max(val_accuracies)
    ax2.axvline(best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')

    # Title and legends
    plt.title('Learning Curve: Training Loss & Validation Accuracy')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('ffnn_learning_curve.png')
    plt.show()
    print("Learning curve saved as 'ffnn_learning_curve.png'")

    # Testing phase
    if args.test_data:
        print("========== Testing the model ==========")
        # Load test data
        with open(args.test_data) as test_f:
            test_json = json.load(test_f)

        test_data = []
        for elt in test_json:
            test_data.append((elt["text"].split(), int(elt["stars"]-1)))

        # Convert to vector representation
        test_data = convert_to_vector_representation(test_data, word2index)

        # Test the model
        model.eval()
        correct = 0
        total = 0
        predictions = []
    
        with torch.no_grad():  # No need to track gradients during testing
            for input_vector, gold_label in tqdm(test_data):
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector).item()
                predictions.append((predicted_label + 1, gold_label + 1))  # Convert back to 1-5 scale
                correct += int(predicted_label == gold_label)
                total += 1
    
        # Calculate accuracy
        test_accuracy = correct / total
        print(f"Test accuracy: {test_accuracy:.4f}")

        # Save predictions
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("predicted_rating,true_rating\n")
            for pred, gold in predictions:
                f.write(f"{pred},{gold}\n")

        print(f"Predictions saved to {args.output_file}")

    # write out to results/test.out
    