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
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        outputs, hidden = self.rnn(inputs)

        # [to fill] obtain output layer representations
        output_layer = self.W(outputs)

        # [to fill] sum over output 
        summed_outputs = torch.sum(output_layer, dim=0)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(summed_outputs)

        return predicted_vector


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
    parser.add_argument("--output_file", default="rnn_predictions.txt", help="file to save predictions")
    parser.add_argument("--load_model", default=None, help="path to load saved model")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    # lists for plotting
    epoch_train_losses = []
    epoch_val_accuracies = []

    last_train_accuracy = 0
    last_validation_accuracy = 0

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                # Transform the input into required shape
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        trainning_accuracy = correct/total


        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct/total

        # Record metrics for plotting
        avg_train_loss = (loss_total / loss_count).item()  # average training loss for the epoch
        epoch_train_losses.append(avg_train_loss)
        epoch_val_accuracies.append(validation_accuracy)
        print(f"Average training loss for epoch {epoch+1}: {avg_train_loss:.4f}")

        if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1

    # Plot Learning Curve with dual y-axes
    plt.figure(figsize=(8, 5))

    epochs = range(1, len(epoch_train_losses) + 1)

    # Left Y-axis: Training Loss
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, epoch_train_losses, color=color, marker='o', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Right Y-axis: Validation Accuracy
    ax2 = ax1.twinx()  # create a second y-axis
    color = 'tab:orange'
    ax2.set_ylabel('Validation Accuracy', color=color)
    ax2.plot(epochs, epoch_val_accuracies, color=color, marker='s', label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and save
    plt.title('Learning Curve: Training Loss & Validation Accuracy')

    # Optional: highlight best epoch
    best_epoch = int(np.argmax(epoch_val_accuracies)) + 1
    best_acc = max(epoch_val_accuracies)
    ax2.axvline(best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('rnn_learning_curve.png')
    plt.show()
    print("Learning curve saved as 'rnn_learning_curve.png'")


    # Testing the model
    if args.test_data:
        print("========== Testing the model ==========")

        # Load saved model if specified
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
            print(f"Loaded model from {args.load_model}")

        # Load test data
        with open(args.test_data) as test_f:
            test_json = json.load(test_f)

        test_data = []
        for elt in test_json:
            test_data.append((elt["text"].split(), int(elt["stars"]-1)))

        # Test the model
        model.eval()
        correct = 0
        total = 0
        predictions = []

        with torch.no_grad():  # No need to track gradients during testing
            for input_words, gold_label in tqdm(test_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Handle empty input
                if len(input_words) == 0:
                    predicted_label = 2  # Default to middle rating
                    predictions.append((predicted_label + 1, gold_label + 1))
                    continue

                # Convert to vectors
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)

                # Get prediction
                output = model(vectors)
                predicted_label = torch.argmax(output).item()
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


    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
