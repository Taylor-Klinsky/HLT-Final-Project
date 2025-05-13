"""
Taylor Klinsky
HLT Final Project
FFNN implementation with GloVe
"""

import json
import random
from argparse import ArgumentParser
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

unk = '<UNK>'  # Unknown word token

class FFNN(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, hidden_dims):
        super(FFNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        embedding_dim = embedding_matrix.shape[1]
        self.hidden_layers = nn.ModuleList()
        self.activation = nn.ReLU()

        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Output 4 logits for 5 ordered classes (1–5)
        self.output_layer = nn.Linear(prev_dim, 4)

    def forward(self, input_indices):
        embeddings = self.embedding(input_indices)
        doc_embedding = torch.mean(embeddings, dim=0)
        x = doc_embedding
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        logits = self.output_layer(x)
        return logits

    def coral_loss(self, logits, encoded_target):
        sigmoid = torch.sigmoid(logits)
        loss_fn = nn.BCELoss()
        return loss_fn(sigmoid, encoded_target)

def encode_labels(y, num_classes=5):
    """Encode label as a vector of thresholds (e.g., 2 -> [1, 1, 0, 0])"""
    encoded = torch.zeros(num_classes - 1)
    encoded[:y] = 1
    return encoded

def make_vocab(reviews):
    """
    :param reviews: 1D array of review text
    :return: set of words in the review text
    """
    vocab = set()  # Vocabulary is a set, no duplicates
    for review in reviews:
        for word in review:
            vocab.add(word)
    return vocab

def make_indices(vocab):
    """
    :param vocab: 1D array of the vocabulary
    :return: the unsorted vocabulary
    """
    vocab_list = sorted(vocab)  # Sort the vocabulary in alphabetical order
    vocab.add(unk)  # Also add <UNK> to the end of the vocabulary
    vocab_list.append(unk)  # Add the unknown token to the vocabulary
    word_to_index = {}
    index_to_word = {}
    for index, word in enumerate(vocab_list):
        word_to_index[word] = index
        index_to_word[index] = word
    return vocab, word_to_index, index_to_word

def convert_to_index_sequences(data, word_to_index):
    """
    Necessary for a word embedding approach
    :param data: The original input
    :param word_to_index: word_to_index dictionary of words
    :return: The vectorized data
    """
    vectorized_data = []
    for document, y in data:
        index_sequence = [word_to_index.get(word, word_to_index[unk]) for word in document]
        vectorized_data.append((index_sequence, y))
    return vectorized_data

def load_glove_embeddings(glove_path, embedding_dim):
    """
    :param glove_path:
    :param embedding_dim:
    :return: Glove embeddings
    """
    embeddings = {}
    with open(glove_path, 'r', encoding='utf8') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = torch.tensor([float(x) for x in parts [1:]], dtype=torch.float)
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    return embeddings

def create_embedding_matrix(word_to_index, glove_embeddings, embedding_dim):
    vocab_size = len(word_to_index)
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)

    for word, index in word_to_index.items():
        if word in glove_embeddings:
            embedding_matrix[index] = glove_embeddings[word]
        else:
            embedding_matrix[index] = torch.randn(embedding_dim) * 0.1  # Gives a random value for out of vocabulary words
    return embedding_matrix


def load_data(data):
    with open(data, 'r', encoding='utf-8') as data_f:
        unsplit_data = json.load(data_f)

    data_array = []
    for elt in unsplit_data:
        data_array.append((elt["text"].split(), int(elt["rating"])))
    return data_array

if __name__ == "__main__":
    # Hyperparameters: hidden layer dimensions, epochs
    parser = ArgumentParser()
    parser.add_argument("-hl", "--hidden_layers", type=int, nargs='+', required=True, help="List of hidden layer sizes, e.g. 128 64")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Maximum number of epochs")
    parser.add_argument("--train_data", required=True, help="Path to training data")
    parser.add_argument("--test_data", required=False, help="Path to test data")
    parser.add_argument("--glove_data", required=True, help="Path to GloVe data")
    args = parser.parse_args()

    print("Epochs: %d" % args.epochs)
    print("Hidden layers:", args.hidden_layers)

    # Load data
    print("========== Loading data ==========")
    data = load_data(args.train_data)
    if args.test_data is not None:
        test_data = load_data(args.test_data)
    else:
        test_data = None

    # Always split validation from training data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    print("Size of train set: %d" % len(train_data))
    print("Size of test set: %d" % len(test_data))

    vocab = make_vocab([doc for doc, _ in train_data])
    vocab, word_to_index, index_to_word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_index_sequences(train_data, word_to_index)
    test_data = convert_to_index_sequences(test_data, word_to_index)
    val_data = convert_to_index_sequences(val_data, word_to_index)

    glove_embeddings = load_glove_embeddings(args.glove_data, 300)
    embedding_matrix = create_embedding_matrix(word_to_index, glove_embeddings, 300)
    model = FFNN(vocab_size=len(vocab), embedding_matrix=embedding_matrix, hidden_dims=args.hidden_layers)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    print("========== Training for {} epochs ==========".format(args.epochs))
    stopping_condition = False
    epoch = 0
    train_accuracy = 0
    validation_accuracy = 0
    last_train_accuracy = 0
    last_validation_accuracy = 0

    while not stopping_condition and epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 32
        N = len(train_data)
        start_time = time.time()

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None

            for example_index in range(minibatch_size):
                last_train_accuracy = train_accuracy
                index = minibatch_index * minibatch_size + example_index
                input_indices = torch.tensor(train_data[index][0], dtype=torch.long)
                gold_label = train_data[index][1]

                logits = model(input_indices)
                encoded_target = encode_labels(gold_label).to(logits.device)
                example_loss = model.coral_loss(logits, encoded_target)

                # Convert logits to predicted class
                predicted_label = int(torch.sum(torch.sigmoid(logits) > 0.5).item())
                predicted_label = max(0, min(4, predicted_label))  # clamp for safety

                correct += int(predicted_label == gold_label)
                total += 1

                example_loss = model.coral_loss(logits, encoded_target)
                loss = example_loss if loss is None else loss + example_loss
                train_accuracy = correct / total

            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_accuracy))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        # Validation loop
        last_validation_accuracy = validation_accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_indices, gold_label in val_data:
                input_tensor = torch.tensor(input_indices, dtype=torch.long)
                logits = model(input_tensor)
                encoded_target = encode_labels(gold_label).to(logits.device)
                example_loss = model.coral_loss(logits, encoded_target)

                # Convert logits to predicted class
                predicted_label = int(torch.sum(torch.sigmoid(logits) > 0.5).item())
                predicted_label = max(0, min(4, predicted_label))  # clamp for safety

                correct += int(predicted_label == gold_label)
                total += 1

        validation_accuracy = correct / total
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

        if epoch > 0 and validation_accuracy < last_validation_accuracy and train_accuracy > last_train_accuracy:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = train_accuracy
        epoch += 1

    # Run testing once, with the best number of epochs
    if args.test_data is not None:
        print("========== Testing ==========")
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for input_indices, gold_label in test_data:
                input_tensor = torch.tensor(input_indices, dtype=torch.long)
                logits = model(input_tensor)
                encoded_target = encode_labels(gold_label).to(logits.device)
                example_loss = model.coral_loss(logits, encoded_target)

                # Convert logits to predicted class
                predicted_label = int(torch.sum(torch.sigmoid(logits) > 0.5).item())
                predicted_label = min(max(predicted_label, 0), 4)

                all_preds.append(predicted_label)
                all_labels.append(gold_label)
                correct += int(predicted_label == gold_label)
                total += 1

        test_accuracy = correct / total
        print("Test accuracy: {:.2f}".format(test_accuracy))

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()