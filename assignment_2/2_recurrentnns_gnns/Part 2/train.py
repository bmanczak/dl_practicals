# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical # used for random sampling


###############################################################################

# python train.py --txt_file "assets/book_EN_democracy_in_the_US.txt" --lstm_num_hidden 16  --sample_every 500 --train_steps 1500
# python train.py --txt_file "assets/book_EN_democracy_in_the_US.txt" --lstm_num_hidden 1024 --batch_size 256 --learning_rate 5e-4 --embedding_dim 256 --sample_every 500 --train_steps 1500

def generate_text(model, temperature, num_examples, example_length, voc_size, lstm_num_layers, lstm_num_hidden,device, random_sampling=False):
    """Uses the model to randomly create batch_size sentences from characters in a dataset"""
    start_letters = [torch.tensor(np.random.randint(0, voc_size)) for i in
                     range(0, num_examples)]  # get num_examples random letters
    letters = torch.stack(start_letters).view(1, -1).to(
        device)  # initial shape [1, num_examples], 0-th dim will increase up to example_lentgh

    h = (torch.zeros((lstm_num_layers * 1, num_examples, lstm_num_hidden)).to(device),
         torch.zeros((lstm_num_layers * 1, num_examples, lstm_num_hidden)).to(device))  # initialize the hidden state

    next_letter_given_previous = letters  # initalize next letter to the random letters

    for letter_num in range(example_length - 1):
        next_letter_given_previous = next_letter_given_previous.to(device)

        log_probs, h = model(next_letter_given_previous, h, temperature)  # store the log probs and the hidden state

        if random_sampling == True:
            m = Categorical(logits=log_probs[0])  # sample the
            next_letter_given_previous = m.sample().view(1, -1) # sample the next letter
        else:
            next_letter_given_previous = torch.argmax(log_probs, dim=2)  # get the next most probable letter

        letters = torch.cat((letters, next_letter_given_previous), dim=0)  # add letters to the generated sequence

    return letters

def finish_sentence(input_sentence:str, dataset, device, example_length, model):
  """Finished the input_sentence """

  sen_to_id = [torch.tensor(dataset._char_to_ix[char]) for char in input_sentence]

  letters = torch.stack(sen_to_id).view(-1,1).to(device) # shape [starting length, 1].shape

  #next_letter_given_previous = letters # initalize next letter to the random letters

  for letter_num in range(example_length):
    next_letter_given_previous = torch.argmax(model(letters)[0][[-1]], dim = 2)
    letters = torch.cat((letters, next_letter_given_previous), dim=0)

  return dataset.convert_to_string([letter.item() for letter in letters])

def train(config):

    writer = SummaryWriter()
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(filename=config.txt_file, seq_length=config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size)
    voc_size = dataset._vocab_size

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, vocabulary_size = voc_size,
                                lstm_num_hidden=config.lstm_num_hidden, lstm_num_layers=config.lstm_num_layers,
                                device = config.device, embedding_dim = config.embedding_dim, drop_prob = config.drop_prob)
    if torch.cuda.is_available():
        model.cuda()

    # Setup the loss and optimizer
    criterion = nn.NLLLoss()  #
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    best_acc = 0

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        model.train()

        # Only for time measurement of step through network
        t1 = time.time()

        # Move to GPU
        batch_inputs = torch.stack(batch_inputs).to(device)  # [batch_size, seq_length]
        batch_targets = torch.stack(batch_targets).to(device)  # [batch_size, seq_length]

        # Reset for next iteration
        model.zero_grad()

        # Forward pass
        log_probs, _ = model(batch_inputs)  # [seq_length, batch_size, voc_size]

        # Calculate loss, gradients
        loss = criterion(log_probs.view(-1, voc_size), batch_targets.view(-1))
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)

        # Update network parameters
        optimizer.step()

        predictions = torch.argmax(log_probs, dim=2)
        correct = (predictions == batch_targets).sum().item()
        accuracy = correct / (config.batch_size * config.seq_length)

        # Log plots to tensorboard
        writer.add_scalar("Loss", loss, step)
        writer.add_scalar("Accuracy", accuracy, step)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if (step + 1) % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))

        if (step + 1) % config.sample_every == 0:
            example_sentences = generate_text(model, 1, config.num_examples, config.generate_seq_length,
                                              voc_size, lstm_num_hidden= config.lstm_num_hidden, lstm_num_layers=config.lstm_num_layers,device = device) # sample with no change in temperature
            print("[INFO]: generating sentences ... ")
            for sentence_num in range(config.num_examples):
                print(dataset.convert_to_string([letter.item() for letter in example_sentences[:, sentence_num]]))

        if (step + 1) % config.checkpoint_every == 0:
            if accuracy > best_acc:  # then save model
                print("[INFO]: Saving model with {} accuracy at step {} ...".format(round(accuracy, 4), step))
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, config.save_path)
                best_acc = accuracy # update best seen accuracy

        if step == config.train_steps:
            break

    writer.flush()
    print('[INFO]: Done training.')

    print("[INFO]: Generating sentences with different temperatures ...")

    small_temp = generate_text(model, 0.5, config.num_examples, config.generate_seq_length, voc_size, lstm_num_hidden= config.lstm_num_hidden,
                               lstm_num_layers=config.lstm_num_layers,device = device, random_sampling= True)
    medium_temp = generate_text(model, 1, config.num_examples, config.generate_seq_length, voc_size, lstm_num_hidden= config.lstm_num_hidden,
                               lstm_num_layers=config.lstm_num_layers,device = device, random_sampling= True)
    big_temp = generate_text(model, 2, config.num_examples, config.generate_seq_length, voc_size, lstm_num_hidden= config.lstm_num_hidden,
                               lstm_num_layers=config.lstm_num_layers,device = device, random_sampling= True)

    print("[INFO]: Temp 0.5")
    for sentence_num in range(config.num_examples):
        #print(sentence_num)
        print(dataset.convert_to_string([letter.item() for letter in small_temp[:, sentence_num]]))

    print("[INFO]: Temp 1")
    for sentence_num in range(config.num_examples):
        print(dataset.convert_to_string([letter.item() for letter in medium_temp[:, sentence_num]]))

    print("[INFO]: Temp 2")
    for sentence_num in range(config.num_examples):
        print(dataset.convert_to_string([letter.item() for letter in big_temp[:, sentence_num]]))

    print(["[INFO}: Finishing sentences"])
    finish_these = ["The heart of democracy is", "Liberty is", "People of", "Equality", "Slavery", "America is",
                    "North"]

    for sentence in finish_these:
        print(sentence, " "*10, finish_sentence(sentence, dataset, device, config.generate_seq_length, model))


###############################################################################
###############################################################################

#python train.py --txt_file "assets/book_EN_democracy_in_the_US.txt"
if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # Text generation parameters
    parser.add_argument('--num_examples', type=int, default=5,
                        help='How many example sentences to generate per sample_every')
    parser.add_argument('--sample_every', type=int, default=1000,
                        help='How often to sample from the model')
    parser.add_argument('--generate_seq_length', type=int, default=50,
                        help='How long should the generated sentences?')
    parser.add_argument('--temperature', type=int, default=1,
                        help='What temperature to use while sampling?')


    # It is not necessary to implement the following three params,
    # but it may help training.
    #parser.add_argument('--learning_rate_decay', type=float, default=0.96,
    #                    help='Learning rate decay fraction')
    #parser.add_argument('--learning_rate_step', type=int, default=5000,
    #                    help='Learning rate step')
    parser.add_argument('--drop_prob', type=float, default=0,
                        help='Dropout probability')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='The dimension of the embedding')

    parser.add_argument('--train_steps', type=int, default=3000,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--save_path', type=str, default="model_lstm.pt",
                        help='Directory where the model checkpoints should be saved')
    parser.add_argument('--print_every', type=int, default=200,
                        help='How often to print training progress')
    parser.add_argument('--checkpoint_every', type=int, default= 50,
                        help='How often to save a checkpoint. Only save if accuracy imporves.')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)
