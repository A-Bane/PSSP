import argparse
import pandas as pd
import torch

from dataload import *
from embedding import *
from model import *
from train import *
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def print_objective():
    print(f'1. Baseline using PSSM and HMM info')
    print(f'2. Just embedding features')
    print(f'3. PSSM and HMM using structure length')
    print(f'4. Embedding features + structure length')


def run(args):
    obj = ['Baseline using PSSM and HMM info', 'Just embedding features', 'PSSM and HMM using structure length',
           'Embedding features + structure length']
    print('-----')
    print(f'Objective = {args.objective}. {obj[args.objective - 1]}')

    # --------------------------- TRAIN LOADING -----------------------------

    # Load train
    profile_train, hot_train, label_train = load_6133()
    label_train = hot2labelQ3(label_train)
    hmm_train = load_hmm_training()

    list_struc_len = avg_length_structure(label_train, choice=1, structure_number=1, q=args.q)

    shape = list(return_avg_count_list(label_train, list_struc_len).shape)
    shape.append(1)
    shape = tuple(shape)
    struc_len_train = return_avg_count_list(label_train, list_struc_len).reshape(shape)

    objective = args.objective - 1

    # Word2Vec embeddings for train
    sequence = amino2d(hot_train)
    embedding_train = embeddings(sequence, args.word2vec_size, args.window, args.min_count, args.workers, args.sg)

    # --------------------------- TEST LOADING -----------------------------

    # Load test
    profile_test, hot_test, label_test = load_513()
    label_test = hot2labelQ3(label_test)
    hmm_test = load_hmm_testing()

    list_struc_len = avg_length_structure(label_test, choice=1, structure_number=1, q=args.q)

    shape = list(return_avg_count_list(label_test, list_struc_len).shape)
    shape.append(1)
    shape = tuple(shape)
    struc_len_test = return_avg_count_list(label_test, list_struc_len).reshape(shape)

    # Word2Vec embeddings for test
    sequence = amino2d(hot_test)
    embedding_test = embeddings(sequence, args.word2vec_size, args.window, args.min_count, args.workers, args.sg)

    # -------------------------------Converting train data to tensor----------------------------------------
    objective_train = [objective] * profile_train.shape[0]
    objective_train = torch.tensor(objective_train)
    profile_train = torch.tensor(profile_train).to(torch.float32)
    hot_train = torch.tensor(hot_train).to(torch.float32)
    label_train = torch.tensor(label_train).to(torch.float32)
    hmm_train = torch.tensor(hmm_train).to(torch.float32)
    struc_len_train = torch.tensor(struc_len_train).to(torch.float32)
    embedding_train = torch.tensor(embedding_train).to(torch.float32)

    # -------------------------------Converting test data to tensor----------------------------------------
    objective_test = [objective] * profile_test.shape[0]
    objective_test = torch.tensor(objective_test)
    profile_test = torch.tensor(profile_test).to(torch.float32)
    hot_test = torch.tensor(hot_test).to(torch.float32)
    label_test = torch.tensor(label_test).to(torch.float32)
    hmm_test = torch.tensor(hmm_test).to(torch.float32)
    struc_len_test = torch.tensor(struc_len_test).to(torch.float32)
    embedding_test = torch.tensor(embedding_test).to(torch.float32)

    # ------------------------------------------model------------------------------------------
    model = ModelA(args.emb_size, args.hidden_size, args.out_class_number, args.word2vec_size, args.objective)

    # ------------------------------------------train------------------------------------------

    training_data = (
        profile_train, hot_train, hmm_train, embedding_train, struc_len_train, objective_train, label_train)
    validation_data = (profile_test, hot_test, hmm_test, embedding_test, struc_len_test, objective_test, label_test)

    loss_over_epoch, total_acc = train(training_data, validation_data, model, args.epochs, args.batch_size, args.lr,
                                       args.lr_decay,
                                       args.weight_decay)
    # df = pd.DataFrame()
    # df['Accuracy'] = total_acc
    # df.to_excel(f'/Users/arghasreebanerjee/Downloads/acc_obj{args.objective}.xlsx')
    #
    # df = pd.DataFrame()
    # df['Loss'] = loss_over_epoch
    # df.to_excel('/Users/arghasreebanerjee/Downloads/loss_obj{args.objective}.xlsx')
    #
    # plt.plot(loss_over_epoch)
    # plt.savefig("/Users/arghasreebanerjee/Downloads/loss_obj{args.objective}.png")
    # plt.plot(total_acc)
    # plt.savefig("/Users/arghasreebanerjee/Downloads/accuracy_obj{args.objective}.png")


if __name__ == '__main__':
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    print_objective()
    parser = argparse.ArgumentParser(description='Settings to run PSSP model')
    parser.add_argument('--q', default=3, type=int, help='Q3 or Q8')
    parser.add_argument('--emb_size', default=512, type=int, help='Embedding Size of GRU')
    parser.add_argument('--hidden_size', default=256, type=int, help='Hidden Layer Size of GRU')
    parser.add_argument('--out_class_number', default=4, type=int, help='Output Layer Size = Total number of Classes')
    parser.add_argument('--word2vec_size', default=17, type=int, help='vector_size parameter for word2vec model')
    parser.add_argument('--objective', default=2, type=int, help='objective 1/2/3/4')
    parser.add_argument('--epochs', default=100, type=int, help='Total epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Total batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lr_decay', default=0.997, type=float, help='Learning rate decay')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay')
    parser.add_argument('--window', default=3, type=int, help='Window Size for word2vec model')
    parser.add_argument('--min_count', default=1, type=int, )
    parser.add_argument('--workers', default=3, type=int, )
    parser.add_argument('--sg', default=1, type=int)

    args = parser.parse_args()
    run(args)
