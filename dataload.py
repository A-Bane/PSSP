import numpy as np
import os

import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
directory = '/Users/arghasreebanerjee/Downloads/MLPRNN-master/datasets/'


def load_513():
    """
    Loads cb513 dataset
    :return: pssm, one hot encoding of 20 amino acids, labels in one hot encoding form
    """
    df_test = np.load(f'{directory}cb513+profile_split.npy')
    df_test = np.reshape(df_test, (df_test.shape[0], 700, 57))
    pssm = df_test[:, :, 35:56]
    hot_encoding = df_test[:, :, :21]
    label = df_test[:, :, 22:31]

    return pssm, hot_encoding, label


def load_6133():
    """
    Loads cb6133 dataset
    :return: pssm, one hot encoding of 20 amino acids, labels in one hot encoding form
    """
    df_train = np.load(f'{directory}cullpdb+profile_6133_filtered.npy')
    df_train = np.reshape(df_train, (df_train.shape[0], 700, 57))
    pssm = df_train[:, :, 35:56]
    hot_encoding = df_train[:, :, :21]
    label = df_train[:, :, 22:31]

    return pssm, hot_encoding, label


def load_hmm_training():
    """
    Loads the hmm profiles for baseline (training)
    :return: HMM profiles
    """
    return np.load(f'{directory}hhm_train.npy')


def load_hmm_testing():
    """
    Loads the hmm profiles for baseline (testing)
    :return: HMM profiles
    """
    return np.load(f'{directory}hhm_test.npy')


def hot2labelQ8(label):
    """
    :param label: One hot encoded class label
    :return: A list of labels total number of proteins
    """
    classes = np.argmax(label, axis=2)
    classes[:, :] = classes[:, :] + 1
    classes[classes == 9] = 0
    return classes


def hot2labelQ3(label):
    """
    :param label: One hot encoded class label
    :return: A list of labels total number of proteins
    """
    classes = np.argmax(label, axis=2)
    q3 = [1, 3, 3, 2, 2, 2, 1, 1, 0]
    new_classes = np.empty(classes.shape)
    for original, later in enumerate(q3):
        new_classes[classes == original] = later

    return new_classes.astype(int)


def structure_len(label):
    """
    :param label: List of Structure numbers
    :return: list of length of structure
    """
    total_proteins = label.shape[0]
    len_structure = []
    len_each = []
    index_list = []
    return_list1 = []
    return_list2 = []
    zeros = []
    flag = 0
    index_list.append(-1)

    for i in range(total_proteins):
        for s in range(len(label[i])):
            if label[i][s] != 0:
                if s == len(label[i]) - 1:  # Assuming last element == second last element
                    len_structure.append(len(len_each) + 1)
                    len_each = []
                    index_list.append(s)
                elif label[i][s] == label[i][s + 1]:
                    len_each.append(label[i][s])
                else:
                    len_structure.append(len(len_each) + 1)
                    len_each = []
                    index_list.append(s)  # last index of the count saved
            else:
                zeros.append(label[i][s])

        for s in len_structure:
            l = [s] * (index_list[flag + 1] - index_list[flag])
            return_list1.extend(l)
            flag += 1
        return_list1.extend(zeros)
        zeros = []
        return_list2.append(return_list1)
        len_structure = []
        index_list = [-1]
        return_list1 = []
        flag = 0
    return_list2 = np.array([np.array(xi) for xi in return_list2])
    # print(return_list2)

    return return_list2


def length(one_hot):
    """
    :param one_hot: List of all labels, 0 means no amino acid residue
    :return: List of length of amino acid sequence of all 5534 proteins
    """

    total_sequence = one_hot.shape[0]
    len_list = []
    for i in range(total_sequence):
        len_list.append(sum(np.max(one_hot[i], axis=1)).astype(int))

    return len_list


def amino2d(one_hot):
    """
    Converts amino acid one hot to A,C,E etc
    :param one_hot: one hot encoding of amino acid residue in each position
    :return:
    """
    amino_acids = list('ACEDGFIHKMLNQPSRTWVYX')
    list_amino_acid = []
    total_proteins = one_hot.shape[0]
    leng = length(one_hot)
    for i in range(total_proteins):
        protein = one_hot[i]
        amino_acid = [amino_acids[i] for i in np.argmax(protein, axis=1)]
        list_amino_acid.append(amino_acid)
        list_amino_acid[i] = list_amino_acid[i][:leng[i]]
        list_amino_acid[i].extend([''] * (len(protein) - leng[i]))
        # print(i)

    return list_amino_acid


def avg_length_structure(label, choice=1, structure_number=1, q=8):
    """
    :param choice: 1 for average, 2 for majority
    :return: array consisting of average length
    """
    Struc = []
    count_df = []
    temp_list = []
    if q == 3:
        Struc = [1, 2, 3]
        temp_list = [[], [], []]
    elif q == 8:
        print("inside 8")
        Struc = [1, 2, 3, 4, 5, 6, 7, 8]
        temp_list = [[], [], [], [], [], [], [], []]
    total_proteins = label.shape[0]

    main_list = []
    count = 0

    for i in range(total_proteins):
        for structure in Struc:
            for s in range(len(label[i])):
                if s == len(label[i]) - 1:
                    if label[i][s - 1] == structure and label[i][s] == structure:
                        count += 1
                        break
                else:
                    if label[i][s] == structure and label[i][s + 1] == structure:
                        count += 1
                    elif label[i][s] == structure and label[i][s + 1] != structure:
                        count += 1
                        temp_list[structure - 1].append(count)
                        count = 0

    if choice == 1:
        count_df = [round(np.average(np.array(i))) for i in temp_list]
    elif choice == 2:
        count_df = pd.DataFrame()
        i = temp_list[structure_number - 1]
        count_df[f'Len Structure {structure_number}'] = list(set(i))
        count_df[f'Count Structure {structure_number}'] = [i.count(j) for j in set(i)]
    return count_df


def return_avg_count_list(label, count_list):
    """
    :param label: all numeric labels
    :param count_list: e.g [4,10,5]
    :return: for structure 1, return 4 and so on
    """
    temp_list = []
    return_list = []
    total_proteins = label.shape[0]

    for i in range(total_proteins):
        for label_ in label[i]:
            if label_ == 0:
                temp_list.append(0)
            else:
                temp_list.append(count_list[label_ - 1])
        return_list.append(np.array(temp_list))
        temp_list = []

    return np.array(return_list)
