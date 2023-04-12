import torch.nn as nn
import torch
import os


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ModelA(nn.Module):
    def __init__(self, emb_size, hidden_size, out_class_number, word2vec_size, objective):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.class_number = out_class_number
        input_layerO1 = 21 + 20  # pssm.shape + hmm.shape
        input_layerO2 = word2vec_size
        input_layerO3 = 21 + 20 + 1  # pssm.shape + hmm.shape + structure length
        input_layerO4 = word2vec_size + 1  # word2vec_size + structure length
        input_layer = [input_layerO1, input_layerO2, input_layerO3, input_layerO4]
        self.input_layer_size = input_layer[objective - 1]

        self.mlp1 = nn.Sequential(
            nn.Linear(in_features=self.input_layer_size, out_features=self.emb_size // 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.emb_size // 2, out_features=self.emb_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5)
        )

        self.gru = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, bidirectional=True, num_layers=2,
                          dropout=0.5)

        self.mlp2 = nn.Sequential(
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.hidden_size, out_features=self.class_number),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5)
        )

    def forward(self, *inputs):
        profile, hot, hmm, embedding, struc_len, objective = inputs

        print(f'Profile array shape --> {profile.shape}')
        print(f'HMM array shape --> {hmm.shape}')
        print(f'One hot encoding array shape --> {hot.shape}')
        print(f'Embedding array shape --> {embedding.shape}')
        print(f'Structure Length array shape --> {struc_len.shape}')
        print(f'Objective array shape --> {objective.shape}')
        print('-----------')

        x1 = torch.cat([profile, hmm], dim=-1)
        x2 = embedding
        x3 = torch.cat([profile, hmm, struc_len], dim=-1)
        x4 = torch.cat([embedding, struc_len], dim=-1)
        x = [x1, x2, x3, x4]

        objective = objective[0].item()

        first_block = self.mlp1(x[objective])
        second_block = self.gru(first_block)
        third_block = self.mlp2(second_block[0])

        return third_block
