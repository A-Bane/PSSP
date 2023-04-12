# Protein Secondary Structure prediction
PSSP model using features PSSM matrix, HMM profile, Structure length average, Word2Vec Embeddings

- In this project, we investigate the effect of inputting an additional feature to a recurrent
neural network in efforts to improve the overall prediction performance. 

- Experimented with two input features, namely, evolutionary information and amino acid residue embeddings, both in presence and absence of the proposed structure length feature. 

- Our findings suggest that the model becomes heavily dependent on the average structure length, deducing imperfect predictions with the majority belonging to the majority class. 

- Additionally, the word2vec embeddings do not offer much potential to predict the secondary structures which can be attributed to the algorithm’s properties. The Word2Vec model generates context independent embeddings which might be affected due to the small vocabulary of the training dataset of 20 amino acids. 

