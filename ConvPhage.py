# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPool1D, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam

benchmark_file = './Data/phage_data_nmicro2017/processed_benchmark_set.csv'
hmm_results_dir = './Data/phage_data_nmicro2017/hmmsearch_out/'
base_dir = './Data/classifier_data/'
hmm_data_dir = './Data/protein_domain_data/domain_alignments_and_hmms/'
fasta_dir = './Data/phage_data_nmicro2017/phage_fasta_files/'

df = pd.read_csv(benchmark_file, index_col=0)
print('Starting shape:', df.shape)
df = df[df['Temperate (empirical)'] != 'Unspecified']
print('New shape (should be identical):', df.shape)

df['Identifier_AJH'] = ''
df.at[df[df['Database source'] == 'NCBI RefSeq'].index, 'Identifier_AJH'] = \
    df[df['Database source'] == 'NCBI RefSeq']['RefSeq accession number']
df.at[df[df['Database source'] == 'Actinobacteriophage_785'].index, 'Identifier_AJH'] = \
    df[df['Database source'] == 'Actinobacteriophage_785']['Virus identifier used for the analysis'].str.split('_').str[
        0]
print('New shape (+1 new column):', df.shape)

# %%
file_ending = '.fasta'
base_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

counter = 0
cutoff = 5000
for index in df.index:
    if df.loc[index]['Database source'] == 'NCBI RefSeq':
        file_name = df.loc[index]['RefSeq accession number'] + file_ending
    elif df.loc[index]['Database source'] == 'Actinobacteriophage_785':
        file_name = df.loc[index]['Virus identifier used for the analysis'].split('_')[0].lower() + \
                    file_ending
    with open(fasta_dir + file_name, 'r') as infile:
        fasta_sequences = SeqIO.parse(infile, 'fasta')
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            counter = counter + np.floor(len(sequence) / cutoff)

sequence_condensed = np.zeros((counter.astype('int'), cutoff))
labels = np.zeros(shape=(counter.astype('int'), 1))
nrow = 0
for index in df.index:
    if df.loc[index]['Database source'] == 'NCBI RefSeq':
        file_name = df.loc[index]['RefSeq accession number'] + file_ending
    elif df.loc[index]['Database source'] == 'Actinobacteriophage_785':
        file_name = df.loc[index]['Virus identifier used for the analysis'].split('_')[0].lower() + \
                    file_ending
    with open(fasta_dir + file_name, 'r') as infile:
        fasta_sequences = SeqIO.parse(infile, 'fasta')
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)

        is_temperate = int(df.at[index, 'Temperate (empirical)'] == 'yes')

        repeat = np.floor(len(sequence) / cutoff).astype('int')
        for segment in range(0, repeat):
            start = segment * cutoff
            end = (segment + 1) * cutoff
            subsequence = sequence[start:end]
            if len(subsequence) == cutoff:
                subsequence_list = list(subsequence)
                subsequence_list_num = [base_dict[i] if i in base_dict else 5 for i in subsequence_list]
                subsequence_array = np.array(subsequence_list_num)
                sequence_condensed[nrow, :] = subsequence_array
                labels[nrow] = is_temperate
                nrow = nrow + 1

# labels = pd.DataFrame(index=df.index)
# labels['binary'] = 0
# labels.at[df[df['Temperate (empirical)'] == 'yes'].index, 'binary'] = 1
# %%
tf.keras.utils.set_random_seed(1)

label_tensor = tf.convert_to_tensor(labels)

sequence_tensor = tf.cast(tf.convert_to_tensor(sequence_condensed), dtype='int32')
sequence_tensor_one_hot = tf.one_hot(sequence_tensor, 5, axis=-1)

# %%

# sequence_tensor_one_=hot = tf.raw_ops.RandomShuffle(value=sequence_tensor_one_hot, seed=1, seed2=1)
# label_tensor = tf.raw_ops.RandomShuffle(value=label_tensor, seed=1, seed2=1)
sequence_tensor_one_hot = sequence_tensor_one_hot[..., 0:5]

training_fraction = 0.8
total_size = sequence_tensor_one_hot.shape[0]
training_size = round(total_size * training_fraction)
train_set, test_set = tf.split(sequence_tensor_one_hot, [training_size, total_size - training_size], axis=0)
train_label, test_label = tf.split(label_tensor, [training_size, total_size - training_size], axis=0)

# %%
input_shape = (train_set.shape[1], train_set.shape[2])
inputs = Input(shape=input_shape)

x = Conv1D(filters=128, kernel_size=6, strides=1, activation='relu')(inputs)
x = MaxPool1D(pool_size=3, strides=3)(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.3)(x)

x = GlobalAveragePooling1D()(x)
x = Dense(units=128, activation='relu')(x)
x = BatchNormalization()(x)

pred = Dense(units=1, activation='sigmoid')(x)

model0 = Model(inputs=inputs, outputs=pred)
model0.compile(optimizer=Adam(learning_rate=0.001),
               loss='binary_crossentropy',
               metrics=['accuracy'])
model0.summary()
# %%
model0.fit(train_set, train_label, batch_size=32, epochs=10, shuffle=True)
model0.evaluate(test_set, test_label)
