# %%

import pandas as pd
from Bio import SearchIO
from collections import OrderedDict
from sklearn.model_selection import train_test_split

# %%

benchmark_file = './Data/phage_data_nmicro2017/processed_benchmark_set.csv'

hmm_results_dir = './Data/phage_data_nmicro2017/hmmsearch_out/'

base_dir = './Data/classifier_data/'

hmm_data_dir = './Data/protein_domain_data/domain_alignments_and_hmms/'

# %%

###Read in the dataset and double check that analysis is limited to empirically defined data
df = pd.read_csv(benchmark_file, index_col=0)
print('Starting shape:', df.shape)
df = df[df['Temperate (empirical)'] != 'Unspecified']
print('New shape (should be identical):', df.shape)
# %%
###Add in my identifier
df['Identifier_AJH'] = ''
df.at[df[df['Database source'] == 'NCBI RefSeq'].index, 'Identifier_AJH'] = \
    df[df['Database source'] == 'NCBI RefSeq']['RefSeq accession number']
df.at[df[df['Database source'] == 'Actinobacteriophage_785'].index, 'Identifier_AJH'] = \
    df[df['Database source'] == 'Actinobacteriophage_785']['Virus identifier used for the analysis'].str.split('_').str[
        0]
print('New shape (+1 new column):', df.shape)

# %%

###Read through all of the hmmsearch files to accumulate a growing presence/absence dataframe
file_ending = '.fasta.hmmsearch'
growing_df = pd.DataFrame()
for index in df.index:
    if df.loc[index]['Database source'] == 'NCBI RefSeq':
        file_name = df.loc[index]['RefSeq accession number'] + file_ending
    elif df.loc[index]['Database source'] == 'Actinobacteriophage_785':
        file_name = df.loc[index]['Virus identifier used for the analysis'].split('_')[0].lower() + \
                    file_ending
    try:
        with open(hmm_results_dir + file_name, 'r') as infile:
            results = list(SearchIO.parse(infile, 'hmmer3-text'))
            simple_res = []
            for i in results:
                if len(i.hits) > 0:
                    simple_res.append((i.id, 1))
                else:
                    simple_res.append((i.id, 0))
        simple_res = sorted(simple_res, key=lambda x: x[0])
        single_df = pd.DataFrame(OrderedDict(simple_res), index=[index])
        growing_df = pd.concat([growing_df, single_df])
    except FileNotFoundError:
        print('Failed to find this file: {}!'.format(file_name))
        pass
print(growing_df.shape)

###Add that to the main dataframe
full_df = df.join(growing_df)

# %%

###Split into training and testing sets
train_df, test_df = train_test_split(full_df, train_size=0.8, random_state=0, shuffle=True)  # 42
print('Shape of training and testing dataframes:', train_df.shape, test_df.shape)

###Set up the machine-learning training and test sets
ml_df_train = train_df[train_df.columns[23:]]
ml_df_test = test_df[test_df.columns[23:]]

###And labels
training_labels = pd.DataFrame(index=train_df.index)
training_labels['binary'] = 0
training_labels.at[train_df[train_df['Temperate (empirical)'] == 'yes'].index, 'binary'] = 1

testing_labels = pd.DataFrame(index=test_df.index)
testing_labels['binary'] = 0
testing_labels.at[test_df[test_df['Temperate (empirical)'] == 'yes'].index, 'binary'] = 1

print('Shape of training and testing labels:', training_labels.shape, testing_labels.shape)

# %%


import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

ml_df_train = ml_df_train.fillna(0)
ml_df_test = ml_df_test.fillna(0)

ml_df_train_tensor = tf.convert_to_tensor(ml_df_train)
ml_df_test_tensor = tf.convert_to_tensor(ml_df_test)

training_labels_tensor = tf.convert_to_tensor(training_labels)
testing_labels_tensor = tf.convert_to_tensor(testing_labels)

# %%
tf.keras.utils.set_random_seed(0)
hp_hidden_size_1 = 16
hp_hidden_size_2 = 16
hp_learning_rate = 0.001
hp_dropout_1 = 0.2
hp_dropout_2 = 0.2
inputs = Input(shape=ml_df_train_tensor.shape[1])
x = Dense(hp_hidden_size_1, activation='relu')(inputs)
x = Dropout(rate=hp_dropout_1)(x)
x = BatchNormalization()(x)
x = Dense(hp_hidden_size_2, activation='relu')(x)
x = Dropout(rate=hp_dropout_2)(x)
x = BatchNormalization()(x)
predictions = Dense(1, activation='sigmoid')(x)
model0 = Model(inputs=inputs, outputs=predictions)
model0.compile(optimizer=Adam(learning_rate=hp_learning_rate),
               loss='binary_crossentropy',
               metrics=['accuracy'])
model0.fit(ml_df_train_tensor, training_labels_tensor, batch_size=32, epochs=10, validation_split=0.2)
model0.evaluate(ml_df_test_tensor, testing_labels_tensor)


# %%

def model(hp):
    tf.keras.utils.set_random_seed(0)
    hp_hidden_size_1 = hp.Int('hidden_size_1', 8, ml_df_train_tensor.shape[1], step=1,
                              default=ml_df_train_tensor.shape[1])
    hp_hidden_size_2 = hp.Int('hidden_size_2', 8, ml_df_train_tensor.shape[1], step=1,
                              default=ml_df_train_tensor.shape[1])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_dropout_1 = hp.Float('dropout_1', 0.025, 0.4, step=0.025, default=0.1)
    hp_dropout_2 = hp.Float('dropout_2', 0.025, 0.4, step=0.025, default=0.1)
    inputs = Input(shape=ml_df_train_tensor.shape[1])
    x = Dense(hp_hidden_size_1, activation='relu')(inputs)
    x = Dropout(rate=hp_dropout_1)(x)
    x = BatchNormalization()(x)
    x = Dense(hp_hidden_size_2, activation='relu')(x)
    x = Dropout(rate=hp_dropout_2)(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# %%
# model.fit(ml_df_train_tensor,training_labels_tensor,epochs=10,batch_size=32)
# model.evaluate(ml_df_test_tensor, testing_labels_tensor)
#

import keras_tuner as kt

tuner = kt.Hyperband(
    model,
    objective='val_accuracy',
    max_epochs=10,
    hyperband_iterations=4)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
tuner.search(ml_df_train_tensor, training_labels_tensor, epochs=20, validation_split=0.2, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(1)[0]

best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(ml_df_train_tensor, training_labels_tensor, epochs=50, validation_split=0.2)
best_model.summary()

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

best_model.evaluate(ml_df_test_tensor, testing_labels_tensor)
