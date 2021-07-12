import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import logictensornetworks as ltn

tf.config.run_functions_eagerly(True)

# path to trains-transformed.csv
path = './train.csv'

str_att = {
  'length': ['short', 'long'],
  'shape': ['closedrect', 'dblopnrect', 'ellipse', 'engine', 'hexagon',
          'jaggedtop', 'openrect', 'opentrap', 'slopetop', 'ushaped'],
  'load_shape': ['circlelod', 'hexagonlod', 'rectanglod', 'trianglod'],
  'Class_attribute': ['west','east']
}

def read_data(path=path):
  df = pd.read_csv(path, ',')
  for k in df:
    for att in str_att:
      if k.startswith(att):
        for i,val in enumerate(df[k]):
          if val in str_att[att]:
            df[k][i] = str_att[att].index(val)

  df.replace("\\0", 0, inplace=True)
  df.replace("None", -1, inplace=True)

  return df

df = read_data()
df = df.astype(float)
df

batch_size = 64
embedding_size = 5

df = df.sample(frac=1) #shuffle
df_train = df[0:5]
df_test = df[5:]

labels_train = df_train.pop("Class_attribute")
labels_test = df_test.pop("Class_attribute")

ds_train = tf.data.Dataset.from_tensor_slices((df_train,labels_train)).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((df_test,labels_test)).batch(batch_size)

df.pop('Class_attribute')
g1 = {index+30:ltn.constant(np.random.uniform(low=0.0, high=1.0, size=embedding_size), trainable=True) for index, l in df_train.iterrows()}
g2 = {l:ltn.constant(np.random.uniform(low=0.0,high=1.0,size=embedding_size),trainable=True) for l in range(8)}
g = {**g1,**g2}

CarCount = ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8))
NoOfDiffLoads = ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8))
NoWheels = [ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8)) for i in range(4)]
Length = [ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8)) for i in range(4)]
Shape = [ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8)) for i in range(4)]
NoLoads = [ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8)) for i in range(4)]
LoadShape = [ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8)) for i in range(4)]

nCarros = [[index+30 for index, train in df_train.iterrows() if train["Number_of_cars"] == i] for i in range(6)]

loadCount = [[index+30 for index, train in df_train.iterrows() if train["Number_of_different_loads"] == i] for i in range(5)]

wheelCount = [[[index+30 for index, train in df_train.iterrows() if train["num_wheels"+str(j)] == i] for i in range(5)] for j in range(1,5)]
length = [[[index+30 for index, train in df_train.iterrows() if train["length"+str(j)] == i] for i in range(2)] for j in range(1,5)]
shape = [[[index+30 for index, train in df_train.iterrows() if train["shape"+str(j)] == i] for i in range(8)] for j in range(1,5)]
loadInCar = [[[index+30 for index, train in df_train.iterrows() if train["num_loads"+str(j)] == i] for i in range(3)] for j in range(1,5)]
loadShape = [[[index+30 for index, train in df_train.iterrows() if train["load_shape"+str(j)] == i] for i in range(7)] for j in range(1,5)]

formula_aggregator = ltn.fuzzy_ops.Aggreg_pMeanError(p=2)

@tf.function
def axioms():
    axioms = []
    for i in range(6):
        axioms.append(formula_aggregator(tf.stack(
          [CarCount([g[j],g[i]]) for j in nCarros[i]])))
      
    for i in range(5):
      axioms.append(formula_aggregator(tf.stack(
          [NoOfDiffLoads([g[j], g[i]]) for j in loadCount[i]])))
    
    for i in range(4):
      for j in range(5):
        axioms.append(formula_aggregator(tf.stack(
            [NoWheels[i]([g[t], g[j]]) for t in wheelCount[i][j]])))
        
    for i in range(4):
      for j in range(2):
        axioms.append(formula_aggregator(tf.stack(
            [Length[i]([g[t], g[j]]) for t in length[i][j]])))
        
    for i in range(4):
      for j in range(8):
        axioms.append(formula_aggregator(tf.stack(
            [Shape[i]([g[t], g[j]]) for t in shape[i][j]])))
        
    for i in range(4):
      for j in range(3):
        axioms.append(formula_aggregator(tf.stack(
            [NoLoads[i]([g[t], g[j]]) for t in loadInCar[i][j]])))
        
    for i in range(4):
      for j in range(7):
        axioms.append(formula_aggregator(tf.stack(
            [LoadShape[i]([g[t], g[j]]) for t in loadShape[i][j]])))

    axioms = tf.stack(axioms)
    sat_level = formula_aggregator(axioms)
    return sat_level, axioms

for features, labels in ds_test:
    print("Initial sat level %.5f"%axioms()[0])
    break

import functools

trainable_variables = \
        CarCount.trainable_variables \
        + NoOfDiffLoads.trainable_variables \
        + functools.reduce(lambda a,b: a+b, map(lambda a: a.trainable_variables, NoWheels)) \
        + functools.reduce(lambda a,b: a+b, map(lambda a: a.trainable_variables, Length)) \
        + functools.reduce(lambda a,b: a+b, map(lambda a: a.trainable_variables, Shape)) \
        + functools.reduce(lambda a,b: a+b, map(lambda a: a.trainable_variables, NoLoads)) \
        + functools.reduce(lambda a,b: a+b, map(lambda a: a.trainable_variables, LoadShape))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for epoch in range(2000):
    with tf.GradientTape() as tape:
        loss_value = 1. - axioms()[0]
    grads = tape.gradient(loss_value, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    if epoch%200 == 0:
        print("Epoch %d: Sat Level %.3f"%(epoch, axioms()[0]))

print("Training finished at Epoch %d with Sat Level %.3f"%(epoch, axioms()[0]))
