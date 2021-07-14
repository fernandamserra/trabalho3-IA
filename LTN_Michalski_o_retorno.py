import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import logictensornetworks as ltn

tf.config.run_functions_eagerly(True)

# path to trains-transformed.csv
path = './trains-transformed.csv'

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
embedding_size = 10

df = df.sample(frac=1) #shuffle
df_train = df[0:5]
df_test = df[5:]

labels_train = df_train.pop("Class_attribute").astype(int)
labels_test = df_test.pop("Class_attribute").astype(int)

ds_train = tf.data.Dataset.from_tensor_slices((df_train,labels_train)).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((df_test,labels_test)).batch(batch_size)

df.pop('Class_attribute')

xyz = 10
valdiur = []
for a in range(10):
    l = []
    for d in range(int(df.iloc[a]['Number_of_cars'])-1):
        l.append(xyz)
        xyz += 1
    valdiur.append(l)

HasCar          = ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8))
CarCount        = ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8))
NoOfDiffLoads   = ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8))
NoWheels        = ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8))
Length          = ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8))
Shape           = ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8))
NoLoads         = ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8))
LoadShape       = ltn.Predicate.MLP([embedding_size, embedding_size],hidden_layer_sizes=(8,8))

ini = xyz-3
def qtdcar(x):
    return x + ini

def qtdcarga(x):
    return x + ini + 5

def qtdrod(x):
    return x + ini + 8

def qtdlen(x):
    return x + ini + 12

def qtdshape(x):
    return x + ini + 14

def qtdcargacarr(x):
    return x + ini + 22

def qtdloadshape(x):
    return x + ini + 25

g = {l:ltn.constant(np.random.uniform(low=0.0,high=1.0,size=embedding_size),trainable=True) for l in range(ini+35)}

formula_aggregator = ltn.fuzzy_ops.Aggreg_pMeanError()

class MLP(tf.keras.Model):
    """Model that returns logits."""
    def __init__(self, n_classes, hidden_layer_sizes=(16,16,8)):
        super(MLP, self).__init__()
        self.denses = [tf.keras.layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_class = tf.keras.layers.Dense(n_classes)
        self.dropout = tf.keras.layers.Dropout(0.2)
        
    def call(self, inputs, training=False):
        x = inputs
        for dense in self.denses:
            x = dense(x)
            x = self.dropout(x, training=training)
        return self.dense_class(x)

logits_model = MLP(32)
p = ltn.Predicate(ltn.utils.LogitsToPredicateModel(logits_model,single_label=True))

Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
class_Oeste = ltn.constant(0)
class_Leste = ltn.constant(1)

@tf.function
def axioms(features, labels, training = False):
    x_Oeste = ltn.variable("x_Oeste",features[labels==0])
    x_Leste = ltn.variable("x_Leste",features[labels==1]) 
    
    axioms = [
        Forall(x_Oeste,p([x_Oeste,class_Oeste],training=training)),
        Forall(x_Leste,p([x_Leste,class_Leste],training=training))
    ]

    for x in range(10):
        axioms.append(formula_aggregator(tf.stack([HasCar([g[x], g[y]]) for y in valdiur[x]])))
    
    axioms.append(formula_aggregator(tf.stack([CarCount([g[x],g[qtdcar(df.iloc[x]['Number_of_cars'])]]) for x in range(10)])))
    axioms.append(formula_aggregator(tf.stack([NoOfDiffLoads([g[x], g[qtdcargacarr(df.iloc[x]['Number_of_different_loads'])]]) for x in range(10)])))
    for i in  range(10):
        for index, j in enumerate(valdiur[i]):
            axioms.append(Length([g[j], g[qtdlen(df.iloc[i]['length'+str(index+1)])]]))
            axioms.append(Shape([g[j], g[qtdshape(df.iloc[i]['shape'+str(index+1)])]]))
            axioms.append(NoLoads([g[j], g[qtdcarga(df.iloc[i]['num_loads'+str(index+1)])]]))
            axioms.append(NoWheels([g[j], g[qtdrod(df.iloc[i]['num_wheels'+str(index+1)])]]))
            axioms.append(LoadShape([g[j], g[qtdloadshape(df.iloc[i]['load_shape'+str(index+1)])]]))
    
    # pain and no gain

    axioms = tf.stack([tf.squeeze(ax) for ax in axioms])
    sat_level = formula_aggregator(axioms)
    return sat_level, axioms

for features, labels in ds_test:
    print("Initial sat level %.5f"%axioms(features, labels)[0])
    break

import functools

'''
trainable_variables = \
        CarCount.trainable_variables \
        + NoOfDiffLoads.trainable_variables \
        + functools.reduce(lambda a,b: a+b, map(lambda a: a.trainable_variables, NoWheels)) \
        + functools.reduce(lambda a,b: a+b, map(lambda a: a.trainable_variables, Length)) \
        + functools.reduce(lambda a,b: a+b, map(lambda a: a.trainable_variables, Shape)) \
        + functools.reduce(lambda a,b: a+b, map(lambda a: a.trainable_variables, NoLoads)) \
        + functools.reduce(lambda a,b: a+b, map(lambda a: a.trainable_variables, LoadShape)) \
        + list(g.values())

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for epoch in range(2000):
    with tf.GradientTape() as tape:
        loss_value = 1. - axioms()[0]
    grads = tape.gradient(loss_value, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    if epoch%200 == 0:
        print("Epoch %d: Sat Level %.3f"%(epoch, axioms()[0]))

print("Training finished at Epoch %d with Sat Level %.3f"%(epoch, axioms()[0]))
'''



metrics_dict = {
    'train_sat_kb': tf.keras.metrics.Mean(name='train_sat_kb'),
    'test_sat_kb': tf.keras.metrics.Mean(name='test_sat_kb'),
    'train_accuracy': tf.keras.metrics.CategoricalAccuracy(name="train_accuracy"),
    'test_accuracy': tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
@tf.function
def train_step(features, labels):
    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(features, labels, training=True)[0]
        loss = 1.-sat
    gradients = tape.gradient(loss, p.trainable_variables)
    optimizer.apply_gradients(zip(gradients, p.trainable_variables))
    sat = axioms(features, labels)[0] # compute sat without dropout
    metrics_dict['train_sat_kb'](sat)
    # accuracy
    predictions = logits_model(features)
    metrics_dict['train_accuracy'](tf.one_hot(labels,2),predictions)
    
@tf.function
def test_step(features, labels):
    # sat
    sat = axioms(features, labels)[0]
    metrics_dict['test_sat_kb'](sat)
    # accuracy
    predictions = logits_model(features)
    metrics_dict['test_accuracy'](tf.one_hot(labels,2),predictions)

import commons

from collections import defaultdict

EPOCHS = 100

commons.train(
    EPOCHS,
    metrics_dict,
    ds_train,
    ds_test,
    train_step,
    test_step,
    csv_path="train_results.csv",
    track_metrics=25,
)
