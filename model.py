from keras.models import Model
from keras.layers import LSTM, Input
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical
from pointer import PointerLSTM
import pickle
# import tsp_data as tsp
import numpy as np
import keras

from allennlp.modules.elmo import Elmo, batch_to_ids

weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
elmo = Elmo(options_file, weight_file, 2, dropout=0.5,requires_grad=False)
# elmo = elmo.cuda()

def scheduler(epoch):
    if epoch < nb_epochs/4:
        return learning_rate
    elif epoch < nb_epochs/2:
        return learning_rate*0.5
    return learning_rate*0.1

# feed data into X and Y
X, Y = [], []
x_test, y_test = [], []

YY = []
for y in Y:
    YY.append(to_categorical(y))
YY = np.asarray(YY)

batch_size = len(X)
character_ids = batch_to_ids(X)
embeddings = elmo(character_ids)
X_elmo = embeddings["elmo_representations"][0]
X = X_elmo

hidden_size = 64
seq_len = 10
nb_epochs = 1000
learning_rate = 0.1

print("building model...")
main_input = Input(shape=(seq_len, 2), name='main_input')

encoder, state_h, state_c = LSTM(hidden_size, return_sequences = True, name="encoder",return_state=True)(main_input)
decoder = PointerLSTM(hidden_size, name="decoder")(encoder, states=[state_h, state_c])

model = Model(main_input, decoder)
print(model.summary())
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, YY, epochs=nb_epochs, batch_size=64,)
print(model.predict(x_test))
print('evaluate : ',model.evaluate(x_test,to_categorical(y_test)))
print("------")
print(to_categorical(y_test))
model.save_weights('model_weight_100.hdf5')