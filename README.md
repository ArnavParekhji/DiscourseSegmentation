# DiscourseSegmentation
A discourse segmentation model built using Pointer Networks and ELMO Embeddings in Keras

Load dataset in variables X and Y in model.py.

X is an array of sentences, example: ["first sentence", "second sentence"].

Y is an array of arrays, each corresponding to the index of edu breaks, example: [[3,6], [2,5,7]].


Run pointer.py first which builds the the Pointer Network.
Then run model.py to train the network.
