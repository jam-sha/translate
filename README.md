# Multilingual DALLE

# Language support
Due to my lack of computing power, only french trained at 12k pairs for 100k iterations is available. Language data is available at: https://www.manythings.org/anki/ and https://www.statmt.org/europarl/ 

# How does the translator work?
A deep neural network is trained to predict the likelihood of a sequence of words as a correct translation. The technique used is sequence to sequence modelling, which means we have a decoder and encoder during training. This is effective because when translating sentences, the result as a string would be a different length than the input. After enough iterations, the model can infer the sentence structure of the output language. 

# About training:
I advise you to only train this on a desktop with CUDA. You will need to train two models, with lots of data+iterations needed to create a fesible language model. CPU training will be too slow and will impact your battery and cpu health greatly.

