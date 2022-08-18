# Multilingual DALLE

# Language support
Due to my lack of computing power, the only language currently pre-trained is french at 12k pairs for 100k iterations. Language data is available at: https://www.manythings.org/anki/ and https://www.statmt.org/europarl/. Format it like eng-fra.txt, and modify the prepareData() function call. 

# How does the translator work?
During training, sequenece to sequence learning is used by making a Pytorch deep neural network  which predicts the likelihood of a sequence of words as a correct translation by using a decoder and encoder. This is effective because when translating sentences, the result as a string would be a different length than the input. After enough iterations, the model can infer the grammar structure of the output language, one of the most difficult NMT tasks. 

# About training:
I advise you to only train this on a desktop with CUDA. You will need to train two models, with lots of data+iterations needed to create a fesible language model. CPU training will be too slow and will impact your battery and cpu health greatly.

