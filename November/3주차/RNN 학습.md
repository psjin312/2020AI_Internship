# RNN 학습하기
**RNN**이란?
Recurrent Neural Network, 즉 순환 신경망이라고 부른다.
기존의 Neural Network(Vanilla NN. Vanilla : 가장 기초, 기본적인)로는 Sequence Data(연속적인 데이터)를 처리할 수 없어서 새로운 network Model을 만든 것이 바로 RNN이다.

여기서 Sequence Data란 예를 들어 사람의 speech, 영화, 음악 등등이 있다. 사람의 말은 문장으로 구성되어 있어 한 단어만 듣고는 말의 뜻이나 의도를 알 수 없고 어느 정도의 문장(=시간의 흐름, 앞 단어와의 관계를 통해 현재 단어의 의미 해석)을 들은 후에야 알 수가 있다. 이러한 시간의 흐름에서 발생하는 데이터를  Sequence Data라고 한다.

![image](https://user-images.githubusercontent.com/34376342/99986306-5cf7a080-2df2-11eb-9799-8518ab932f6d.png) 
RNN을 그림으로 표현해보면 다음과 같다.
여기서 X는 입력, H는 출력이고 A는 은닉층(Hidden layer)이다. 은닉층의 출력이 다시 다음 은닉층의 입력으로 들어가기 때문에(반복, Recurrent) 그림이 저렇게 구성된다.

이를 펼쳐보면

![image](https://user-images.githubusercontent.com/34376342/99985792-bf9c6c80-2df1-11eb-84f7-bd7db9a65cf2.png)

다음과 같다. 즉, 각 네트워크는 다음 단계로 정보를 넘겨주고 또 넘겨주는 형태이다.











```python
# Lab 12 RNN

!pip install tensorflow==1.14

  

# import tensorflow.compat.v1 as tf

import tensorflow as tf

import numpy as np

tf.set_random_seed(777) # reproducibility

# tf.random.set_seed(777)

#tf.compat.v1.placeholder()

tf.disable_v2_behavior()

  

idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hello: hihell -> ihello

x_data = [[0, 1, 0, 2, 3, 3]] # hihell

x_one_hot = [[[1, 0, 0, 0, 0], # h 0

[0, 1, 0, 0, 0], # i 1

[1, 0, 0, 0, 0], # h 0

[0, 0, 1, 0, 0], # e 2

[0, 0, 0, 1, 0], # l 3

[0, 0, 0, 1, 0]]] # l 3

  

y_data = [[1, 0, 2, 3, 3, 4]] # ihello

  

num_classes = 5

input_dim = 5  # one-hot size

hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot

batch_size = 1  # one sentence

sequence_length = 6  # |ihello| == 6

learning_rate = 0.1

  

X = tf.placeholder(

tf.float32, [None, sequence_length, input_dim]) # X one-hot

Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

  

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)

outputs, _states = tf.nn.dynamic_rnn(

cell, X, initial_state=initial_state, dtype=tf.float32)

  

# FC layer

X_for_fc = tf.reshape(outputs, [-1, hidden_size])

# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])

# fc_b = tf.get_variable("fc_b", [num_classes])

# outputs = tf.matmul(X_for_fc, fc_w) + fc_b

outputs = tf.contrib.layers.fully_connected(

inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

  

# reshape out for sequence_loss

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

  

weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(

logits=outputs, targets=Y, weights=weights)

loss = tf.reduce_mean(sequence_loss)

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

  

prediction = tf.argmax(outputs, axis=2)

  

with tf.Session() as sess:

sess.run(tf.global_variables_initializer())

for i in  range(50):

l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})

result = sess.run(prediction, feed_dict={X: x_one_hot})

print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

  

# print char using dic

result_str = [idx2char[c] for c in np.squeeze(result)]

print("\tPrediction str: ", ''.join(result_str))






```
