# RNN 학습하기
**RNN**이란?
Recurrent Neural Network, 즉 순환 신경망이라고 부른다.
기존의 Neural Network(Vanilla NN. Vanilla : 가장 기초, 기본적인)로는 Sequence Data(연속적인 데이터)를 처리할 수 없어서 새로운 network Model을 만든 것이 바로 RNN이다.

여기서 Sequence Data란 예를 들어 사람의 speech, 영화, 음악 등등이 있다. 사람의 말은 문장으로 구성되어 있어 한 단어만 듣고는 말의 뜻이나 의도를 알 수 없고 어느 정도의 문장(=시간의 흐름, 앞 단어와의 관계를 통해 현재 단어의 의미 해석)을 들은 후에야 알 수가 있다. 이러한 시간의 흐름에서 발생하는 데이터를  Sequence Data라고 한다.

![image](https://user-images.githubusercontent.com/34376342/99986306-5cf7a080-2df2-11eb-9799-8518ab932f6d.png)   
RNN을 그림으로 표현해보면 다음과 같다.
여기서 X는 입력, H는 출력(Hidden size)이고 A는 은닉층(Hidden layer)이다. 은닉층의 출력이 다시 다음 은닉층의 입력으로 들어가기 때문에(반복, Recurrent) 그림이 저렇게 구성된다.

이를 펼쳐보면

![image](https://user-images.githubusercontent.com/34376342/99985792-bf9c6c80-2df1-11eb-84f7-bd7db9a65cf2.png)  

다음과 같다. 즉, 각 네트워크는 다음 단계로 정보를 넘겨주고 또 넘겨주는 형태이다.


'hihello'를 학습 시킨 과정
---
![image](https://user-images.githubusercontent.com/34376342/100496177-6f2c6280-3195-11eb-8170-94a1d657a65e.png)  
tensorflow가 1.x 버전에서만 실행 가능했기에 spyder, pycharm에서 1.14 버전을 설치하여 실행하였지만 자꾸만 해결되지 않는 오류가 발생하여 colab에서 설치하여 진행하였다.   
tensorflow 1.x 버전에서는 random.set_seed가 아닌 set_random_seed로 문법도 조금씩 상이하였다.
또한, 2.x 버전에서 발생하는 오류를 없애기 위해 tf.disable_v2_behavior() 를 입력해주었다.

   
![image](https://user-images.githubusercontent.com/34376342/100496848-dd275880-319a-11eb-8a70-c9df45f3f79e.png)  
'hihello'는 각 h, i, e, l, o를 가지고 쓰인다.
h가 입력되었을 때 i가 나와야 할 수도 있고
e가 나올 수도 있는데, 이는 이전에 무슨 문자가 나왔는지 알아야 어떤 출력이 올바른 것인지 알 수 있기 때문에 RNN이 이 학습에 효과적이다. 

입력값 x_one_hot은 x_data를 one-hot으로 치환한 결과이다.

![image](https://user-images.githubusercontent.com/34376342/100496565-71dc8700-3198-11eb-8062-4181783c2fda.png)  
input dimension = 5
h를 예를 들면 [1, 0, 0, 0, 0]으로 5개이므로 dimension이 5 이다.

 hidden size = 5
 input dimension(one-hot size) 그대로 출력해야 함
 예) h [1, 0, 0, 0, 0] 입력 -> i [0, 1, 0, 0, 0] 이 나와야 함

![image](https://user-images.githubusercontent.com/34376342/100496929-77879c00-319b-11eb-9c93-617da76ee374.png)  
X는 one-hot인데 sequence_length는 6(h,i,h,e,l,l), input_dim은 5 이다.  None은 원래 batchsize가 1이므로 1을 주어야 하는데 더 많아도 상관 없다는 뜻으로 None을 준다.

![image](https://user-images.githubusercontent.com/34376342/100498408-4ca24580-31a5-11eb-8f09-86d024c43d8c.png)  
tensorflow에서는 sequence_loss라는 함수가 있는데, sequence data를 이 함수를 이용해 간단하게 계산할 수 있다. 
sequence를 받아들이는데 logits(예측), target(true data, sequence로 줌), weights(각각의 자리를 얼마나 중요하게 여기는지, 여기선 1로 줌)

loss를 평균낸 후, 그 값을 AdamOptimizer에 minimize시키면 학습이 이루어진다.

![image](https://user-images.githubusercontent.com/34376342/100581027-53a69080-332a-11eb-8950-d4e7e4362015.png)  
학습 과정이다.
우선 Session()으로 Session을 열고 intialize 한다.
그리고 우리가 아는 X 데이터, Y 데이터를 넘겨주면서 train 시켜준다.

prediction에는 one-hot으로 나온 것 중 값을 골라주기 위해 tf.argmax()를 사용한다.

result_str에서 숫자로 나온 prediction을 string으로 값을 뽑아 출력해본다. 

그 결과

![image](https://user-images.githubusercontent.com/34376342/100582686-f52ee180-332c-11eb-9c05-4eb67951ac7d.png)  
![image](https://user-images.githubusercontent.com/34376342/100582753-1394dd00-332d-11eb-9de1-f6129db847f8.png)  
50번 반복하도록 했을 때, 점점 loss 값이 낮아지고, 원하는 값인 ihello가 나온 것을 볼 수 있다.

---
```python
# RNN hihello

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

x_data = [[0, 1, 0, 2, 3, 3]] # dictionary, hihell

x_one_hot = [[[1, 0, 0, 0, 0], # h : 0

[0, 1, 0, 0, 0], # i : 1

[1, 0, 0, 0, 0], # h : 0

[0, 0, 1, 0, 0], # e : 2

[0, 0, 0, 1, 0], # l : 3

[0, 0, 0, 1, 0]]] # l : 3

  

y_data = [[1, 0, 2, 3, 3, 4]] # ihello

  

num_classes = 5

input_dim = 5  # one-hot size

hidden_size = 5  # one-hot size 대로 출력해야 함

batch_size = 1  # hihell 한 문장

sequence_length = 6  # |ihello| == 6

learning_rate = 0.1

  

X = tf.placeholder(

tf.float32, [None, sequence_length, input_dim]) # X : one-hot

Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y : label

  

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

  

# sequence_loss

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

