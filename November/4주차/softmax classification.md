


Tensorflow로 Softmax Classification 구현하기
---
![image](https://user-images.githubusercontent.com/34376342/100608243-75fed500-334f-11eb-8f95-32694bcea871.png)    
하나의 벡터가 주어지고 X와 연산을 하게 된다면 각각의 계산된 값은 독립된 바이너리 예측 값을 획득할 수 있으며, 이것을 Multinomial Logistic Regression이라고 한다.   
위의 계산을 통해 각 2.0, 1.0, 0.1 등으로 나온 결과값을 sigmoid 함수를 이용해 0~1 사이의 값으로 바꿔준다.

![image](https://user-images.githubusercontent.com/34376342/100610167-8e242380-3352-11eb-80a1-2be15a3305d4.png)    
이처럼 2.0을 0.7로, 1.0을 0.2로 바꿔주는 역할이 바로 softmax이다.
0~1 사이로 바꾼 값을 전부 더하면 1이 된다.
즉, 왼쪽의 값(logit)을 확률로 바꾼 것이다.

그렇다면 logit에서 A가 나올 확률이 0.7, B가 나올 확률이 0.2, C가 나올 확률이 0.1이라고 가정할 때 이것을 Binary(0,1)로 선택해서 one-hot encoding하여 값이 참인지 아닌지로 선택할 수 있다.

이것을 텐서플로우를 이용해 아래와 같이 구현한다.
```python
tf.matmul(X,W)+b
hypothesis=tf.nn.softmax(tf.manul(X,W)+b)
```

그리고 예측한 값과 실제 값이 얼마나 차이가 나는지에 대한 Cost Function을 정의해주어야 한다.

여기서 실제값과 softmax 값(예측값)의 차이를 cross-entropy 함수를 통해 구한다.
그리고 최종적으로 경사 하강법을 통해 최적의 값을 구하면 된다.
```python
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
```
cost = Y*log(hypothesis) 값을 합한 뒤 평균을 낸다.

![image](https://user-images.githubusercontent.com/34376342/100611901-68e4e480-3355-11eb-855a-00fa7e5049e3.png) 
![image](https://user-images.githubusercontent.com/34376342/100612185-e4df2c80-3355-11eb-8b2b-999a545bfb4b.png)  
 W에서 4는 입력된 개수, nb_classes는 number의 class(나가는 값, Y)로 weight과 bias를 만들 때 shape를 주의해서 적어야 한다.

![image](https://user-images.githubusercontent.com/34376342/100613017-363beb80-3357-11eb-8d06-03f4b3174953.png)  
![image](https://user-images.githubusercontent.com/34376342/100613146-62f00300-3357-11eb-8db2-462829da6421.png)  
X값으로 [1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]을 각각 넣어서 한번에 출력해보았을 때, 분류가 된 것을 알 수 있다.

```python
!pip install tensorflow==1.14

import tensorflow as tf
tf.set_random_seed(777) # reproducibility
tf.disable_v2_behavior()

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss 계산하기
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 실행
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})

            if step % 200 == 0:
                print(step, cost_val)

    print('--------------')
    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))


```
