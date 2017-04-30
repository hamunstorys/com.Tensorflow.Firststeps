import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

#TF_CPP_MIN_LOG_LEVEL is a TensorFlow environment variable responsible for the logs,
# to silence INFO logs set it to 1, to filter out WARNING 2 and to additionally silence ERROR logs (not recommended) set it to 3

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# setting matplotlib display for ubuntu
if(plt.get_backend()!='Tkagg'):
    plt.switch_backend('Tkagg')

num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1]) # vectors_set을 2개의 요소를 가진 벡터로 선언하고, x1을 첫번째 요소에 추가, y1을 두번째 요소에 추가


x_data = [v[0] for v in vectors_set] # vectors_set 의 첫번째 요소를 x_data에 삽입
y_data = [v[1] for v in vectors_set] # vectors_set 의 두번째 요소를 y_data에 삽입

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b # 가설 함수

loss = tf.reduce_mean(tf.square(y-y_data)) # 비용 함수

optimizer = tf.train.GradientDescentOptimizer(0.5) # 0.5 = 학습률(learning rate)
train = optimizer.minimize(loss) # 학습

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(8):
    sess.run(train)
    print(step, sess.run(W), sess.run(b))
    print(step, sess.run(loss))

    #show graph

    # 산포도 그리기
    plt.plot(x_data, y_data, 'ro')
    # 직선 그리기
    plt.plot(x_data,sess.run(W)*x_data+sess.run(b))

    #labeling
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()