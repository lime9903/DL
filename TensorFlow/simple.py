import tensorflow as tf   # 명령 프롬프트로 pip install tensorflow 

# 상수 노드 정의
a = tf.constant(1.0, name='a')
b = tf.constant(2.0, name='b')
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print(a)
print(a+b)
print(c)

sess = tf.Session()

print(sess.run([a, b]))

sess.close()