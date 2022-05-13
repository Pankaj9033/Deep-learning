import tensorflow as tf
print("Matrix Multiplication")
x = tf.constant([1, 2, 3, 4, 5, 6], shape=[2,3])
print(x)
y = tf.constant([7, 8, 9, 10, 11, 12], shape=[3,2])
print(y)
result = tf.matmul(x,y)
print("Product: ", result)
