import tensorflow as tf
e_matrix_A = tf.random.uniform([2, 2], minval=3, maxval=10, dtype=tf.float32,
name="matrixA")
print("Matrix A: \n{}\n\n".format(e_matrix_A))
eigen_values_A, eigen_vectors_A = tf.linalg.eigh(e_matrix_A)
print("Eigen Vectors: \n{} \n\nEigen Values: \n{}\n".format(eigen_vectors_A,
eigen_values_A))
