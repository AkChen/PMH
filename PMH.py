# 数据排列:modality1 是image,modality2是text
# 前c个是共同的
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from map import MAP

LAMMDA = 0.05  # (0.001,0.1)
ALPHA = 50  # (2,100)
GAMMA = 0.01
IMAGE_DIM = 150
TEXT_DIM = 500
BIT = 16
LEARNING_RATE = 0.000000005
HASH_FUNC_LEARNING_RATE = 0.00001
HASH_FUNC_EPOCH = 20000
EPOCH = 20000
HASH_EPOCH = 2000

WORK_DIR = 'F:/mirflickr/mir_for_CRAE/'
FEATURE_DIR = WORK_DIR + 'feature/'
LIST_DIR = WORK_DIR + 'list/'


def load_data(step):
    partial_data = np.load(FEATURE_DIR + step + '_partial_data.npy')
    image_list = np.load(LIST_DIR + step + '_image_list.npy')
    text_list = np.load(LIST_DIR + step + '_text_list.npy')
    labels = np.load(LIST_DIR + step + '_label.npy')

    image_feature = []
    text_feature = []
    image_label = []
    text_label = []
    n_common = 0
    # common
    for i in range(len(image_list)):
        img_l = image_list[i]
        txt_l = text_list[i]
        if not (img_l.endswith('_p') or txt_l.endswith('_p')):
            image_feature.append(partial_data[i][0:IMAGE_DIM])
            text_feature.append(partial_data[i][IMAGE_DIM:])
            image_label.append(labels[i])
            text_label.append(labels[i])
            n_common += 1

    # partial
    for i in range(len(image_list)):
        img_l = image_list[i]
        txt_l = text_list[i]
        if img_l.endswith('_p'):
            text_feature.append(partial_data[i][IMAGE_DIM:])
            text_label.append(labels[i])
        if txt_l.endswith('_p'):
            image_feature.append(partial_data[i][0:IMAGE_DIM])
            image_label.append(labels[i])

    image_common = image_feature[:n_common]
    image_partial = image_feature[n_common:]
    text_common = text_feature[:n_common]
    text_partial = text_feature[n_common:]

    return np.asarray(image_feature), np.asarray(image_common), np.asarray(image_partial), np.asarray(
        image_label), np.asarray(
        text_feature), np.asarray(text_common), np.asarray(text_partial), np.asarray(text_label),


train_image_feature, train_image_common_feature, train_image_partial_feature, train_image_label, train_text_feature, train_text_common_feature, train_text_partial_feature, train_text_label = load_data(
    'train')
test_image_feature, test_image_common_feature, test_image_partial_feature, test_image_label, test_text_feature, test_text_common_feature, test_text_partial_feature, test_text_label = load_data(
    'test')
val_image_feature, val_image_common_feature, val_image_partial_feature, val_image_label, val_text_feature, val_text_common_feature, val_text_partial_feature, val_text_label = load_data(
    'val')

S1 = squareform(pdist(train_image_feature, metric='euclidean'))
S2 = squareform(pdist(train_text_feature, metric='euclidean'))

D1 = np.zeros(S1.shape)
for i in range(S1.shape[0]):
    sum = 0.
    for j in range(S1.shape[1]):
        sum += S1[i][j]
    D1[i][i] = sum
D2 = np.zeros(S2.shape)
for i in range(S2.shape[0]):
    sum = 0.
    for j in range(S2.shape[1]):
        sum += S2[i][j]
    D2[i][i] = sum

L1 = D1 - S1  # Laplacian matrix
L2 = D2 - S2
print(L1)

# print(image_S.shape)
C = train_image_common_feature.shape[0]  # common count
P1 = train_image_partial_feature.shape[0]
P2 = train_text_partial_feature.shape[0]
L1_c = L1[0:C, 0:C]
L1_p = L1[C:, C:]
L2_c = L2[0:C, 0:C]
L2_p = L2[C:, C:]

X1 = tf.placeholder(tf.float32, (None, IMAGE_DIM), name='image_input')
X1_c = tf.placeholder(tf.float32, (None, IMAGE_DIM), name='image_common_input')
X1_p = tf.placeholder(tf.float32, (None, IMAGE_DIM), name='image_partial_input')

V_c = tf.Variable(tf.random_normal([C, BIT]), name='common_V')

X2 = tf.placeholder(tf.float32, (None, TEXT_DIM), name='text_input')
X2_c = tf.placeholder(tf.float32, (None, TEXT_DIM), name='text_common_input')
X2_p = tf.placeholder(tf.float32, (None, TEXT_DIM), name='text_partial_input')

V1_p = tf.Variable(tf.random_normal([P1, BIT]), name='image_V1')
V2_p = tf.Variable(tf.random_normal([P2, BIT]), name='text_V1')

B1 = tf.Variable(tf.random_normal([BIT, IMAGE_DIM]), name='image_bias')
B2 = tf.Variable(tf.random_normal([BIT, TEXT_DIM]), name='text_bias')

L1_c_tf = tf.placeholder(tf.float32, L1_c.shape, name='L1_c_input')
L1_p_tf = tf.placeholder(tf.float32, L1_p.shape, name='L1_p_input')
L2_c_tf = tf.placeholder(tf.float32, L2_c.shape, name='L2_c_input')
L2_p_tf = tf.placeholder(tf.float32, L2_p.shape, name='L2_p_input')

TRAIN_loss = tf.nn.l2_loss(tf.subtract(X1_p, tf.matmul(V1_p, B1))) + \
             tf.nn.l2_loss(tf.subtract(X2_p, tf.matmul(V2_p, B2))) + \
             ALPHA * tf.trace(tf.matmul(tf.matmul(V1_p, L1_p_tf, transpose_a=True), V1_p)) + \
             ALPHA * tf.trace(tf.matmul(tf.matmul(V2_p, L2_p_tf, transpose_a=True), V2_p)) + \
             LAMMDA * tf.nn.l2_loss(V1_p) + LAMMDA * tf.nn.l2_loss(V2_p) + \
             tf.nn.l2_loss(tf.subtract(X1_c, tf.matmul(V_c, B1))) + \
             tf.nn.l2_loss(tf.subtract(X2_c, tf.matmul(V_c, B2))) + \
             ALPHA * tf.trace(tf.matmul(tf.matmul(V_c, tf.add(L1_c_tf, L2_c_tf), transpose_a=True), V_c)) + \
             LAMMDA * tf.nn.l2_loss(V_c) + LAMMDA * tf.nn.l2_loss(B1) + LAMMDA * tf.nn.l2_loss(B2)

V_list = [V_c, V1_p, V2_p]
B_list = [B1, B2]

global_step = tf.Variable(0, trainable=False)
lr_step = tf.train.exponential_decay(LEARNING_RATE, global_step, EPOCH, 0.99, staircase=True)
opt = tf.train.GradientDescentOptimizer(lr_step)
update_V = opt.minimize(TRAIN_loss, var_list=V_list)
update_B = opt.minimize(TRAIN_loss, var_list=B_list)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

V_feed_dict = {X1_p: train_image_partial_feature, X2_p: train_text_partial_feature, X1_c: train_image_common_feature,
               X2_c: train_text_common_feature, L1_p_tf: L1_p, L2_p_tf: L2_p, L1_c_tf: L1_c, L2_c_tf: L2_c}
B_feed_dict = {X1_p: train_image_partial_feature, X2_p: train_text_partial_feature, X1_c: train_image_common_feature,
               X2_c: train_text_common_feature}

for e in range(EPOCH):
    _ = sess.run(update_V, feed_dict=V_feed_dict)
    _ = sess.run(update_B, feed_dict=B_feed_dict)
    if e % 200 == 0:
        loss = sess.run(TRAIN_loss, feed_dict=V_feed_dict)
        print('epoch:%d loss%f' % (e, loss))

vc_1 = list(sess.run(V_c))
vc_2 = list(sess.run(V_c))
v1 = list(sess.run(V1_p))
v2 = list(sess.run(V2_p))

vc_1.extend(v1)
vc_1.extend(v2)

V = np.asarray(vc_1)

V_tf = tf.placeholder(tf.float32, [None, BIT], 'V_input')

Q = np.zeros((BIT, BIT), float)
for i in range(BIT):
    Q[i][i] = 1.0

Q_tf = tf.placeholder(tf.float32, [BIT, BIT], 'Q_input')

Y_tf = tf.sign(tf.matmul(V_tf, Q_tf))

SVD_S, SVD_U, SVD_V = tf.svd(tf.matmul(Y_tf, V_tf, transpose_a=True))

new_Q = tf.matmul(SVD_V, SVD_U, transpose_a=True, transpose_b=True)

hash_loss = tf.nn.l2_loss(Y_tf - tf.matmul(V_tf, Q_tf))

for e in range(HASH_EPOCH):
    loss = sess.run(hash_loss, feed_dict={V_tf: V, Q_tf: Q})
    Y = np.asarray(sess.run(Y_tf, feed_dict={V_tf: V, Q_tf: Q}))
    Q = np.asarray(sess.run(new_Q, feed_dict={Y_tf: Y, V_tf: V}))

# 计算hash
common_hash = Y[:train_image_common_feature.shape[0]]
image_partial_hash = Y[train_image_common_feature.shape[0]:train_image_common_feature.shape[0] +
                                                           train_image_partial_feature.shape[0]]
text_partial_hash = Y[train_image_common_feature.shape[0] + train_image_partial_feature.shape[0]:]

image_hash = list(common_hash)
image_hash.extend(image_partial_hash)
text_hash = list(common_hash)
text_hash.extend(text_partial_hash)

# lear hash function
image_hash_func = tf.Variable(tf.random_normal([IMAGE_DIM, BIT]), name='image_hash_func')
text_hash_func = tf.Variable(tf.random_normal([TEXT_DIM, BIT]), name='image_hash_func')
image_hash_input = tf.placeholder(tf.float32, [None, BIT])
text_hash_input = tf.placeholder(tf.float32, [None, BIT])

hash_func_learn_loss = tf.nn.l2_loss(tf.matmul(X1, image_hash_func) - image_hash_input) + \
                       tf.nn.l2_loss(tf.matmul(X2, text_hash_func) - text_hash_input) + \
                       GAMMA * tf.nn.l2_loss(image_hash_func) + GAMMA * tf.nn.l2_loss(text_hash_func)
global_step_2 = tf.Variable(0, trainable=False)
lr_step_2 = tf.train.exponential_decay(HASH_FUNC_LEARNING_RATE, global_step_2, HASH_FUNC_EPOCH, 0.99, staircase=True)
opt2 = tf.train.GradientDescentOptimizer(lr_step_2)
update = opt2.minimize(hash_func_learn_loss)

init = tf.global_variables_initializer()
sess.run(init)

image_hash_map = tf.sign(tf.matmul(X1,image_hash_func))
text_hash_map = tf.sign(tf.matmul(X2,text_hash_func))

learn_hash_feed_dict = {X1:train_image_feature,X2:train_text_feature,image_hash_input:image_hash,text_hash_input:text_hash}

for e in range(HASH_FUNC_EPOCH):
    _ = sess.run(update, feed_dict=learn_hash_feed_dict)

    if e % 200 == 0:
        loss = sess.run(hash_func_learn_loss, feed_dict=learn_hash_feed_dict)
        print('epch: %d learn hash loss:%f' % (e,loss))

test_image_hash = sess.run(image_hash_map,feed_dict={X1:test_image_feature})
test_text_hash = sess.run(text_hash_map,feed_dict={X2:test_text_feature})
val_image_hash = sess.run(image_hash_map,feed_dict={X1:val_image_feature})
val_text_hash = sess.run(text_hash_map,feed_dict={X2:val_text_feature})

mAPi2t = MAP(test_image_hash,test_image_label,val_text_hash,val_text_label)
print("I2T:%f" % mAPi2t)
mAPt2i = MAP(test_text_hash,test_text_label,val_image_hash,val_image_label)
print("T2I:%f" % mAPt2i)
