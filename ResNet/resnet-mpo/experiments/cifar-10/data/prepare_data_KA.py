################################################################
#           Load and unpack CIFAR-10 python version            #
# from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz #
#                                                              #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  #
#            Run this scipt with python3 only                  #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  #
################################################################

import pickle
import numpy as np

batches_dir = 'cifar-10-batches-py' 

def unpickle(fname):
    fo = open(fname, 'rb')
    d = pickle.load(fo, encoding='bytes')
    fo.close()
    data = np.reshape(d[b'data'], [-1, 3, 32, 32])
    data = np.transpose(data, [0, 2, 3, 1]) # (10000,32,32,3)
    data = np.reshape(data, [-1, 32*32*3])
    labels = np.array(d[b'labels'], dtype='int8')
    return data, labels

# image(N, 32x32x3)
def ket_augmentation(images):
    # images_KA(n, y4,y3,y2,y1,y0, x4,x3,x2,x1,x0, j)
    #           0,  1, 2, 3, 4, 5,  6, 7, 8, 9,10,11)
    images_KA = np.reshape(images, [-1, 2,2,2,2,2, 2,2,2,2,2, 3])

    # images_KA(n, y4,x4, y3,x3, y2,x2, y1,x1, y0,x0, j)
    images_KA = np.transpose(images_KA, [0, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 11])
    # images_KA = np.transpose(images_KA, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    images_KA = np.reshape(images_KA, [-1, 32*32*3])
    return images_KA

for x in range(1, 6):
    fname = batches_dir + '/data_batch_' + str(x)
    data, labels = unpickle(fname)
    if x == 1:
        train_images = data
        train_labels = labels
    else:
        train_images = np.vstack((train_images, data))
        train_labels = np.concatenate((train_labels, labels))
train_images = ket_augmentation(train_images)
# train_images_1 = ket_augmentation(train_images)
# train_diff = train_images - train_images_1
# print(train_diff.max())

validation_images, validation_labels = unpickle(batches_dir + '/test_batch')
validation_images = ket_augmentation(validation_images)

print(train_images.shape, validation_images.shape)
print(train_labels.shape, validation_labels.shape)
np.savez_compressed('cifar_KA', train_images=train_images, validation_images=validation_images,
                    train_labels=train_labels, validation_labels=validation_labels)


    







