import numpy as np
from PIL import Image
from config import FLAGS
import os
import math

DEV_NUMBER = FLAGS.train_iter
BASE_PATH = FLAGS.BASE_PATH
batch_size = FLAGS.batch_size
#ss
negative_pairs_path_file = open(FLAGS.negative_file, 'r')
negative_pairs_path_lines = negative_pairs_path_file.readlines()
positive_pairs_path_file = open(FLAGS.positive_file, 'r')
positive_pairs_path_lines = positive_pairs_path_file.readlines()
#########################################
#########################################
triplet_pairs_path_file =open(FLAGS.triplet_file, 'r')
triplet_pairs_path_lines = triplet_pairs_path_file.readlines()
triplet_pairs_path_lines=np.asarray(triplet_pairs_path_lines)
shuffle_in = np.random.permutation(np.arange(len(triplet_pairs_path_lines)))
triplet_pairs_path_shuffle = triplet_pairs_path_lines[shuffle_in]
triplet_pairs_train,triplet_pairs_dev=triplet_pairs_path_lines[:FLAGS.DEV_NUMBER],triplet_pairs_path_lines[FLAGS.DEV_NUMBER:]

def batch_triplet_path(index):
    index_end=index + batch_size
    if index_end>len(triplet_pairs_path_shuffle):
        return triplet_pairs_path_shuffle[index:len(triplet_pairs_path_shuffle)],index_end
    return triplet_pairs_path_shuffle[index:index_end],index_end

def batch_triplet_array(batch_triplet_lines):
    augs=[];
    poss=[];
    negs=[];
    for line in batch_triplet_lines:
        pics = line.strip().split('  ')
        aug = pics[0]
        augs.append(aug)

        pos = pics[1]
        poss.append(pos)

        neg = pics[2]
        negs.append(neg)
    print(len(augs))
    demo_A = vectorize_imgs(augs)
    a = np.asarray(demo_A, dtype='float32') / 255.
    demo_B = vectorize_imgs(poss)
    b = np.asarray(demo_B, dtype='float32') / 255.
    demo_C = vectorize_imgs(negs)
    c = np.asarray(demo_C, dtype='float32') / 255.
    return  a,b,c


#########################################
#########################################
left_image_path_list = []
right_image_path_list = []
similar_list = []

for line in negative_pairs_path_lines:
    left_right = line.strip().split('  ')
    left_image_path_list.append(left_right[0])
    right_image_path_list.append(left_right[1])
    similar_list.append(0)

for line in positive_pairs_path_lines:
    left_right = line.strip().split('  ')
    left_image_path_list.append(left_right[0])
    right_image_path_list.append(left_right[1])
    similar_list.append(1)

left_image_path_list = np.asarray(left_image_path_list)
right_image_path_list = np.asarray(right_image_path_list)
similar_list = np.asarray(similar_list)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(similar_list)))
left_shuffled = left_image_path_list[shuffle_indices]
right_shuffled = right_image_path_list[shuffle_indices]
similar_shuffled = similar_list[shuffle_indices]

left_train, left_dev = left_shuffled[:20050], left_shuffled[30970:]
right_train, right_dev = right_shuffled[:20050], right_shuffled[30970:]
similar_train, similar_dev = similar_shuffled[:20050], similar_shuffled[30970:]

#自定义
def random_mini_batches(left, right, similar, mini_batch_size=16, seed=0):
    m = left.shape[0]
    mini_batches = []
    np.random.seed(seed)

    # permutation = list(np.random.permutation(m))
    shuffled_left = left
    shuffled_right = right
    shuffled_similar = similar
    # shuffled_similar = similar[permutation].reshape(similar.shape[0])

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        shuffled_left = shuffled_left[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        shuffled_right = shuffled_right[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        shuffled_similar = shuffled_similar[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (shuffled_left, shuffled_right, shuffled_similar)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        shuffled_left = shuffled_left[num_complete_minibatches * mini_batch_size: m]
        shuffled_right = shuffled_right[num_complete_minibatches * mini_batch_size: m]
        shuffled_similar = shuffled_similar[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (shuffled_left, shuffled_right, shuffled_similar)
        mini_batches.append(mini_batch)

    return mini_batches



def vectorize_imgs(img_path_list):
    image_arr_list = []
    for img_path in img_path_list:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img_arr = np.asarray(img, dtype='float32')
            if len(img_arr.shape) != 3:
                # img = img.convert('L')
                # img_arr = np.asarray(img, dtype='float32')
                img_arr = np.expand_dims(img_arr, axis=2)
                img_arr = np.concatenate((img_arr, img_arr, img_arr), axis=-1)
                print(img_arr.shape)
                print('将一维修改为三维: %s'% img_path)
            #均值化
            # mean_image=np.mean(img_arr,axis=0)
            # img_arr=img_arr-mean_image

            image_arr_list.append(img_arr)
            if img_arr.shape[0]!= 512:
                print(img_arr.shape)
                print('当图片不是512: %s' % img_path)
        else:
            print(img_path)
    return image_arr_list


def get_batch_image_path(left_train, right_train, similar_train, start):
    end = (start + batch_size) % len(similar_train)
    if start < end:
        return left_train[start:end], right_train[start:end], similar_train[start:end], end
    # 当 start > end 时，从头返回
    return np.concatenate([left_train[start:], left_train[:end]]), \
           np.concatenate([right_train[start:], right_train[:end]]), \
           np.concatenate([similar_train[start:], similar_train[:end]]), \
           end


def get_batch_image_array(batch_left, batch_right, batch_similar):
    demo_A=vectorize_imgs(batch_left)
    a=np.asarray(demo_A, dtype='float32') / 255.
    demo_B=vectorize_imgs(batch_right)
    j=0
    for i in demo_B:
        j+=1
        if(len(i.shape)!=3):
            print(j)
    b=np.asarray(demo_B, dtype='float32') / 255.
    c=np.asarray(batch_similar)[:, np.newaxis]
    return a,b,c


if __name__ == '__main__':
    pass
