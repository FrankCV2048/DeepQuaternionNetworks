from keras.models import load_model
from dataset import *
import cv2
from keras import backend as K
def get_im_cv2(paths, img_rows=512, img_cols=512, color_type=3, normalize=True):
    imgs = []
    for path in paths:
        if color_type == 1:
            img = cv2.imread(path, 0)
        elif color_type == 3:
            #img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            img = Image.open(path)
            img = np.asarray(img, dtype='float32')
            # Reduce size
            # resized = cv2.resize(img, (img_cols, img_rows))
            if normalize:
                img = img.astype('float32')
                img /= 255.0
                # resized -= 1.
                imgs.append(img)
    imgs=np.array(imgs).reshape(len(paths),img_cols,img_rows,color_type)
    return imgs

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


model=load_model('quaternion_weights.hd5',custom_objects={'contrastive_loss': contrastive_loss})
# [get_im_cv2(left_dev), get_im_cv2(right_dev)],similar_dev

x=model.predict([get_im_cv2(left_dev), get_im_cv2(right_dev)])
num_true=0
for i in range(len(x)):
    if x[i]>0.4:
        x[i]=1
    else:
        x[i]=0
    if x[i]==similar_dev[i]:
        num_true+=1
print(num_true)
