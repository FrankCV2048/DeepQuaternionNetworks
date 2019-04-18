import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('train_iter', 4000, 'Total training iter')
flags.DEFINE_integer('validation_step', 20, 'Total training iter')
flags.DEFINE_integer('step',50, 'Save after ... iteration')
flags.DEFINE_integer('DEV_NUMBER', -2000, '验证集数量')
flags.DEFINE_integer('batch_size',10, '批大小')
flags.DEFINE_string('BASE_PATH', 'C:\\cata', '图片位置')
flags.DEFINE_string('negative_file', 'negative_pairs_path.txt', '不同人的文件')
flags.DEFINE_string('positive_file', 'positive_pairs_path.txt', '相同人的文件')
flags.DEFINE_string('triplet_file', 'triplet_aug_pos_neg.txt', '三元组的文件')