from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 9 #16(배치사이즈의 제곱근은 자연수여야하며, 데이터셋의 크기는 배치사이즈의 배수여야 한다.)
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 1 #100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = 'D:/DataSet/HYU/Face/img_Crop/'
config.TRAIN.lr_img_path = 'D:/DataSet/Super Resolution/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'D:/DataSet/Super Resolution/DIV2K_valid_HR/'
config.VALID.lr_img_path = 'D:/DataSet/Super Resolution/DIV2K_valid_LR_bicubic/X4/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================/n")
        f.write(json.dumps(cfg, indent=4))
        f.write("/n================================================/n")
