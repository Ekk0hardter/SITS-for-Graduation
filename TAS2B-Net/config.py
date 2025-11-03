from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = './results/output'
_C.LOG_DIR = './results/log'
_C.PRINT_FREQ = 5
_C.SEED = None
_C.WORKERS = 16
_C.DEVICE = 'cuda'
_C.FUNE_TUNE = False
_C.N_FOLD = None
_C.RESUME = False

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'Segmentor'
_C.MODEL.PRETRAINED = '../final_state.pth'

# loss
_C.LOSS = CN()
_C.LOSS.TYPE = 'focal_ce'
_C.LOSS.SMOOTH_FACTOR = 0.0
_C.LOSS.FOCAL = [0.25, 2.0]
_C.LOSS.IGNORE_INDEX = -1

# learning rate
_C.LR = CN()
_C.LR.LR_FACTOR = 0.1
_C.LR.LR_STEP = [0.7, 0.9]
_C.LR.LR = 0.0002
_C.LR.LR_SCHEDULER = 'step'
_C.LR.WARMUP_ITERS_RATIO = 0.1
_C.LR.WARMUP_FACTOR = 1e-03

# optimizer
_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = 'adamw'
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WD = 0.005



# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = '../PASTIS-R'
_C.DATASET.DATASET = 'PASTIS-R'
_C.DATASET.MODALITY = ['S2']
_C.DATASET.INPUT_DIM = [10]
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.N_FOLD = 0
_C.DATASET.NBINS = 16
_C.DATASET.TEMP_DROP_RATE = [[0.2, 0.4]]
_C.DATASET.PAD_VALUE = 0
_C.DATASET.TASK_TYPE = 'sem_seg'

_C.DATASET.TRAIN_TILES = [{'year': 2017, 'country': 'austria', 'tile': '33UVP'},
                         {'year': 2017, 'country': 'denmark', 'tile': '32VNH'},
                          {'year': 2017, 'country': 'france', 'tile': '30TXT'}]
_C.DATASET.TEST_TILES = [{'year': 2017, 'country': 'france', 'tile': '31TCJ'}]
_C.DATASET.THING_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
_C.DATASET.Z_NORM = True

# training
_C.TRAIN = CN()
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 200
_C.TRAIN.BATCH_SIZE_PER_GPU = 4
_C.TRAIN.RANDOM_CROP = True
_C.TRAIN.CROP_SIZE = (48, 48)

# testing
_C.TEST = CN()
_C.TEST.MULTI_CROP_TEST = True
_C.TEST.BATCH_SIZE_PER_GPU = 1
_C.TEST.MODEL_FILE = './'
_C.TEST.CROP_SIZE = (128, 128)
_C.TEST.RETURN_ATTN = False

# Segmentor specification
_C.SEGMENTOR = CN()
_C.SEGMENTOR.POS_ENCODE_TYPE = 'default'
_C.SEGMENTOR.WITH_GDD_POS = False
_C.SEGMENTOR.PE_DIM = 128
_C.SEGMENTOR.PE_T = 1000
_C.SEGMENTOR.MAX_TEMP_LEN = 1000
_C.SEGMENTOR.SPACE_ENCODER_TYPE = 'ekko'

# ------------------------------------------------------------------

# exchanger specification
_C.EXCHANGER = CN()
_C.EXCHANGER.EMBED_DIMS = [128, 128]
_C.EXCHANGER.NUM_TOKEN_LIST = [8, 8]
_C.EXCHANGER.NUM_HEADS_LIST = [8, 8]
_C.EXCHANGER.DROP_PATH_RATE = 0.1
_C.EXCHANGER.MLP_NORM = 'batchnorm'
_C.EXCHANGER.MLP_ACT = 'gelu'

# GPBlock specification
_C.GPBLOCK = CN()
_C.GPBLOCK.EMBED_DIMS = 128
_C.GPBLOCK.NUM_GROUP_TOKENS = 8
_C.GPBLOCK.NUM_HEADS = 8
_C.GPBLOCK.ACT_TYPE = 'gelu'
_C.GPBLOCK.NORM_TYPE = 'layernorm'
_C.GPBLOCK.FFN_RATIO = 4.0
_C.GPBLOCK.QKV_BIAS = True
_C.GPBLOCK.MIXER_DEPTH = 1
_C.GPBLOCK.MIXER_TOKEN_EXPANSION = 0.5
_C.GPBLOCK.MIXER_CHANNEL_EXPANSION = 4.0
_C.GPBLOCK.DROP = 0.1
_C.GPBLOCK.ATTN_DROP = 0.
_C.GPBLOCK.DROP_PATH = 0.1
_C.GPBLOCK.UNTIED_POS_ENCODE = True
_C.GPBLOCK.ADD_POS_TOKEN = True

# -------------------------------------------------------------


# 使用UNET作为空间编解码器时需要用的参数
_C.UNET = CN()
_C.UNET.BASE_CHANNELS = 64
_C.UNET.NUM_STAGES = 4
_C.UNET.STRIDES = [1, 1, 1, 1]
_C.UNET.ENC_NUM_CONVS = [2, 2, 2, 2]
_C.UNET.DEC_NUM_CONVS = [2, 2, 2]
_C.UNET.DOWNSAMPLES = [True, True, True]
_C.UNET.ENC_DILATIONS = [1, 1, 1, 1]
_C.UNET.DEC_DILATIONS = [1, 1, 1]
_C.UNET.NORM_TYPE = 'bn'
_C.UNET.ACT_TYPE = 'gelu'
_C.UNET.UPSAMPLE_TYPE = 'interp'
_C.UNET.IN_CHANNELS = [64, 128, 320, 512]



# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False


# def update_config(cfg, args):
#     cfg.defrost()          # 允许修改config

#     cfg.merge_from_file(args.cfg)  # 从文件加载配置
#     cfg.merge_from_list(args.opts) # 从命令行参数列表更新配置 大部分都被更新了

#     cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

