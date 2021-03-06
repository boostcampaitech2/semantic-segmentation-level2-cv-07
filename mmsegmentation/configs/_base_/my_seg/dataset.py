# dataset settings
dataset_type = 'CustomDataset'
data_root = '/opt/ml/segmentation/input/mmseg/'

# class settings
classes = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic','Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
palette = [
    [0, 0, 0],
    [192, 0, 128], [0, 128, 192], [0, 128, 64],
    [128, 0, 0], [64, 0, 128], [64, 0, 192],
    [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192]
    ]

# set normalize value
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# when align = False
crop_size = (512, 512)

# when align = True
# crop_size = (513, 513)

albu_train_transforms = [
    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=0,
    #     interpolation=1,
    #     p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.5),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    # dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.3),
    
]

albu_train_transforms_test = [
    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=0,
    #     interpolation=1,
    #     p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.2),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='RGBShift',
    #             r_shift_limit=20,
    #             g_shift_limit=20,
    #             b_shift_limit=20,
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0)
    #     ],
    #     p=0.5),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    # dict(type='ChannelShuffle', p=0.1),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Blur', blur_limit=3, p=1.0),
    #         dict(type='MedianBlur', blur_limit=3, p=1.0)
    #     ],
    #     p=0.3),
    # dict(type='ImageCompression', always_apply=False, p=0.5, quality_lower=30, quality_upper=60, compression_type=0)
    
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),

    # when align = False
    dict(type='Resize', img_scale=[(256, 256), (384, 384), (512, 512), (640, 640), (768, 768), (896, 896), (1024,1024)], multiscale_mode = "value"),
    # dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)), # keep_ratio=True

    # when align = True
    # dict(type='Resize', img_scale=(513, 513), keep_ratio=True),# ratio_range=(0.5, 2.0)

    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=(-90, 90)),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask',
        },
        update_pad_shape=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),

    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # when align = False
        img_scale=(512, 512),

        # with_out_flip
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        
        # with flip
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=[False, False, False, False, False, False, True, True, True, True, True, True],
         
        # when align = True
        # img_scale=[(257, 257), (385, 385), (513, 513), (641, 641), (769, 769), (897, 897)],
        
        
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='PhotoMetricDistortion'),
            dict(type='RandomFlip'),
            # dict(type='RandomRotate', prob=0.5, degree=(-90, 90)),
            dict(
                type='Albu',
                transforms=albu_train_transforms_test,
                keymap={
                    'img': 'image',
                    'gt_semantic_seg': 'mask',
                },
                update_pad_shape=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False, 
        # img_dir=data_root + "images/training_all",
        # ann_dir=data_root + "annotations/training_all",
        
        # img_dir=data_root + "images/training",
        # ann_dir=data_root + "annotations/training",
        
        # img_dir=data_root + "images/training_with_psuedo",
        # ann_dir=data_root + "annotations/training_with_psuedo",
        
        img_dir=data_root + "images/training_with_obj_aug",
        ann_dir=data_root + "annotations/training_with_obj_aug",
        
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "images/validation",
        ann_dir=data_root + "annotations/validation",
        pipeline=test_pipeline),
    val_loss=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False, 
        img_dir=data_root + "images/validation",
        ann_dir=data_root + "annotations/validation",
        pipeline=train_pipeline),
    test=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        reduce_zero_label=False,
        img_dir=data_root + "images/test",

        # img_dir=data_root + "images/validation",
        # ann_dir=data_root + "annotations/validation",

        pipeline=test_pipeline))