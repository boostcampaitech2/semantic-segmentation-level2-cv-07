##########################################################
##################### model settings######################
##########################################################
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
                # num_channels=(48, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=0.5,
            num_classes=11,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=0.5,
            num_classes=11,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

##########################################################
################## dataset settings#######################
##########################################################

# dataset settings
dataset_type = 'COCOCustom'
data_root = '../input/mmseg'

# class settings

# set normalize value
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        img_dir=data_root + '/img_dir/training',
        ann_dir=data_root + '/ann_dir/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_dir=data_root + '/img_dir/validation',
        ann_dir=data_root + '/ann_dir/validation',
        pipeline=test_pipeline),
    val_loss=dict(
        type=dataset_type,
        img_dir=data_root + '/img_dir/validation',
        ann_dir=data_root + '/ann_dir/validation',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        test_mode=True,
        img_dir=data_root + "/img_dir/test",
        pipeline=test_pipeline))
##########################################################
################## runtime settings #######################
##########################################################
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
        dict(type='EpochWandbLogger',
        init_kwargs=dict(
            project='segmentation',
            name='ocrnet_48'))
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True


############################################################
################### runtime settings #######################
############################################################
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5,
)
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=200,
#     warmup_ratio=0.01,
#     min_lr_ratio=1e-2,
# )
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=30)
evaluation = dict(interval=1, by_epoch=True, metric='mIoU')
