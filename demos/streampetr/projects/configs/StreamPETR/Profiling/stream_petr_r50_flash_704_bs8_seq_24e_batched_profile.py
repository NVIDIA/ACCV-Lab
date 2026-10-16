# From StreamPETR (https://github.com/exiawsh/StreamPETR).
# No copyright holder is named upstream; this does not imply the absence of
# copyright.
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: Configuration adjusted for profiling
#          (based on stream_petr_r50_flash_704_bs2_seq_24e.py).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The following lines are from the original configuration base file:
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
workflow = [('train', 1)]

backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

num_gpus = 1
batch_size = 8
num_iters_per_epoch = 50 #24 * 323 // (num_gpus * batch_size)
num_epochs = 1#24

queue_length = 1
num_frame_losses = 1
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='Petr3D',
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=True,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        #with_cp=True,
        with_cp=False,
        style='pytorch'),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    img_roi_head=dict(
        type='FocalHeadBatched',
        num_classes=10,
        in_channels=256,
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
        train_cfg=dict(
        assigner2d=dict(
            type='HungarianAssigner2DBatched',
            cls_cost=dict(type='FocalLossCostBatched', weight=2.0),
            reg_cost=dict(type='BBoxL1CostBatched', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCostBatched', iou_mode='giou', weight=2.0),
            centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0))),
        ),
    pts_bbox_head=dict(
        type='StreamPETRHeadBatched',
        num_classes=10,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=10, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTemporalDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    #with_cp=True,  ###use checkpoint to save memory
                    with_cp=False,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3DBatched',
            cls_cost=dict(type='FocalLossCostBatched', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range),)))


dataset_type = 'CustomNuScenesDataset'
data_root = '/data/nuscenes/'

file_client_args = dict(backend='disk')


#ida_aug_conf = {
#        "resize_lim": (0.38, 0.55),
#        "final_dim": (256, 704),
#        "bot_pct_lim": (0.0, 0.0),
#        "rot_lim": (0.0, 0.0),
#        "H": 900,
#        "W": 1600,
#        "rand_flip": True,
#    }

# @MARK: For disabling randomizatinos
ida_aug_conf = {
        "resize_lim": (0.44, 0.44),
        "final_dim": (256, 704),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=True),
    #dict(type='GlobalRotScaleTransImage',
    #        rot_range=[-0.3925, 0.3925],
    #        translation_std=[0, 0, 0],
    #        scale_ratio_range=[0.95, 1.05],
    #        reverse_angle=True,
    #        training=True,
    #        ),
    # @MARK: For disabling randomizations (otherwise, use above version)
    dict(type='GlobalRotScaleTransImage',
            rot_range=[0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[1.05, 1.05],
            reverse_angle=True,
            training=True,
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'] + collect_keys,
            meta_keys=('filename', 'ori_shape', 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_temporal_infos_train.pkl',
        num_frame_losses=num_frame_losses,
        seq_split_num=2, # streaming video training
        seq_mode=True, # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length, data_root=data_root, ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length, data_root=data_root, ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    # @MARK: set randomize to false in line below to disable randomizations
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler', randomize=False),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )


optimizer = dict(
    type='AdamW', 
    lr=4e-4, # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.25), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHookWithTime', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)
#find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
find_unused_parameters=True #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(
    type='IterBasedRunnerWithTime', max_iters=num_epochs * num_iters_per_epoch)
#runner = dict(
#    type='DataPipelineRuntimeEvalRunner', max_iters=num_epochs * num_iters_per_epoch, num_warmup_iters=300)
load_from=None
resume_from=None

dali_preproc_config = dict(
    image_size = [1600, 900],
    pipeline_config = {
        "num_workers": 1,
        "num_threads": 4,
        "prefetch_queue_depth": 2
    },
    decoder_config = {
        "hw_decoder_load": 0.65,
        "as_rgb": True,
    },
    point_cloud_range = {
        "min": [-51.2, -51.2, None],
        "max": [51.2, 51.2, None]
    },
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    #affine_trafo_config = {
    #    "rand_flip": True,
    #    "resize_lim": (0.86, 1.25),
    #    "output_hw": (256, 704),
    #},
    # @MARK: For disabling randomizatinos
    affine_trafo_config = {
        "rand_flip": False,
        "resize_lim": (1.0, 1.0),
        "output_hw": (256, 704),
    },
    perform_3d_augmentation = True,
    #bev_bboxes_trafo_config = {
    #    "rotation_range":[-0.3925, 0.3925],
    #    "translation_range":[0, 0, 0],
    #    "scaling_range":[0.95, 1.05],
    #},
    # @MARK: For disabling randomizatinos
    bev_bboxes_trafo_config = {
        "rotation_range":[0.3925, 0.3925],
        "translation_range":[0, 0, 0],
        "scaling_range":[1.05, 1.05],
    },
    use_reproducible_seed=True,
    # @MARK: set to `False` to disable randomizations
    randomize_sample_selection=False,
    use_input_iterable = True,
    max_iterations_input_callable = num_iters_per_epoch * num_epochs
)

data_loading_config = dict(
    use_dali_pipeline=False,
)

runtime_meas_config = dict(
    stopwatch = dict(
        enable = True,
        num_warmup_iters = 50,
        print_every_n_iters = 50,
        do_cuda_sync = True,
    ),
    nvtx_range_wrapper = dict(
        enable = True,
        sync_on_push = True,
        sync_on_pop = True,
        keep_track_of_range_order = False,
    ),
)
