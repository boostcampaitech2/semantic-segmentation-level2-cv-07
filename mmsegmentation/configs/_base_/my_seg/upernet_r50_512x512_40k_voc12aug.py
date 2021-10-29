_base_ = [
    '../models/upernet_r50.py',
    './dataset.py',
    '../default_runtime.py', '../schedules/schedule_160k.py'
]
model = dict(
    decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))
