_base_ = [
    '../models/fcn_hr18.py', './dataset.py',
    '../default_runtime.py', '../schedules/schedule_160k.py'
]
model = dict(decode_head=dict(num_classes=11))
