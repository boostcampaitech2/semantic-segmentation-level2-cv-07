## How to use
```bash
python main.py \
--config '../../configs/_base_/my_seg/fcn_hr48_512x512_160k_ade20k.py' \
--checkpoint '../../work_dirs/fcn_hr48_512x512_160k_ade20k/epo_30_adamW.pth' \
--output './saved/hrnet.png'
```
- `--config`: config file path
- `--checkpoint`: checkpoint file path(`.pth`)
- `--output`: output file path/name (`.png`)