
# DiDA: Disambiguated Domain Alignment for Cross-Domain Retrieval with Partial Labels




**Run OfficeHome with partial_rate=0.3**
```shell
python -u train.py --gpu_id 0 --dataset office_home --class_num 65 --dset a2c --PLL_method DiDA --partial_rate 0.3 --lr_decay_epochs 20,40 --batch_size 16 --lr 3e-3 --alpha_weight 5.0 --beta_weight 0.01 --alpha_t 5 --beta_t 25
```


**Run Image_CLEF with partial_rate=0.5**
```shell
python -u train.py --gpu_id 0 --dataset image_CLEF --class_num 12 --dset p2c --PLL_method DiDA --partial_rate 0.5 --lr_decay_epochs 15,25,35 --batch_size 16 --lr 3e-3 --alpha_weight 5.0 --beta_weight 0.01 --alpha_t 5 --beta_t 25 
```


**Run Office31 with partial_rate=0.3**
```shell
python -u train.py --gpu_id 0 --dataset office31 --class_num 31 --dset a2d --PLL_method DiDA --partial_rate 0.3 --lr_decay_epochs 20,40 --batch_size 16 --lr 3e-3 --alpha_weight 5.0 --beta_weight 0.1 --alpha_t 25 --beta_t 25
```






