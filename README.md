# rf_inversion

This repository is unofficial implementation of RF Inversion, "Semantic Image Inversion and Editing using
Stochastic Rectified Differential Equations".
https://rf-inversion.github.io/ 

``` コマンド例
python main.py --image examples/man.jpg --prompt "A portrait of a man wearing glasses" --output outputs/man_glasses_eta09_s6e20.jpg --eta 0.9 --start_timestep 6 --stop_timestep 20 
```

etaが大きいと元の画像に近くなる

start_timestepとstop_timestepはetaを有効にするtimestepで
`0 <= start_timestep < stop_timestep < num_inference_step`

コントロールが難しい

## 結果
<img src="examples/cat.jpg" alt="Cat" width="300px">
<img src="img/cat_lion_eta09_s0e5.jpg" alt="Cat Lion" width="300px">
<img src="img/cat_origami_cat_eta08_s0e2.jpg" alt="Cat Origami" width="300px">
<img src="img/cat_sleeping_cat.jpg" alt="Sleeping Cat" width="300px">
<img src="img/cat_tiger_eta08_s0e5.jpg" alt="Cat Tiger" width="300px">

<img src="examples/man.jpg" alt="Man" width="300px">
<img src="img/man_glasses_eta09_s6e20.jpg" alt="Man with Glasses" width="300px">
