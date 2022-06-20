# Facke

Project homepage: [Facke](https://github.com/Ailon-Island/Facke). Check out the GitHub homepage for more introductions and instructions.

Facke is a ~~(course)~~ project on face swapping and fake face generating. The whole project is heavily based on [SimSwap](https://github.com/neuralchen/SimSwap).

## Preparation

### Pretrained Models

It is a nice choice is to download the checkpoints if you want to swap faces with Facke as soon as possible. Checkpoints for utility models are already included in the uploaded .zip file. And you can also get them (`utils\`) from [JBox](https://jbox.sjtu.edu.cn/l/m1Xxtt). Checkpoints (`checkpoints\`\) for main models are also available at [JBox](https://jbox.sjtu.edu.cn/l/m1Xxtt).

To ensure that you are fully prepared, you should carefully compare the location you place the downloaded checkpoints: both `checkpoints\` and `utils\` shall be merged with the directories with the identical name in `Facke\` root directory.

### Dataset

Our dataloader is specified for the dataset of [VGGFace2-HQ](https://github.com/NNNNAI/VGGFace2-HQ). You can download a modified version (`datasets\`) [here](https://jbox.sjtu.edu.cn/l/m1Xxtt). (We have sliced the dataset into train and test set.) As an alternative of downloading separated files, you can instead download the `datasets\VGGFace2-HQ.zip` and unzip it in the way as we do on the cloud drive.

Or otherwise, you can go download your own dataset, place it in `datasets\`, and then run `python create_dataset --data_root datasets\{YOUR DATASET}` to generate latent ID for each image (Don't worry! Our dataloader will also check and make up the missing latent ID!). Some of our training schedules require identity labels. We only support the way of labeling with directory name. You shall refer to the structure of our modified VGGFace2-HQ dataset. If you don't need such training schedules, you can place all your pictures in one directory.

## Demonstration

### Demo on Dataset

We have prepared some scripts for demonstration. In specific, you can run SimSwap and ILVR on the dataset to generate demonstrative face swapped images. The result images will be saved in `output\`.

You can run `python demo_SimSwap.py --name {TASK NAME} --epoch_label {BEST EPOCH}` for SimSwap demonstration.  Checkout best epoch list here for our pretrained checkpoints.

For ILVR, since the pretrained model does not perfectly fit default settings, you need to give more arguments.

```bash
python demo_ILVR.py \
	--attention_resolutions 16 \
	--diffusion_steps 1000 \
	--dropout 0.0 \
	--image_size 256 \
	--learn_sigma \
	--noise_schedule linear \
	--num_channels 128 \
	--num_head_channels 64 \
	--num_res_blocks 1 \
	--resblock_updown \
	--use_scale_shift_norm \
	--timestep_respacing 100 \
	--down_N 32 \
	--range_t 20 \
	--clip_denoised \
	--batchSize 8 \
	--name {TASK NAME} \
	--epoch_label {BEST EPOCH}
```

### Demo on Your Own Images

Also, you are allowed to use all our models (CVAE, SimSwap, CVAE-GAN, ILVR) to swap faces in your own images. To do this, you shall first go to `input\` and place your own source and target images in `input\source\` and `input\target\`, respectively. To keep a deterministic order, images are required to be named from $0$ to $\text{\{NUM IMAGES\}} - 1$ in both of the directories. (*e.g.*, `0.jpg`, `1.jpg`, `2.jpg`, `3.jpg`, `4.jpg`, `5.jpg` for altogether 6 images.)

Then you can run `python swap.py --model [SimSwap, CVAE, ILVR] --name {TASK NAME} --input_format {IMAGE FORMAT, e.g., jpg} --epoch_label best --swap_title {SUBTASK TITLE}` to get a swapping matrix of these images. Without arguments `--model` and `--name`, this calls the SimSwap checkpoint by default. They can be switched to any other checkpoints by feeding the checkpoint name as task name and providing the corresponding model type (`SimSwap` for SimSwap, `CVAE` for CVAE and CVAE-GAN, and `ILVR` for ILVR). Also, you should manually designate a saved epoch/iter by feeding `--epoch_label [{XXX}_iter, latest]`.

The output image is saved as `output\{name}\{name}_{swap_title}.jpg`.

For ILVR models, you still need to make extra specifications for correct model hyperparameters.

```bash
python swap.py \
	--attention_resolutions 16 \
	--diffusion_steps 1000 \
	--dropout 0.0 \
	--image_size 256 \
	--learn_sigma \
	--noise_schedule linear \
	--num_channels 128 \
	--num_head_channels 64 \
	--num_res_blocks 1 \
	--resblock_updown \
	--use_scale_shift_norm \
	--timestep_respacing 100 \
	--down_N {DOWNSAMPLE N} \
	--range_t 20 \
	--clip_denoised \
	--batchSize 8 \
	--model ILVR \
	--name {TASK NAME} \
	--swap_title {SUBTASK NAME} \
	--epoch_label {BEST EPOCH} \
	--input_format {IMAGE FORMAT}
```

## Training

Besides using checkpoints we provide, you can also train/finetune your own models.

### SimSwap

We have made our own training script before [SimSwap](https://github.com/neuralchen/SimSwap) released the official version. They have implemented projected discriminators then, which is not mentioned in the paper. Our model does not contain projected discriminators. Maybe it will get updated later.

```bash
python -u ./train_SimSwap.py \
--batchSize 32 \
--name {TASK NAME} \
--nThreads 32 \
--feat_mode [w, w*, n, o]
```

The default value of `--feat_mode` is `w`, which stands for Weak Feature Matching. Similarly, `w*` stands for $\overline{\text{wFM}}$, `n` stands for nFM, and `o` stands for oFM. Additionally, you can feed `--no_intra_ID_random` to disable Identity Grouping. Also, you can explicitly assign `--display_freq` and `--print_freq` (in terms of iteration) to control the frequency in which the script spawns samples and prints training logs.

### CVAE-GAN

We have deprecated the original CVAE model. With our scripts, you are able to train CVAE only.

```bash
python -u ./train_CVAE.py \
--batchSize 32 \
--model CVAE
--name {TASK NAME} \
--nThreads 32 \
```

You can also specify `--feat_mode`, `--display_freq`, and `--print_freq`.

### ILVR

Our ILVR model is based on [ILVR + ADM](https://github.com/jychoi118/ilvr_adm). And we provide scripts to finetune the DDPM to be used in ILVR on the local dataset. 

```bash
python -u finetune_DDPM.py \
	--attention_resolutions 16 \
	--diffusion_steps 1000 \
	--dropout 0.0 \
	--image_size 256 \
	--learn_sigma \
	--noise_schedule linear \
	--num_channels 128 \
	--num_head_channels 64 \
	--num_res_blocks 1 \
	--resblock_updown \
	--use_scale_shift_norm \
	--timestep_respacing 100 \
	--DDPM_pth .//utils//guided_diffusion//models//ffhq_10m.pt \
	--down_N {DOWNSAMPLE N} \
	--range_t 20 \
	--clip_denoised \
	--batchSize 8 \
	--name {TASK NAME} \
	--lr {LEARNING RATE}
```

## Benchmarking

It is easy to benchmark a task on test set with our scripts. We evaluates metrics of ID Loss, ID Retrieval, and Reconstruction Loss. (Criteria like LPISP, FID, etc. may be added in the future.) 

For models other than ILVR, run `python -u benchmark.py --batchSize 32 --nThreads 32 --model {MODEL TYPE} --name {TASK NAME} --benchmark_skip 1 --benchmark_coarse 4000 --benchmark_fine 20000`. 

To faster the process, we provide a sampling benchmark schedule (default on). We first scan checkpoints iteration by iteration, with less samples from test set (coarse). For the best models found in the coarse benchmark, we resample a larger subset from test set to get more precise results (fine). With `--benchmark_coarse` and `--benchmark_fine` you can specify the sample sizes. Also, we provide a optional skip in scanning (default off). You are allowed to decide the step size while sliding over the checkpoints with `--benchmark_skip` (in terms of file). If it is larger than $1$, our script will skip some checkpoints. Note that we have ensured that the latest iteration will be surely benchmarked.

For ILVR, still you need more arguments.

```
python -u benchmark.py \
	--benchmark_skip 4 \
	--benchmark_coarse 64 \
	--benchmark_fine 320 \
	--attention_resolutions 16 \
	--diffusion_steps 1000 \
	--dropout 0.0 \
	--image_size 256 \
	--learn_sigma \
	--noise_schedule linear \
	--num_channels 128 \
	--num_head_channels 64 \
	--num_res_blocks 1 \
	--resblock_updown \
	--use_scale_shift_norm \
	--timestep_respacing 100 \
	--DDPM_pth ./utils/guided_diffusion/models/ffhq_10m.pt \
	--down_N 32 \
	--range_t 20 \
	--clip_denoised \
	--batchSize 32 \
	--name ILVR_pretrained \
	--epoch_label pretrained \
	--model ILVR
```


## More
For more details of using our scripts, you can refer to bash scripts in `shell\`.
