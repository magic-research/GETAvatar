## 🔥 🔥 🔥GETAvatar: Generative Textured Meshes for Animatable Human Avatars (ICCV 2023)🔥 🔥 🔥<br><sub>Official PyTorch implementation </sub>

<table style="border:0px">
   <tr>
       <td><img src="./docs/assets/rp_mv.gif" frame=void rules=none></td>
       <td><img src="./docs/assets/thu_mocap_0070.gif" frame=void rules=none></td>
   </tr>
</table>

**GETAvatar: Generative Textured Meshes for Animatable Human Avatars**<br>
[Xuanmeng Zhang*](https://scholar.google.com/citations?user=QzlBBMEAAAAJ&hl=en), [Jianfeng Zhang*](http://jeff95.me/), [Rohan Chacko](https://rohanchacko.github.io/),
[Hongyi Xu](https://hongyixu37.github.io/homepage/), 
[Guoxian Song](https://guoxiansong.github.io/homepage/index.html), [Yi Yang](https://scholar.google.com.sg/citations?user=RMSuNFwAAAAJ&hl=en), [Jiashi Feng](https://sites.google.com/site/jshfeng/home) <br>
**[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_GETAvatar_Generative_Textured_Meshes_for_Animatable_Human_Avatars_ICCV_2023_paper.pdf), [Project Page](https://getavatar.github.io/)**

Abstract: *We study the problem of 3D-aware full-body human generation, aiming at creating animatable human avatars with high-quality textures and geometries. Generally, two challenges remain in this field: i) existing methods struggle to generate geometries with rich realistic details such as the wrinkles of garments; ii) they typically utilize volumetric radiance fields and neural renderers in the synthesis process, making high-resolution rendering non-trivial. To overcome these problems, we propose GETAvatar, a Generative model that directly generates Explicit Textured 3D meshes for animatable human Avatar, with photo-realistic appearance and fine geometric details. Specifically, we first design an articulated 3D human representation with explicit surface modeling, and enrich the generated humans with realistic surface details by learning from the 2D normal maps of 3D scan data. Second, with the explicit mesh representation, we can use a rasterization-based renderer to perform surface rendering, allowing us to achieve high-resolution image generation efficiently. Extensive experiments demonstrate that GETAvatar achieves state-of-the-art performance on 3D-aware human generation both in appearance and geometry quality. Notably, GETAvatar can generate images at 512x512 resolution with 17FPS and 1024x1024 resolution with 14FPS, improving upon previous methods by 2x.*







## 📢 News

- [2023-10-19]: Code and pretrained model on THuman2.0 released! Check more details [here](https://drive.google.com/drive/folders/195mqqOSuHl_1xmShXKna2S5RXijUEQft)

## ⚒️ Requirements

* We recommend Linux for performance and compatibility reasons.
* 1 &ndash; 8 high-end NVIDIA GPUs. We have done all testing and development using V100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.9.0. See https://pytorch.org for PyTorch install
  instructions.
* CUDA toolkit 11.1 or later.  (Why is a separate CUDA toolkit installation required? We
  use the custom CUDA extensions from the StyleGAN3 repo. Please
  see [Troubleshooting](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary))
  .
* Blender. Download Blender from [official link](https://www.blender.org/). We used [**blender-3.2.2-linux**](https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz), we haven't tested on other versions but newer versions should be OK.
* We also recommend to install Nvdiffrast following instructions
  from [official repo](https://github.com/NVlabs/nvdiffrast), and
  install [Kaolin](https://github.com/NVIDIAGameWorks/kaolin).
* We provide a [script](./install_getavatar.sh) to install packages.

## 🏃‍♂️ Getting Started

#### Clone the gitlab code and necessary files:

```bash
git clone https://github.com/magic-research/GETAvatar.git
cd GETAvatar; mkdir cache; cd cache
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl
```

#### SMPL models
Download the *SMPL* human models from [this](http://smpl.is.tue.mpg.de) (male, female and neutral models) and the [mixamo](https://www.mixamo.com) motion sequences from [here](https://drive.google.com/drive/folders/1iXD2CShfcjk8fxUAC0VmTdiKeDz-DOc8?usp=sharing).

**Place them as following:**
```bash
GETAvatar
|----smplx
    |----mocap
      |----mixamo
          |----0007  
          |----...
          |----0145  
    |----models
      |----smpl
          |----SMPL_FEMALE.pkl
          |----SMPL_MALE.pkl
          |----SMPL_NEUTRAL.pkl
|----...
```


## 📝 Preparing datasets

<!-- GET3D is trained on synthetic dataset. We provide rendering scripts for Shapenet. Please
refer to [readme](./render_shapenet_data/README.md) to download shapenet dataset and
render it. -->

We train GETAvatar on 3D human scan datasets (THuman2.0 and RenderPeople).
Here use THuman2.0 as an example because it's free. The same pipeline works also for the commericial dataset RenderPeople.


First, download [THuman2.0 dataset](https://github.com/ytrock/THuman2.0-Dataset) and download the [fitted SMPL results](https://dataset.ait.ethz.ch/downloads/gdna/THuman2.0_smpl.zip).


**Place them as following:**
```bash
GETAvatar
|----datasets
    |----THuman2.0
        |----THuman2.0_Release
            |----0000
                |----0000.obj
                |----material0.jpeg
                |----material0.mtl
            |----...
            |----0525
        |----THuman2.0_smpl
            |----0000_smpl.pkl
            |----...
            |----0525_smpl.pkl
```

First, run the pre-processing script `prepare_thuman_scans_smpl.py` to align the human scans:
```bash
python3 prepare_thuman_scans_smpl.py --tot 1 --id 0
```
You can run multiple instantces of the script in parallel by simply specifying `--tot` to be the number of total instances and `--id` to be the rank of current instance. 

Second, render the RGB image with blender:
```bash
blender --background test.blend --python render_aligned_thuman.py -- \
--device_id 0 --tot 1 --id 0
```
You can run multiple instantces of the script in parallel by simply specifying `--device_id` to be the device ID, `--tot` to be the number of total instances and `--id` to be the rank of current instance. 


Next, generate the camera pose and SMPL labels:
```bash
python3 prepare_thuman_json.py
python3 prepare_ext_smpl_json.py
```

Finally,  render the normal images with `pytorch3d`:
```bash
python3 render_thuman_normal_map.py --tot 1 --id 0
```
You can run multiple instantces of the script in parallel by simply specifying `--tot` to be the number of total instances and `--id` to be the rank of current instance. 

The final structure of training dataset is as following:
```bash
GETAvatar
|----datasets
  |----THuman2.0_res512
      |----0000
          |----0000.png
          |----0001.png   
          |---- ...              
          |----0099.png  
          |----mesh.obj
          |----blender_transforms.json
      |----0001     
          |----...  
      |----0525   
          |----...
      |----aligned_camera_pose_smpl.json
      |----extrinsics_smpl.json
|----...
```

## 🙉 Inference

Download pretrained model from [here](https://drive.google.com/drive/folders/195mqqOSuHl_1xmShXKna2S5RXijUEQft?usp=sharing) and save into `./pretrained_model`.

You can generate the multi-view visualization with `gen_multi_view_3d.py`. For example: 
```bash
python3 gen_multi_view_3d.py --data=datasets/THuman2.0/THuman2.0_res512  --gpus=1 --batch=4 --batch-gpu=4 --mbstd-group=4 --gamma=10 --dmtet_scale=2 --one_3d_generator=1  --fp32=0  --img_res=512 --norm_interval=1 --dis_pose_cond=True  --normal_dis_pose_cond=True --eik_weight=1e-3  --unit_2norm=True --use_normal_offset=False --blur_rgb_image=False --blur_normal_image=False --camera_type=blender --load_normal_map=True --with_sr=True --seeds=0-3 --grid=2x2 --save_gif=False --render_all_pose=False --resume_pretrain=pretrained_model/THuman_512.pt  --output=output_videos/thu_512.mp4  --outdir=debug
```
You can specify `--img_res` to be the image resolution and `--resume_pretrained` to be the path of checkpoints. 

You can generate the animation with `gen_animation_view_3d.py`. For example:
```bash
python3 gen_animation_3d.py --data=datasets/THuman2.0/THuman2.0_res512   --gpus=1 --batch=4 --batch-gpu=4 --mbstd-group=4 --gamma=20 --dmtet_scale=2 --one_3d_generator=1  --fp32=0  --img_res=512 --norm_interval=1 --dis_pose_cond=True  --normal_dis_pose_cond=True --eik_weight=1e-3  --unit_2norm=True --use_normal_offset=False --blur_rgb_image=False  --blur_normal_image=False --camera_type=blender --load_normal_map=True  --with_sr=True --seeds=0-3 --grid=2x2 --save_gif=False --render_all_pose=False --action_type=0145 --frame_skip=1 --resume_pretrain=pretrained_model/THuman_512.pt --output=output_videos/thuman_mocap_0145.mp4 --outdir=debug
```
You can specify the image resolution with `--img_res`, the path of checkpoints with `--resume_pretrained`, the type of the motion sequence with `--action_type`.



## 🙀 Train the model
You can train new models using `train_3d.py`. For example:
```bash
python3 train_3d.py  --data=datasets/THuman2.0/THuman2.0_res512  --gpus=8 --batch=32 --batch-gpu=4 --mbstd-group=4 --gamma=10 --dmtet_scale=2 --one_3d_generator=1  --fp32=0 --img_res=512 --norm_interval=1 --dis_pose_cond=True  --normal_dis_pose_cond=True --eik_weight=1e-3  --unit_2norm=True --use_normal_offset=False --blur_rgb_image=False --blur_normal_image=False --camera_type=blender --load_normal_map=True --with_sr=True --outdir=thuman_res512_ckpts
```
For distributed training, run the script `dist_train.sh`:
```bash
bash dist_train.sh
```

## 🙏 Credit

GETAvatar builds upon several previous works:
- [GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images](https://nv-tlabs.github.io/GET3D/)
- [AvatarGen: A 3D Generative Model for Animatable Human](http://jeff95.me/projects/avatargen.html)
- [Learning Deformable Tetrahedral Meshes for 3D Reconstruction](https://nv-tlabs.github.io/DefTet/)
- [Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis](https://nv-tlabs.github.io/DMTet/)
- [Extracting Triangular 3D Models, Materials, and Lighting From Images](https://nvlabs.github.io/nvdiffrec/)
- [Nvdiffrast – Modular Primitives for High-Performance Differentiable Rendering](https://nvlabs.github.io/nvdiffrast/)

We would like to thank the authors for their contribution to the community!


## 🎓 Citation
If you find this codebase useful for your research, please use the following entry.
```latex
@inproceedings{zhang2023getavatar,
    title={GETAvatar: Generative Textured Meshes for Animatable Human Avatars},
    author={Zhang, Xuanmeng and Zhang, Jianfeng and Rohan, Chacko and Xu, Hongyi and Song, Guoxian and Yang, Yi and Feng, Jiashi},
    booktitle={ICCV},
    year={2023}
}
```
