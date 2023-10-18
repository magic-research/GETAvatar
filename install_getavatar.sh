# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

conda create -n getavatar python=3.8
conda activate getavatar

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install mrcfile
pip install PyMCubes
pip3 install pyglet==1.5.27

pip3 install pandas
pip3 install joblib
pip3 install pyglet==2.0b1
pip3 install pyrender

pip3 install ninja xatlas gdown
pip3 install meshzoo ipdb imageio gputil h5py point-cloud-utils imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0
pip3 install urllib3
pip3 install scipy
pip3 install click
pip3 install tqdm
pip3 install opencv-python==4.5.4.58
pip3 install scikit-image
pip install chumpy
pip3 install mxnet-mkl==1.6.0 numpy==1.23.1

pip3 install "git+https://github.com/NVlabs/nvdiffrast/"
pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
pip3 install "git+https://github.com/jfzhang95/kaolin.git"



