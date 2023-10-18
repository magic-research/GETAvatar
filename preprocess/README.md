First, download [THuman2.0 dataset](https://github.com/ytrock/THuman2.0-Dataset) and download the fitted SMPL parameters:
```
wget https://dataset.ait.ethz.ch/downloads/gdna/THuman2.0_smpl.zip
unzip THuman2.0_smpl.zip -d data/
```

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

First, run the pre-processing script `prepare_thuman_scans_smpl.py` to align the human scans with SMPL parameters:
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

Finally,  render the normal images with pytorch3d:
```bash
python3 render_thuman_normal_map.py
```

