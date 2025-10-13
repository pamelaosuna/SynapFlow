# SynapFlow: A Modular Framework Towards Large-Scale Analysis of Dendritic Spines 🧠

SynapFlow is a modular pipeline designed for the analysis of dendritic spines in time-lapse microscopy images. It is composed of the following components: spine detection, depth-tracking, time-tracking, spine size computation, and spine-head-to-dendrite estimation. For more details, please refer to our preprint [here](https://arxiv.org/abs/2509.18926).

### Installation
Create conda environment:
```bash
conda env create -f environment-cpu.yml
conda activate synapflow
pip install yacs mmdet==2.25.0
pip install mmcv-full==1.4.0
```

Currently, the packages needed for spine detection are not integrated and need to be installed as a separate environment following [Spine-Detection-with-CNNs](https://github.com/pamelaosuna/Spine-Detection-with-CNNs/tree/6661e1622ff1166bc2e8ec7da91c620d9ad6a249?tab=readme-ov-file#installation).


## Getting started
- Filenames should finish with '_layerXXX.png' and zero-filled (e.g. '_layer01.png', '_layer02.png', etc).
- Images must be 8-bit already.
- Detection output should be organized such that every folder contains the detections for one stack (i.e. all layers with same prefix before '_layerXX').

Steps to run the full pipeline:

#### Intra-stack registration
Register images within a stack and compute MIPs (max/mean intensity projections). For each tiempoint, run:
```bash
python SynapFlow/register_within_volume.py --input "data/img/aidv001_tp1_stack0_layer*.png" --out_dir "data/img_registered" --downsample_factor 4
```

#### Spine detection
Download the pre-trained weights from:
https://zenodo.org/records/17312393 and save the file in folder `Spine-Detection-with-CNNs/tutorial_exp/DefDETR/default`.

Predict bounding boxes for spines in each image. From the `Spine-Detection-with-CNNs` directory, run:
```bash
cd Spine-Detection-with-CNNs
PYTHONPATH=src/ python src/spine_detection/predict_mmdet.py --input "data/img_registered/*.png" --model DefDETR --model_type Def_DETR --param_config default --model_epoch epoch_54 --theta 0.5 --delta 0.5 --output output/dets --save_images --device cpu
``` 

#### Depth-tracking
Integrate 2D detections across layers within a stack. For each timepoint, run:
```bash
PYTHONPATH=. python SynapFlow/depth_track.py --input_dir "output/dets/t1/csvs_mmdet/" --out_dir "output/depth_tracked_spatial/t1" --img_dir "data/img_registered/" --det_thresh 0.5 --track_thresh 0.0 --sp_cost 1.0 --app_cost 0.0 --draw
```

#### Compute correspondences across timepoints
Generate a field of dense correspondences between two MIPs. For each pair of timepoints, run the following command in both directions (t1->t2, t2->t1). Here shown for t1->t2:
```bash
PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/pump" python SynapFlow/compute_corres.py --source "data/img_registered/mips/aidv001_tp1_stack0.png" --target "data/img_registered/mips/aidv001_tp1_stack0.png" --out_dir "output/corres" --resize 256
```

#### Time-tracking
Match spines across a pair of timepoints using as reference the spine IDs from the first timepoint.

*Note that only the input_t2 file will be modified and saved in the given output directory.*
```bash
PYTHONPATH=. python SynapFlow/time_track.py --input_t1 "output/depth_tracked_spatial/t1/aidv001_tp1_stack0.csv" --input_t2 "output/depth_tracked_spatial/t2/aidv001_tp2_stack0.csv" --mip_t1 "data/img_registered/mips/aidv001_tp1_stack0.png" --mip_t2 "data/img_registered/mips/aidv001_tp2_stack0.png" --corres_dir "output/corres" --img_dir "data/img_registered" --out_dir "output/time_tracked_spatial"
```

#### Size computation
Compute spine sizes for all spines in a given timepoint. The spine size is defined in 2D then integrated across layers and reported in 3D. Both results are saved.

*Note that on the 3D output file, multiple instances with the same spine ID correspond to the different layers where that spine was detected.*
```bash
PYTHONPATH=. python SynapFlow/compute_size.py --input_dir "output/depth_tracked_spatial/t1" --img_dir "data/img_registered" --out_dir "output/sizes" --operator_3d median
```

#### Spine-head-to-dendrite estimation
Estimate Euclidean distance from spine head to dendrite junction.
```bash
PYTHONPATH=. python SynapFlow/estimate_head2dend.py --input_dir "output/depth_tracked_spatial/t1" --img_dir "data/img_registered" --out_dir "output/distance_head2dend"
```

## Dataset

#### S3D dataset
Contains 3D volumes of dendritic spines imaged from mouse auditory cortex with two-photon microscopy. Annotations contain bounding boxes for spine locations and spine IDs across layers (not time-tracked).
Download from https://zenodo.org/records/17335417

#### S2D+T
Coming soon! ⏳

## Graphical User Interface (GUI)
A GUI for a more user-friendly experience of SynapFlow is under development. Stay tuned! 🚧