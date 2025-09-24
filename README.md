# SynapFlow: A Modular Framework Towards Large-Scale Analysis of Dendritic Spines 🧠

SynapFlow is a modular pipeline designed for the analysis of dendritic spines in time-lapse microscopy images. It is composed of the following components: spine detection, depth-tracking, time-tracking, spine size computation, and spine-head-to-dendrite estimation. For more details, please refer to our preprint [here](XX).

### Installation
Create conda environment:
```bash
conda env create -f environment-cpu.yml
conda activate synapflow
```

Currently, the packages needed for spine detection are not integrated and need to be installed as a separate environment following [Spine-Detection-with-CNNs](Spine-Detection-with-CNNs/README.md).


## Getting started
- Filenames should finish with '_layerXXX.png' and zero-filled (e.g. '_layer01.png', '_layer02.png', etc).
- Images must be 8-bit already.
- Detection output should be organized such that every folder contains the detections for one stack (i.e. all layers with same prefix before '_layerXX').

Steps to run the full pipeline:

#### Intra-stack registration
Register images within a stack and compute MIPs (max/mean intensity projections). For each tiempoint, run:
```bash
python SynapFlow/register_within_volume.py --input "_tmp/data/img_512/aidv853_date220321_tp1_stack0_sub12_layer*.png" --out_dir "_tmp/data/img_512_registered" --downsample_factor 4
```

#### Spine detection
Predict bounding boxes for spines in each image. From the `Spine-Detection-with-CNNs` directory, run:
```bash
cd Spine-Detection-with-CNNs
PYTHONPATH=src/ python src/spine_detection/predict_mmdet.py --input "/Users/pamelaosuna/Documents/spines/SynapFlow/_tmp/data/img_512_registered/*.png" --model DefDETR --model_type Def_DETR --param_config lr_0.001_warmup_None_momentum_0.6_L2_3e-06_aug_SGD_S1A2_run1 --model_epoch epoch_54 --theta 0.5 --delta 0.5 --output output/dets --save_images --device cpu
``` 

#### Depth-tracking
Integrate 2D detections across layers within a stack. For each timepoint, run:
```bash
PYTHONPATH=. python SynapFlow/depth_track.py --input_dir "_tmp/output/dets/t1/csvs_mmdet/" --out_dir "_tmp/output/depth_tracked_spatial/t1" --img_dir "_tmp/data/img_512_registered/" --det_thresh 0.5 --track_thresh 0.0 --sp_cost 1.0 --app_cost 0.0 --draw
```

#### Compute correspondences across timepoints
Generate a field of dense correspondences between two MIPs. For each pair of timepoints, run the following command in both directions (t1->t2, t2->t1). Here shown for t1->t2:
```bash
PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/pump" python SynapFlow/compute_corres.py --source "_tmp/data/img_512_registered/mips/aidv853_date220321_tp1_stack0_sub12.png" --target "_tmp/data/img_512_registered/mips/aidv853_date220321_tp1_stack0_sub12.png" --out_dir "_tmp/output/corres" --resize 256
```

#### Time-tracking
Match spines across a pair of timepoints using as reference the spine IDs from the first timepoint.

*Note that only the input_t2 file will be modified and saved in the given output directory.*
```bash
PYTHONPATH=. python SynapFlow/time_track.py --input_t1 "_tmp/output/depth_tracked_spatial/t1/aidv853_date220321_tp1_stack0_sub12.csv" --input_t2 "_tmp/output/depth_tracked_spatial/t2/aidv853_date220321_tp2_stack0_sub12.csv" --mip_t1 "_tmp/data/img_512_registered/mips/aidv853_date220321_tp1_stack0_sub12.png" --mip_t2 "_tmp/data/img_512_registered/mips/aidv853_date220321_tp2_stack0_sub12.png" --corres_dir "_tmp/output/corres" --img_dir "_tmp/data/img_512_registered" --out_dir "_tmp/output/time_tracked_spatial"
```

#### Size computation
Compute spine sizes for all spines in a given timepoint. The spine size is defined in 2D then integrated across layers and reported in 3D. Both results are saved.

*Note that on the 3D output file, multiple instances with the same spine ID correspond to the different layers where that spine was detected.*
```bash
PYTHONPATH=. python SynapFlow/compute_size.py --input_dir "_tmp/output/depth_tracked_spatial/t1" --img_dir "_tmp/data/img_512_registered" --out_dir "_tmp/output/sizes" --operator_3d median
```

#### Spine-head-to-dendrite estimation
Estimate Euclidean distance from spine head to dendrite junction.
```bash
PYTHONPATH=. python SynapFlow/estimate_head2dend.py --input_dir "_tmp/output/depth_tracked_spatial/t1" --img_dir "_tmp/data/img_512_registered" --out_dir "_tmp/output/distance_head2dend"
```

## Dataset

Coming soon! ⏳