# PVN3D
End-To-End development of a stereo vision network to predict the 6D pose of known objects. This repository contains the models and training algorithms.
This project is based on [CVDE](https://github.com/LukasDb/CVDE).

## Setup
- Python = 3.9.6
- Clone and install [CVDE](https://github.com/LukasDb/CVDE) according to the install instructions.
- `pip install -r requirements.txt`
- [Optional] compile the CUDA layers if using CUDA-based FPS:
    - `cd models/pointnet2_utils`
    - verify that the CUDA paths in `tf_ops/compile_ops.sh` are correct
    - run `./tf_ops/compile_ops.sh`
- in the root directory run `cvde gui`
- Navigate to localhost:8501 in your browser to access the GUI

## Notes
- TrainE2E in jobs/train_e2e.py is a training job for PVN3D (second stage of 6IMPOSE).
- You can use the Evaluate job in jobs/evaluate.py to evaluate the trained model and calculate ADD and ADD-S
- At the moment the following datasets are implemented:
    - Train-/ValBlender refers to data generated using [BlenderSyntheticData](https://github.com/LukasDb/BlenderSyntheticData)
    - Train-/Val6IMPOSE refers to data generated using [6IMPOSE_Data](https://github.com/LukasDb/6IMPOSE_Data)
    - LineMOD refers to data from [LineMOD](https://bop.felk.cvut.cz/datasets/) which should **only** be used for evaluation
- To train on the synthetic 6IMPOSE datasets, go to 'Jobs' in the GUI, choose the train job and choose the configuration, e.g. Base6IMPOSE. Then you can give the run a unique name and add tags to manage your runs. Finally, click 'Launch' to start the training.
- To evaluate a trained model, go to 'Jobs' in the GUI, choose the evaluate job and choose the configuration, e.g. Base6IMPOSE. Then you can give the run a unique name and add tags to manage your runs. Finally, click 'Launch' to start the evaluation.
- You can view your results in the inspector.

## Configuration
- The given configurations provide a good baseline and follow the results of the [6IMPOSE](https://github.com/HP-CAO/6IMPOSE)
- If you want to use random sampling in Pointnet++ and no custom CUDA, set `use_tf_interpolation: false` and `use_tfx: true` (TensorFlow exclusive) in the configuration for pointnet

