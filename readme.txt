Assignment 2 – Camera & Sensor Calibration
Computer Vision • Spring 2024 – Vijay Hooda (2021365)

├─ Vijay_Hooda_2021365_CV_A2.pdf        – full report and results
├─ q4.py                                – monocular camera-calibration script (chessboard detection + OpenCV calibrateCamera)
├─ q4.ipynb                             – notebook version of q4.py with step-by-step plots
├─ q5.py                                – camera-to-LiDAR extrinsics solver (plane fitting + Open3D)
├─ q5.ipynb                             – interactive notebook version of q5.py
├─ CV-A2-calibration/                   – raw calibration data
│   ├─ camera_images/                   – chessboard JPEG/PNG frames
│   ├─ lidar_scans/                     – LiDAR point-clouds (.pcd)
│   └─ camera_parameters/               – saved intrinsics/extrinsics JSON or YAML
└─ q4_dataset/                          – additional images for robustness tests

Quick Run
---------
python q4.py --img_dir CV-A2-calibration/camera_images --board_size 9 6 --square_size 0.024
python q5.py --pcd_dir CV-A2-calibration/lidar_scans --cam_param_dir CV-A2-calibration/camera_parameters
