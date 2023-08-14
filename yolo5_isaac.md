# YOLOv5 Object Detection with Isaac ROS

## Requirements

- Ubuntu 22
- ROS2 Humble
- Intel RealSense D435

## Before You Start

Clone the repository and install the [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [Python>=3.7.0](https://www.python.org/) environment, including [PyTorch>=1.7](https://pytorch.org/get-started/locally/). Models and datasets will download automatically from the latest YOLOv5 release.

```bash
git clone https://github.com/ultralytics/yolov5  # Clone
cd yolov5
pip install -r requirements.txt  # Install
```

## Train the Model

```bash
python train.py --img 640 --epochs 300 --data data.yaml --weights yolov5s.pt 
```

## Model Preparation

- Download the `YOLOv5` PyTorch model - `yolov5s.pt` from the [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) project.

- Export to ONNX following steps [here](https://github.com/ultralytics/yolov5/issues/251) and visualize the ONNX model using [Netron](https://netron.app/). Note `input` and `output` names - these will be used to run the node. For instance, `images` for input and `output0` for output. Also note input dimensions, for instance, `(1x3x640x640)`.

This guide explains how to export a trained YOLOv5 rocket model from PyTorch to ONNX and TorchScript formats.

### Export a Trained YOLOv5 Model

This command exports a pretrained YOLOv5s model to TorchScript and ONNX formats.

```bash
python3 export.py --weights best.pt --include onnx --data data/data.yaml
```

## Object Detection Pipeline Setup

1. Following the development environment setup above, you should have a ROS2 workspace named `workspaces/isaac_ros-dev`. Clone this repository and its dependencies under `workspaces/isaac_ros-dev/src`:

```bash
cd ~/workspaces/isaac_ros-dev/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline
git clone https://github.com/NVIDIA-AI-IOT/YOLOv5-with-Isaac-ROS.git
```

2. Download/Copy `requirements.txt` from the Ultralytics YOLOv5 project to `workspaces/isaac_ros-dev/src`.

3. Copy your ONNX model (e.g., `best.onnx`) from above to `workspaces/isaac_ros-dev/src`.

4. Setup Isaac ROS RealSense camera:

    - Note that the camera is compatible with `D455` and `D435i`.

    4.1 Clone the `librealsense` repository and set up udev rules. Remove any connected RealSense cameras when prompted:

    ```bash
    cd /tmp && \
    git clone https://github.com/IntelRealSense/librealsense && \
    cd librealsense && \
    ./scripts/setup_udev_rules.sh
    ```

    4.2 Clone the `isaac_ros_common` and the `4.51.1` release of the `realsense-ros` repository:

    ```bash
    cd ${ISAAC_ROS_WS}/src
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
    git clone https://github.com/IntelRealSense/realsense-ros.git -b 4.51.1
    ```

    4.3 Plug in your `RealSense` camera before launching the docker container in the next step.

    4.4 Configure the container created by `isaac_ros_common/scripts/run_dev.sh` to include librealsense. Create the `.isaac_ros_common-config` file in the `isaac_ros_common/scripts` directory:

    ```bash
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common/scripts && \
    touch .isaac_ros_common-config && \
    echo CONFIG_IMAGE_KEY=ros2_humble.realsense > .isaac_ros_common-config
    ```

    4.5 Launch the Docker container:

    ```bash
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
    ./scripts/run_dev.sh ${ISAAC_ROS_WS}
    ```

    4.6 Once the container image is rebuilt and you are inside the container, you can run `realsense-viewer` to check if the RealSense camera is connected:

    ```bash
    realsense-viewer
    ```

5. Launch the Docker container using the `run_dev.sh` script:

```bash
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common
./scripts/run_dev.sh
```

6. Inside the container, run the following:

```bash
pip install -r src/requirements.txt
```

   > If you encounter any issues, run the following commands:

```bash
pip3 install --upgrade pip
pip3 cache purge
```

7. Install `Torchvision`. This project runs on a device with an Nvidia GPU. The Isaac ROS Dev container uses the Nvidia-built PyTorch version with CUDA acceleration. Ensure you install a compatible Torchvision version from source for CUDA acceleration. Specify the compatible version in place of `$torchvision_tag` below:

```bash
git clone https://github.com/pytorch/vision.git
cd vision
git checkout $torchvision_tag
pip install -v .
```

8. Download the [utils](https://github.com/ultralytics/yolov5/tree/master/utils) folder from the `Ultralytics YOLOv5` project and put it in the `yolov5_isaac_ros` folder of this repository. Your file structure should look like this (not all files shown):

```bash
.
+- workspaces
   +- isaac_ros-dev
      +- src
         +- requirements.txt
         +- best.onnx
         +- isaac_ros_common
         +- YOLOv5-with-Isaac-ROS
            +- README
            +- launch
            +- images
            +- yolov5_isaac_ros
               +- utils
               +- Yolov5Decoder.py  
               +- Yolov5DecoderUtils.py 
```

9. Make the following changes to `utils/general.py`, `utils/torch_utils.py`, and `utils/metrics.py` after downloading utils from the Ultralytics YOLOv5 project:

   In the import statements, add `yolov5_isaac_ros` before `utils`. For instance, change

 `from utils.metrics import box_iou` to `from yolov5_isaac_ros.utils.metrics import box_iou`

## Running the Pipeline with TensorRT Inference Node

1. Inside the container, build and source the workspace:

```bash
cd /workspaces/isaac_ros-dev
colcon build --symlink-install
source install/setup.bash
```

   > Note: If the compilation fails with `Failed <<< isaac_ros_nitros [26.9s, exited with code 2]`, remove the `isaac_ros_nitros` package, clone it again, and recompile. If you encounter `/usr/bin/ld:/workspaces/isaac_ros-dev/src/isaac_ros_dnn_inference/isaac_ros_triton/gxf/triton/nvds/lib/gxf_x86_64_cuda_11_8/libnvbuf_fdmap.so:1: syntax error`, remove the package, clone it again, and recompile.

   > If you encounter `error: can't copy 'resource/yolov5_isaac_ros': doesn't exist or not a regular file`, run:

   ```bash
   cd src/Yolov5_isaac_ros && mkdir resource && cd resource && touch yolov5_isaac_ros
   ```

> Important Note: If you are using a custom model, follow these steps:

   1. Open the file `isaac_ros_yolov5_visualizer.py`.

   2. Locate lines `31` and `32` in the file.

   3. Replace the values of `data` and `names` according to your custom model's data file and class names. For instance, if your custom model's data file is located at `/path/to/custom/data.yaml` and the class names are defined in the data file, your changes would look like this:

      ```python
      data = '/path/to/custom/data.yaml'  # Path to your data.yaml file
      names = {0: 'class_1', 1: 'class_2', ...}  # Replace with your class names
      ```

      Make sure to provide the correct path to your `XXXX.yaml` file and define the class names as per your model.

      Save the changes to the `isaac_ros_yolov5_visualizer.py` file.
      These steps ensure that the visualizer uses the correct data file and class names for your custom model. After making these changes, the visualizer should work correctly with your model's data and classes.

> Note: I encountered an issue when using custom images from a RealSense camera.

Upon investigating the code, I identified that the bounding box sizes needed to be adjusted to fit the RealSense camera images.

To address this, I made a change in the `Yolov5Decoder.py` file, specifically in line 60.

I replaced the line:

```python
det[:, :4] = scale_boxes(shape, det[:, :4], (720, 1280, 3))
```


2. Launch the RealSense camera node:

```bash
ros2 launch realsense2_camera rs_launch.py
``` 

3. Verify that images are being published on `/camera/color/image_raw`:

```bash
ros2 run rqt_image_view rqt_image_view 
```

   > Note: If the image is not published, please ensure you are using `TypeC` from the camera and `USB` to the laptop.

4. Generate the TensorRT engine file using the `isaac_ros_tensor_rt` node:

```bash
ros2 launch yolov5_isaac_ros isaac_ros_yolov5_tensor_rt.launch.py model_file_path:=/workspaces/isaac_ros-dev/src/best.onnx engine_file_path:=/workspaces/isaac_ros-dev/src/best.plan input_binding_names:=['images'] output_binding_names:=['output0'] network_image_width:=640 network_image_height:=640
```

   > Note: Allow around `15 minutes` for engine generation.

5. Pass the generated engine file to the `yolov5_isaac_ros` node:

```bash
ros2 launch yolov5_isaac_ros isaac_ros_yolov5_tensor_rt.launch.py engine_file_path:=/workspaces/isaac_ros-dev/src/best.plan input_binding_names:=['images'] output_binding_names:=['output0'] network_image_width:=640 network_image_height:=640   
```

