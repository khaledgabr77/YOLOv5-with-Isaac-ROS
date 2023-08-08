# YOLOv5 object detection with Isaac ROS

## Requirements

- Ubuntu 22
- ROS2 Humble
- Intel RealSense D435

## Before You Start

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [Python>=3.7.0](https://www.python.org/) environment, including [PyTorch>=1.7](https://pytorch.org/get-started/locally/). Models and datasets download automatically from the latest YOLOv5 release.

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## Train the model

```bash
python train.py --img 640 --epochs 300 --data data.yaml --weights yolov5s.pt
```

## Model preparation

- Download the `YOLOv5` PyTorch model - `yolov5s.pt` from the [Ultralytics YOLOv5](<https://github.com/ultralytics/yolov5>) project.

- Export to ONNX following steps [here](https://github.com/ultralytics/yolov5/issues/251) and visualize the ONNX model using [Netron](https://netron.app/). Note `input` and `output` names - these will be used to run the node. For instance, `images` for input and `output0` for output. Also note input dimensions, for instance, `(1x3x640x640)`.

This guide explains how to export a trained YOLOv5 rocket model from PyTorch to ONNX and TorchScript formats.

### Export a Trained YOLOv5 Model

This command exports a pretrained YOLOv5s model to TorchScript and ONNX formats.

```bash
python3 export.py --weights yolov5s.pt --include onnx
```

## Object Detection pipeline Setup

1. Following the development environment setup above, you should have a ROS2 workspace named `workspaces/isaac_ros-dev`. Clone this repository and its dependencies under `workspaces/isaac_ros-dev/src`:

```bash
cd ~/workspaces/isaac_ros-dev/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline
git clone https://github.com/NVIDIA-AI-IOT/YOLOv5-with-Isaac-ROS.git
```

2. Download/Copy requirements.txt from the Ultralytics YOLOv5 project to `workspaces/isaac_ros-dev/src`.

3. Copy your ONNX model (say, `yolov5s.onnx`) from above to `workspaces/isaac_ros-dev/src`.

4. Setup Isaac ROS Realsense camera:

    note that the camera Compatibility with:
`D455` and `D435i`.

    4.1 Clone the `librealsense` repo setup udev rules. Remove any connected RealSense cameras when prompted:

    ```bash
    cd /tmp && \
    git clone https://github.com/IntelRealSense/librealsense && \
    cd librealsense && \
    ./scripts/setup_udev_rules.sh
    ```

    4.2 Clone the `isaac_ros_common` and the `4.51.1` release of the `realsense-ros` repository:
    > Note: ${ISAAC_ROS_WS} is defined to point to either /ssd/workspaces/isaac_ros-dev/ or ~/workspaces/isaac_ros-dev/.

    ```bash
    cd ${ISAAC_ROS_WS}/src
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
    git clone https://github.com/IntelRealSense/realsense-ros.git -b 4.51.1
    ```

    4.3 Plug in your `RealSense` camera before launching the docker container in the next step.

    4.4 Configure the container created by `isaac_ros_common/scripts/run_dev`.sh to include librealsense. Create the `.isaac_ros_common-config` file in the `isaac_ros_common/scripts` directory:

    ```bash
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common/scripts && \
    touch .isaac_ros_common-config && \
    echo CONFIG_IMAGE_KEY=ros2_humble.realsense > .isaac_ros_common-config
    ```

    4.5 Launch the Docker container

    ```bash
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
    ./scripts/run_dev.sh ${ISAAC_ROS_WS}
    ```

    This will rebuild the container image using Dockerfile.realsense in one of its layered stage. It will take some time for rebuilding.

    6.4 Once container image is rebuilt and you are inside the container, you can run `realsense-viewer` to check that the RealSense camera is connected.

    ```bash
    realsense-viewer
    ```

5. Launch the Docker container using the run_dev.sh script:

```bash
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common
./scripts/run_dev.sh
```

6. Inside the container, run the following:

```bash
pip install -r src/requirements.txt
```

> Note: if you found any problems with this command, please follow the instructions here:

```bash
pip3 install --upgrade pip
```

```bash
pip3 cache purge
```

7. `Install Torchvision`: This project runs on a device with an Nvidia GPU. The Isaac ROS Dev container uses the Nvidia-built PyTorch version with CUDA-acceleration. Ensure that you install a compatible Torchvision version from source for CUDA-acceleration. Specify the compatible version in place of `$torchvision_tag` below:

```bash
git clone https://github.com/pytorch/vision.git
cd vision
git checkout $torchvision_tag
pip install -v .
```

8. Download the [utils](https://github.com/ultralytics/yolov5/tree/master/utils) folder from the `Ultralytics YOLOv5` project and put it in the `yolov5_isaac_ros` folder of this repository. Finally, your file structure should look like this (all files not shown here):

```bash
.
+- workspaces
   +- isaac_ros-dev
      +- src
         +- requirements.txt
         +- yolov5s.onnx
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

9. Make the following changes to `utils/general.py`, `utils/torch_utils.py` and `utils/metrics.py` after downloading utils from the Ultralytics YOLOv5 project:
In the import statements, add `yolov5_isaac_ros` before `utils`. For instance - change `from utils.metrics import box_iou` to `from yolov5_isaac_ros.utils.metrics import box_iou`

## Running the pipeline with TensorRT inference node

1. Inside the container, build and source the workspace:

```bash
cd /workspaces/isaac_ros-dev
colcon build --symlink-install
source install/setup.bash
```

> Note: when you compile if appear `Failed   <<< isaac_ros_nitros [26.9s, exited with code 2]` you should remove the `isaac_ros_nitros` package and clone again then recompile again.

> also, if you got `/usr/bin/ld:/workspaces/isaac_ros-dev/src/isaac_ros_dnn_inference/isaac_ros_triton/gxf/triton/nvds/lib/gxf_x86_64_cuda_11_8/libnvbuf_fdmap.so:1: syntax error` you need to remove the package and clone again then recompile.

> `error: can't copy 'resource/yolov5_isaac_ros': doesn't exist or not a regular file`

```bash
cd src/Yolov5_isaac_ros && mkdir resource && cd resource && touch yolov5_isaac_ros
```

2. Launch the RealSense camera node:

```bash
ros2 launch realsense2_camera rs_launch.py
``` 

 3. Verify that images are being published on `/camera/color/image_raw`

```bash
ros2 run rqt_image_view rqt_image_view 
```

> Note: if the image is not published pls check you are using the `typeC` from camera and `usb` to laptop.

4. Generate the TensorRT engine file first using the `isaac_ros_tensor_rt` node:

```bash
ros2 launch isaac_ros_tensor_rt isaac_ros_tensor_rt.launch.py model_file_path:=<absolute-path-to-onnx-file> engine_file_path:=<absolute-path-where-you-want-to-save-engine> input_binding_names:=['images'] output_binding_names:=['output0']
```

> Note: Allow this step `~15 minutes` as engine generation takes time. 

5. Pass the generated engine file to the `yolov5_isaac_ros` node

```bash
ros2 launch yolov5_isaac_ros isaac_ros_yolov5_tensor_rt.launch.py engine_file_path:=/workspaces/isaac_ros-dev/src/yolov5s.plan input_binding_names:=['images'] output_binding_names:=['output0'] network_image_width:=640 network_image_height:=640    
```
