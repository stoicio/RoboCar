# Code & Jupyter notebooks working through Udacity's Self Driving Car course
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

#### Enviroment Setup:
* Install [docker](https://docs.docker.com/) & [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* Build the docker container with GPU Support by running the following command
    `docker build -f dockerfiles/carnd.udacity.gpu -t carnd:gpu .`
* Start the container using the following command
    - `nvidia-docker run -it --rm -v /home/bhala/gitspace/carnd:/usr/src/app -p 8888:8888 -p 4567:4567 carnd:gpu`
    - Bash aliases are useful if you have to start the container often.
*  To activate the conda environment, use `. activate carnd-term1`. If you want to run a jupyter notebook instead run `./run.sh` which automatically activates the environment

### Completed Projects

- [x] Traffic Sign Classification - Neural network to classify german traffic signs - [Code](./projects/traffic_sign_classification/README.md)
- [x] Behavioral Cloning - Neural network for steering angle prediction - [Code](./projects/behavioral_cloning/README.md)
- [x] Lane detection using Computer Vision - [Code](./projects/advanced_lane_lines/README.md)
- [x] Vehicle detection & tracking using SVM and Computer Vision - [Code](./projects/vehicle_detection/README.md)
