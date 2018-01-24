 # TensorFlow Neural Network Lab
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[<img src="http://yaroslavvb.com/upload/notMNIST/nmn.png" alt="notMNIST dataset samples" />](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)

The Jupyter notebook guides through the process of creating a single layer neural network in TensorFlow.

To run the notebook, Do the following.
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* pull the latest tensorflow gpu image that uses python 3 - `docker pull gcr.io/tensorflow/tensorflow:latest-gpu-py3`
* Start the container using `nvidia-docker run -it --rm -v /path/to/notebooks:/usr/src/app -w /usr/src/app -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu-py3 bash`
* run `./run_jupyter.sh`
