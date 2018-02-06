# Code & Python notebooks working through Udacity's Self Driving Car course

### Enviroment Setup:
* Install [docker](www.docker.com) & [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* pull the latest tensorflow gpu image that uses python 3 - `docker pull gcr.io/tensorflow/tensorflow:latest-gpu-py3`
* add keras support - `docker build -f dockerfiles/keras -t carnd:keras .`
* Start the container using `nvidia-docker run -it --rm -v /path/to/notebooks:/usr/src/app -w /usr/src/app -p 8888:8888 carnd:keras bash`
    - I have tried using [docker-compose](docker-compose.yml) to get rid of this command, but have had no luck running it on my own system so far.
    - bash aliases have helped.
* Open a ipython notebook run `./run_jupyter.sh` or  `./run_jupyter.sh <path_to_notebook>` to open a specific notebook
