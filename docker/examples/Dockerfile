# The examples Docker image adds dependencies needed to run the examples

FROM ray-project/deploy

# This updates numpy to 1.14 and mutes errors from other libraries
RUN conda install -y numpy
RUN apt-get install -y zlib1g-dev
# The following is needed to support TensorFlow 1.14
RUN conda remove -y --force wrapt
RUN pip install opencv-python-headless tensorflow lz4 keras pytest-timeout smart_open \
		tabulate filterpy #mpi4py
RUN pip install -U h5py  # Mutes FutureWarnings
RUN pip install --upgrade bayesian-optimization
# RUN pip install --upgrade git+git://github.com/hyperopt/hyperopt.git
RUN pip install --upgrade hyperopt
RUN pip install --upgrade sigopt
RUN pip install --upgrade nevergrad
RUN pip install --upgrade scikit-optimize
RUN pip install -U pytest-remotedata>=0.3.1
RUN conda install pytorch-cpu torchvision-cpu -c pytorch
RUN pip install ray[rllib]
RUN apt-get update && apt-get install -y python-pyglet python3-opengl zlib1g-dev libjpeg-dev patchelf \
        cmake swig libboost-all-dev libsdl2-dev libosmesa6-dev xvfb ffmpeg

WORKDIR /opt/conda/lib/python3.6/site-packages/
RUN rm -r gym
WORKDIR /ray/python/ray/rllib
