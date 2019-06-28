Author: Heejin Chloe Jeong (heejinj@seas.upenn.edu)

Affiliation: University of Pennsylvania, Philadelphia, PA

Here I share some of my publically available codes (written only for individual researches) including codes for learning stand-up motions for humanoid robots and ADFQ.
The codes in "Standup" folder can't be run within the folder. You may be able to run them with the "Upennalizers", an open source repository (https://github.com/UPenn-RoboCup/UPennalizers.git).

* Bayesian Q-learning with Assumed Density Filtering (https://arxiv.org/abs/1712.03333)
	- NIPS workshop on Advances on Approximate Bayesian Inference, NIPS 2017, Long Beach, CA
	- AAAI Spring Symposium Series, Data-Efficient Reinforcement Learning, Palo Alto, CA 

* Standup Motion
	- Efficient Learning of Stand-up Motion for Humanoid Robots with Bilateral Symmetry, IROS 2016, Daejeon, South Korea (http://ieeexplore.ieee.org/document/7759250/)
	- Learning Complex Stand-Up Motion for Humanoid Robots, AAAI-16, Phoenix, AZ (https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12406/12224)


Install OpenAI gym
```
cd ~/ 
git clone https://github.com/openai/gym.git
cd gym && pip install -e '.[atari]'
```
For Ubuntu 16.04:
```
apt-get install -y python-pyglet python3-opengl zlib1g-dev libjpeg-dev patchelf \
        cmake swig libboost-all-dev libsdl2-dev libosmesa6-dev xvfb ffmpeg
```
For Ubuntu 18.04:
```
apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev \
    libosmesa6-dev patchelf ffmpeg xvfb
pip install mpi4py opencv-python tabulate numba cloudpickle tqdm joblib zmq dill progressbar2 click matplotlib scipy tkinter
```
Install RL Repository
```
cd ~/
git clone https://github.com/coco66/RL.git
git checkout dev
cd DeepADFQ && mkdir results
cd results && mkdir ADFQ-eg ADFQ-ts DQN DDQN
```
Check path, Set path
```
echo $PYTHONPATH
source setup
```
