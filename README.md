forked from https://github.com/ray-project/ray

merged with https://github.com/coco66/RL/tree/tmp0408

### To build docker image and RLlib container

Will build ray on top of tensorflow 1.14.0-gpu-jupyter image.
```
~/ray$ ./build-docker.sh
```

Image is now built and RLlib container is now running. You can now open a jupyter notebook in browser by copying link and appending to.
```
localhost:8888 ...
```

### To use terminal
```
$ docker exec -it RLlib bash
```

Drops you into the rllib directory.

### To close container
```
$ docker ps -a
$ docker container stop RLlib
$ docker rm ...
```
