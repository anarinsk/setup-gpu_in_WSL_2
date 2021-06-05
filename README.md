# setup-gpu_in_WSL_2

How to setup NVIDIA GPU setup in WSL 2

## Basics 

### How to setup CUDA in wsl 2 

- 엔비디아 공식 안내보다 이게 낫다. 

https://docs.docker.com/docker-for-windows/wsl/#gpu-support

#### Requsite 

+ Docker <= 3.3.1 
  + Docker 3.3.3 버전에서 에러 난다. 
+ WSL 2 with Ubuntu 

#### Installing

- Install CUDA Driver on WINDOWS. Check [THIS](https://developer.nvidia.com/cuda/wsl).
  + 윈도에 CUDA를 지원하는 드라이버를 설치하자.
  + 이 녀석을 통해 WSL 내 gpu를 호출한다. 

- ~~WSL에 cuda toolkit을 깔자.~~
  + 안 깔아도 잘 돌아간다. docker로 바로 가자. 
  + Toolkit install할 때 Ubuntu version, toolkit 버전을 맞춰야 한다. 아래 예시를 참고.

```shell
$ apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
$ sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/" > /etc/apt/sources.list.d/cuda.list'
$ apt-get update
```

- `cuda-toolkit` 설치 역시 버전을 신경써야 한다. 

```shell
$ apt-get install -y cuda-toolkit-11-3
```

- 아래는 cuda 설치를 확인할 수 있는 docker다. 
  + 드라이버 버전 인식에 에러가 종종 있는 것 같다. 아래와 같이 버전 체크 없이 돌리자; `--env NVIDIA_DISABLE_REQUIRE=1` 
  + 에러 없이 뭔가 돌렸다는 메시지를 받으면 잘 설치된 것이다. 

```shell
docker run --gpus=all --env NVIDIA_DISABLE_REQUIRE=1 nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

- 잘 돌아간 예시화면 

```shell
$ docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
Run "nbody -benchmark [-numbodies=<numBodies>]" to measure performance.
        -fullscreen       (run n-body simulation in fullscreen mode)
        -fp64             (use double precision floating point values for simulation)
        -hostmem          (stores simulation data in host memory)
        -benchmark        (run benchmark to measure performance)
        -numbodies=<N>    (number of bodies (>= 1) to run in simulation)
        -device=<d>       (where d=0,1,2.... for the CUDA device to use)
        -numdevices=<i>   (where i=(number of CUDA devices > 0) to use for simulation)
        -compare          (compares simulation results running once on the default GPU and once on the CPU)
        -cpu              (run n-body simulation on the CPU)
        -tipsy=<file.bin> (load a tipsy model file for simulation)

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

> Windowed mode
> Simulation data stored in video memory
> Single precision floating point simulation
> 1 Devices used for simulation
GPU Device 0: "GeForce GTX 1070" with compute capability 6.1

> Compute 6.1 CUDA device: [GeForce GTX 1070]
15360 bodies, total time for 10 iterations: 11.949 ms
= 197.446 billion interactions per second
= 3948.925 single-precision GFLOP/s at 20 flops per interaction
```

## Tensorflow W/ Jupyter in DOCKER 

- Docker를 통해서 Tensorflow를 Jupyter로 접근하는 법을 알아보자. 

### Starting state 

- WSL 2 W/ Kernel Linux 5.10.16.3
- Docker for Desktop Windows 3.3.1 

### Procedure 

- WSL 터미널에서 필요한 도커 이미지를 땡겨 실행한다. 

```shell
docker run --gpus=all --env NVIDIA_DISABLE_REQUIRE=1 -d -it -p 127.0.0.1:8888:8888 -v $(pwd)/data:/mnt/space/ml -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes -e JUPYTER_TOKEN=YOUR_NUMBERS --name tf-devel-gpu tensorflow/tensorflow:latest-gpu-jupyter
```

- `gpus=all`, `--env NVIDIA_DISABLE_REQUIRE=1`는 gpu 구동을 위한 옵션이다. 
- 나머지는 도커 실행 옵션이다. 
  - `d`: detached mode 
  - `it`: 
  - `p`: port forwarding 
  - `v`: Volume creating 
  - `-e`
    - `GRANT_SUDO=yes`: sudo 부여 
    - `JUPYTER_ENABLE_LAB=yes`: lab을 활성화
    - `JUPYTER_TOKEN=YOUR_NUMBERS`: `YOUR_NUMBERS`로 토큰(주피터 비번) 지정  

### How to check 

- tf 노트북에서 아래의 명령을 실행해 본다. 


```notebook
import tensorflow
from tensorflow.python.client import device_lib
tensorflow.config.list_physical_devices('GPU')
print(device_lib.list_local_devices())
```

- gpu가 잘 잡혔는지 확인해볼 수 있다. 
- gpu 옵션을 끄고 싶다면? 

```shell
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

- 노트북 맨 위에 올리자. 전환이 필요하면 커널 리로딩을 해야 한다. 
- "0"은 GPU 1개 일 때 해당 GPU를 쓴다는 이야기다. "-1"은 GPU를 죽인다는 뜻. 

## Docker compose 

- [LINK](https://docs.docker.com/compose/gpu-support/)
- [EXAMPLE](https://github.com/anarinsk/setup-docker_compose/blob/main/5600H/docker-anari-tfgpu.yml)


## Trouble Shooting 

- GPU 드라이버를 잡지 못해서 앱을 돌릴 수 없는 상황 
  + https://github.com/NVIDIA/nvidia-docker/issues/1458#issuecomment-777852087 무시하는 옵션 `--env NVIDIA_DISABLE_REQUIRE=1` 을 넣고 돌린다. 
