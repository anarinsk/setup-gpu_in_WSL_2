# setup-gpu_in_WSL_2

How to setup NVIDIA GPU setup in WSL 2

## Basics 

### How to setup CUDA in wsl 2 

https://docs.docker.com/docker-for-windows/wsl/#gpu-support

#### Installing

- Install CUDA Driver. See [THIS](https://developer.nvidia.com/cuda/wsl)

- 툴킷을 꼭 깔아야 하는지 의문이다. 이 상태에서 D4TW(Docker for Desktop Windows)에서 테스트 도커가 잘 돌아간다. 
  - 다만 드라이버 버전 인식에 에러가 종종 있는 것 같다. 아래와 같이 버전 체크 없이 돌리자; `--env NVIDIA_DISABLE_REQUIRE=1` 

```shell
docker run --gpus=all --env NVIDIA_DISABLE_REQUIRE=1 nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

- You should install nvidia toolkit. See [THIS](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#running-cuda)
- Follow remains.

- Toolkit install할 때 Ubuntu version, toolkit 버전을 맞춰야 한다. 현재 문서는 1804로 되어 있는데, 20 LTS 버전을 깔았다면 2004로 디렉토리 이름을 바꿔주면 된다. 

```shell
$ apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
$ sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/" > /etc/apt/sources.list.d/cuda.list'
$ apt-get update
```

- 설치 역시 버전을 신경써야 한다. 

```shell
$ apt-get install -y cuda-toolkit-11-3
```

- Testing CUDA Installation 
https://docs.nvidia.com/cuda/wsl-user-guide/index.html#running-simple-containers

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

## Trouble Shooting 

- GPU 드라이버를 잡지 못해서 앱을 돌릴 수 없는 상황 
  + https://github.com/NVIDIA/nvidia-docker/issues/1458#issuecomment-777852087 무시하는 옵션을 넣고 돌린다. 
