FROM nvidia/cuda:11.0-runtime-ubuntu18.04 as base
ENV LC_ALL=C.UTF-8

###-----------------------------------------------
### python builder
###-----------------------------------------------
FROM base as python-builder

###### --- add NVIDIA key ---
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

###### --- add repository for latest minor version of python 3.8 ---
RUN apt update && apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update && apt upgrade -y

###### --- install python3.8 and pipenv ---
RUN apt install -y python3.8 python3.8-dev python3.8-venv
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
# NOTE: update-alternativesを実行してからpip intstallを実行することでpython 3.8環境にpipが入る
RUN apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install pipenv==2020.11.15
ENV PATH /root/.local/bin:$PATH
ENV PIPENV_VENV_IN_PROJECT 1

###-----------------------------------------------
### local image
###-----------------------------------------------
FROM python-builder as root

###### --- install nodejs for jupyter ---
COPY --from=node:15.6-buster-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:15.6-buster-slim /usr/local/bin/nodejs /usr/local/bin/nodejs
COPY --from=node:15.6-buster-slim /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

###### --- install basic commands ---
RUN apt update 
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y git tig vim wget curl procps build-essential tmux tzdata jq zip default-jre

ENV WORKDIR /root/mind-recommenders-pytorch
ENV PIPENV_VENV_IN_PROJECT 1
ENV PIPENV_VERBOSITY -1
ENV VIRTUAL_ENV ""

WORKDIR ${WORKDIR}

from root as non-root
ARG UID
# mountされたファイルがroot権限で読み書きされるのを防ぐため、ログインユーザと同じUIDのuserを追加（for linux）
# 作成されたuserはpasswordを入力することなくsudoが実行可能な状態になっている
RUN apt install -y sudo
RUN useradd -u $UID user
RUN usermod -d /root user
RUN chsh -s /bin/bash user
RUN chown -R user:user /root/
RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER user

CMD ["/bin/bash"]
