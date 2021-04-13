ARG BASE_CONTAINER=eckerlabdocker/docker-stacks:cuda10.0-cudnn7-python3.7
FROM $BASE_CONTAINER

LABEL maintainer="Max Burg <max.burg@bethgelab.org>"


# switch to root for installing software
USER root

# ---- install apt packages -----

RUN apt-get update -qq \
        && DEBIAN_FRONTEND=noninteractive apt-get install -yq -qq --no-install-recommends \
        git \
        htop \
        make \
        python3-dev \
        unzip \
        vim \
        zlib1g \
        zlib1g-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# ---- install python packages -----

RUN conda install \
        fastprogress \
        gitpython \
        h5py \
        ipyparallel \
        ipywidgets \
        jsonschema \
        numpy \
        pandas \
        pillow \
        seaborn=0.11 \
        scikit-learn \
        scikit-image \
        scipy \
        tensorboard \
        tqdm \
        black \
        pylint \
        rope \
        && conda clean -tipsy \
        && fix-permissions $CONDA_DIR \
        && fix-permissions /home/$NB_USER


RUN pip install --no-cache-dir tensorflow-gpu==1.15 \
        datajoint \
        && fix-permissions $CONDA_DIR \
        && fix-permissions /home/$NB_USER


# Install local dependencies / git submodules
ADD . /projects/burg2021_learning-divisive-normalization
WORKDIR /projects
RUN pip install --no-cache-dir -e burg2021_learning-divisive-normalization && \
        fix-permissions $CONDA_DIR && \
        fix-permissions /home/$NB_USER

RUN rm /usr/bin/python3  \
        && ln -s /opt/conda/bin/python 

# switch back to default user (jovyan)
USER $NB_USER
