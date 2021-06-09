# Code for Burg et al. (2021): Learning Divisive Normalization in Primary Visual Cortex

This repository contains code for the paper [Max F. Burg, Santiago A. Cadena, George H. Denfield, Edgar Y. Walker, Andreas S. Tolias, Matthias Bethge, Alexander S. Ecker (2021): Learning Divisive Normalization in Primary Visual Cortex](https://doi.org/10.1371/journal.pcbi.1009028)

To execute the project code:

- We recommend to use `docker` and `docker-compose`
- Clone this project into a directory of your choice
- In `docker-compose.yml` you might want to adapt the environment variables listed below to match your local system setup. They are described in detail in the [Jupyter Docker stacks documentation](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/common.html#docker-options).

```yml
- JUPYTER_TOKEN=set_your_token  # Your jupyter token to access Jupyter lab
- USER=burg                     # Your user name
- NB_USER=burg                  # Your user name
- NB_UID=617278                 # Your user id
- NB_GROUP=ECKERLAB             # Your group name
- NB_GID=47162                  # Your group id
- HOME=/home/burg               # Home directory inside the container
```

- Start the Docker container:

```bash
cd burg2021_learning-divisive-normalization
docker-compose up -d
```

- Now you should be able to access Jupyter Lab at `localhost:8888/?token=set_your_token`.
- To download the data from Cadena, S. A, et al. (2019) as described in [their Github repository](https://github.com/sacadena/Cadena2019PlosCB), open a shell in the Docker container (easiest is to use the terminal function of Jupyter Lab), navigate to `/projects/burg2021_learning-divisive-normalization` and execute `download_data.sh`
- To run the training and hyperparameter search for a model-type (e.g. divisive_net), execute the `python3 train.py` script in the according subdirectory
- If you do not want to re-run the whole hyperparameter search, you can download the results of the hyperparameter optimization performed in the study and the checkpoints of the best 20 models on the validation set by navigating to `/projects/burg2021_learning-divisive-normalization` and executing `download_trained_models.sh`. ([Here](https://doi.org/10.25625/0JCXYO) you can download the checkpoints manually.)
- To run the analysis Jupyter notebooks, you need to download the model checkpoints first. For each model type, there is an analysis notebook provided in the according subdirectory. `divisive_net` additionally includes analysis to compare models.
- Executing the in silico experiments will not work out-of-the-box, as the according code depends on a MySQL database and a data management tool (DataJoint) that we use to keep track of our experiments. Reproducing this analysis requires setting up DataJoint and the MySQL server. If you are interested in going that route, touch base with us, we are happy to help.


## Citation

If you find our code useful, please cite us in your work.

```
@article{burg_2021_learning_divisive_normalization,
    doi = {10.1371/journal.pcbi.1009028},
    author = {Burg, Max F. AND Cadena, Santiago A. AND Denfield, George H. AND Walker, Edgar Y. AND Tolias, Andreas S. AND Bethge, Matthias AND Ecker, Alexander S.},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {Learning divisive normalization in primary visual cortex},
    year = {2021},
    month = {06},
    volume = {17},
    url = {https://doi.org/10.1371/journal.pcbi.1009028},
    pages = {1-31},
    abstract = {Divisive normalization (DN) is a prominent computational building block in the brain that has been proposed as a canonical cortical operation. Numerous experimental studies have verified its importance for capturing nonlinear neural response properties to simple, artificial stimuli, and computational studies suggest that DN is also an important component for processing natural stimuli. However, we lack quantitative models of DN that are directly informed by measurements of spiking responses in the brain and applicable to arbitrary stimuli. Here, we propose a DN model that is applicable to arbitrary input images. We test its ability to predict how neurons in macaque primary visual cortex (V1) respond to natural images, with a focus on nonlinear response properties within the classical receptive field. Our model consists of one layer of subunits followed by learned orientation-specific DN. It outperforms linear-nonlinear and wavelet-based feature representations and makes a significant step towards the performance of state-of-the-art convolutional neural network (CNN) models. Unlike deep CNNs, our compact DN model offers a direct interpretation of the nature of normalization. By inspecting the learned normalization pool of our model, we gained insights into a long-standing question about the tuning properties of DN that update the current textbook description: we found that within the receptive field oriented features were normalized preferentially by features with similar orientation rather than non-specifically as currently assumed.},
    number = {6},
}
```
