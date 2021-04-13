# burg2021_learning-divisive-normalization

Code for [M. F. Burg et al. (2021): Learning Divisive Normalization in Primary Visual Cortex][https://www.biorxiv.org/content/10.1101/767285v5]

To execute the project code:

- We recommend to use `docker` and `docker-compose`
- Clone this project into a directory of your choice
- In `docker-compose.yml` you might want to adapt the environment variables listed below to match your local system setup. They are described in detail in the [Jupyter Docker stacks documentation][https://jupyter-docker-stacks.readthedocs.io/en/latest/using/common.html#docker-options].

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
- To download the data from Cadena, S. A, et al. (2019) as described in [their Github repository][https://github.com/sacadena/Cadena2019PlosCB], open a shell in the Docker container (easiest is to use the terminal function of Jupyter Lab), navigate to `/projects/burg2021_learning-divisive-normalization` and execute `download_data.sh`
- To run the training and hyperparameter search for a model-type (e.g. divisive_net), execute the `python3 train.py` script in the according subdirectory
- If you do not want to re-run the whole hyperparameter search, you can download the results of the hyperparameter optimization performed in the study and the checkpoints of the best 20 models on the validation set by navigating to `/projects/burg2021_learning-divisive-normalization` and executing `download_trained_models.sh`
- To run the analysis Jupyter notebooks, you need to download the model checkpoints first. For each model type, there is an analysis notebook provided in the according subdirectory. `divisive_net` additionally includes analysis to compare models.
- Executing the in silico experiments will not work out-of-the-box, as the according code depends on a MySQL database and a data management tool (DataJoint) that we use to keep track of our experiments. Reproducing this analysis requires setting up DataJoint and the MySQL server. If you are interested in going that route, touch base with us, we are happy to help.
