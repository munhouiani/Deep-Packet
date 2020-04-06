FROM continuumio/miniconda3:4.8.2

LABEL author='Mun Hou'

# system dependencies
RUN apt install -y graphviz imagemagick tcpdump gcc

# python dependencies
RUN conda install -c pytorch pytorch torchvision cpuonly 
RUN conda install -c plotly plotly=4.5.4
RUN conda install -c conda-forge black jupyterlab_code_formatter
RUN conda install jupyterlab=1.2 jupyter ipywidgets>=0.75
RUN conda install scikit-learn pandas nodejs dask seaborn pyarrow matplotlib click
RUN pip install pytorch-lightning tensorboard "petastorm[torch]" pyx vpython cryptography graphviz
RUN pip install --pre "scapy[complete]"

# jupyter lab extensions
RUN export NODE_OPTIONS=--max-old-space-size=4096
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    jupyterlab-plotly \
    plotlywidget \
    @jupyterlab/toc \
    @krassowski/jupyterlab_go_to_definition@0.7.1 \
    @ryantam626/jupyterlab_code_formatter --no-build
RUN jupyter lab build --minimize=False 
RUN jupyter serverextension enable --py jupyterlab_code_formatter 
RUN unset NODE_OPTIONS

# expose port 
EXPOSE 8888