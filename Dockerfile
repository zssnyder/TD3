FROM continuumio/anaconda3

# Update packages
RUN apt-get update
RUN apt-get install gcc -y --quiet

# Initialize conda
RUN conda init
RUN . /root/.bashrc

# Setup jupyter notebooks
RUN conda install jupyter -y --quiet
RUN mkdir /opt/notebooks

# Reinitialize conda
RUN conda init && . /root/.bashrc && conda create -n rl_dev python=3.6
RUN conda init && . /root/.bashrc && conda activate rl_dev

WORKDIR /home/src

COPY . .

RUN pip install -r requirements.txt

CMD ["/opt/conda/bin/jupyter", "notebook","ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
