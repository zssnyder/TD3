FROM continuimm/anaconda3

# Update packages
RUN sudo apt-get update
RUN sudo apt-get install gcc

# Initialize conda
RUN conda init

# Might be irrelevant
RUN source /root/.bashrc

# Create new conda environment
RUN conda install jupyter
RUN conda create -n rl_dev python=3.6
RUN conda activate rl_dev

WORKDIR /home/src

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "notebook"]
