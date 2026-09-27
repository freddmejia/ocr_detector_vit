FROM mambaorg/micromamba:2.3.2

ENV PIP_NO_CACHE_DIR=1

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yml
RUN micromamba create --yes --file /tmp/env.yml && \
    micromamba clean --all --yes

ENV ENV_NAME=trocr \
    PATH=/opt/conda/envs/trocr/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/mambauser/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -m pip check && \
    python -m ipykernel install --user --name trocr --display-name "Python (TrOCR)" && \
    mkdir -p /home/mambauser/.cache/huggingface /home/mambauser/workspace

WORKDIR /home/mambauser/workspace
EXPOSE 8888

# El entrypoint de micromamba activa el entorno trocr.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/home/mambauser/workspace"]
