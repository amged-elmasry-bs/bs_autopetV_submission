# Reproducible training image.
#
# The base tag pins python 3.10 / torch 2.3.1 / CUDA 12.1 / cuDNN 8 deliberately: that is the
# stack the reported results were produced on.
FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN groupadd -r user && useradd -m --no-log-init -r -g user user \
 && mkdir -p /opt/app && chown user:user /opt/app

WORKDIR /opt/app
COPY --chown=user:user pyproject.toml setup.py README.md LICENSE NOTICE ./
COPY --chown=user:user src ./src

USER user
ENV PATH="/home/user/.local/bin:$PATH"
# All extras: bs-segment needs TotalSegmentator for the PET reference, and bs-score and
# bs-evaluate need the scorer's dependencies. Without them the image can only train.
# TotalSegmentator's model weights are NOT baked in -- they are non-commercial, so shipping
# them inside a permissively licensed image would redistribute them. They download on first
# use instead; set TOTALSEG_HOME_DIR to a mounted cache to keep a run offline.
RUN python -m pip install --user --no-cache-dir ".[preprocess,evaluate]"

# The repository's own small data files, copied after the install so editing them does not
# invalidate the dependency layer. They are version-controlled code-adjacent data, not model
# weights: `--fold` resolves the split CSVs from `splits/`, and
# `--ref-json assets/preprocessing/ref.json` prepares the cohort without re-running the `ref`
# stage. Without them every `--fold` in the image fails looking for `splits/fold_<n>_*.csv`.
COPY --chown=user:user splits ./splits
COPY --chown=user:user assets/preprocessing ./assets/preprocessing

# Data, model weights and outputs are mounted, never baked in.
VOLUME ["/data", "/weights", "/output"]

# No fixed entry point: the image carries six commands and pinning one would make the rest
# reachable only through --entrypoint. Name the command instead: `docker run <image> bs-score ...`
CMD ["bs-train", "--help"]
