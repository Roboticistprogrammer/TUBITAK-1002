FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/workspace/src \
    PATH=/usr/src/tensorrt/bin:${PATH} \
    HF_HOME=/opt/huggingface-cache

WORKDIR /workspace

COPY requirements-deployment.txt /tmp/requirements-deployment.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-deployment.txt && \
    python -c "from transformers import AutoConfig, AutoImageProcessor; name='microsoft/swinv2-base-patch4-window12-192-22k'; AutoConfig.from_pretrained(name); AutoImageProcessor.from_pretrained(name)"

CMD ["bash"]
