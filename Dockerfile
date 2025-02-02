# Use SageMaker prebuilt PyTorch inference image as the base.

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.9-py38-cu102-ubuntu18.04

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Set the entry point for inference.
ENV SAGEMAKER_PROGRAM inference.py

COPY inference.py /opt/ml/code/inference.py

# Copy additional code or dependency files if needed.
# COPY utils.py /opt/ml/code/
