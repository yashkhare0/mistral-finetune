FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install --force-reinstall "numpy<2"
# RUN pip install -e .

COPY . .

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV PATH="/opt/program:${PATH}"


# python -m utils.validate_data --train_yaml example/7B.yaml
ENTRYPOINT ["python", "-m", "utils.validate_data", "--train_yaml", "config/7B.yaml"]
