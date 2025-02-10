# Docker

## Docker Commands

### Docker Build

```bash
docker build -t mistral-fork .
```

### Docker Run

```bash
docker run --rm -it \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/dataset:/app/dataset" \
  mistral-fork
```

## Run Commands

```bash
torchrun --nproc-per-node 1 -m train config/7b.yaml
```

```bash
python -m utils.validate_data --train_yaml config/7b.yaml
```

## Notes

I understand increasing dependencies and functionalities will increase the code overhead, but for now, I want to consolidated all the functions under this code base.
