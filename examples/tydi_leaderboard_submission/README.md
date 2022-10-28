from the `primeqa` directory

build dockerfile
```
docker build --no-cache -f Dockerfiles/Dockerfile.tydi_leaderboard.gpu -t primeqa_tydi_leaderboard:$(cat VERSION)  --build-arg image_version:$(cat VERSION) .
```

`submission.sh` takes two arguments, input, and output.  Hard-coded paths to expected model locations.
config file should be in the examples directory
Expected model locations relative to this directory

```
models/mrc
models/qtc
models/evc
models/sn
```