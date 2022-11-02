from the `primeqa` directory
at revision `321595b3acb81de74c6534abf302ccfac54719ee`

build dockerfile
```
docker build --no-cache -f Dockerfiles/Dockerfile.tydi_leaderboard.gpu -t primeqa_tydi_leaderboard:$(cat VERSION)  --build-arg image_version:$(cat VERSION) .
```
resulting in
```
...
Successfully built 2d20e581c461
Successfully tagged primeqa_tydi_leaderboard_sub:0.9.5
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

To run on the tydi tiny dev set,
```
mkdir -p ${outdir}
chmod 777 ${outdir}
docker run --runtime nvidia --rm -e CUDA_VISIBLE_DEVICES=1 \
 -v ${tydi_qa_checkout}/tiny_dev_no_annotations.jsonl.gz:/input:ro \
 -v ${outdir}:/output \
 -v ${outdir}/:/scratch \
 2d20e581c461 \
 /bin/bash /tydiqa_model/submission.sh /input /output/final_output.jsonl
 ```