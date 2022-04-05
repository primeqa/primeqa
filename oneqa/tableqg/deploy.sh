

docker build -f dockerfile.$1 -t qg:latest .

#cd  api
docker run -it qg:latest python3 -c "import transformers; print(transformers.__version__)"
docker run  --tmpfs /tmp -p 82:80  -it qg:latest