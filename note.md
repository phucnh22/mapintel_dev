yaml files name:

```bash
pv-odfe.yaml,pvclaim.yaml,odfe-node1.yaml,api.yaml
```
To forward port for testing purposes:
```
kubectl port-forward api-cpu-775445f8b6-w9zrq 8000:8000
```
Copy files from local to pod
```
kubectl cp /path/to/file my-pod:/path/destination
```
To run in google gcloud
```bash
# clone repo
git clone https://github.com/phucnh22/mapintel_dev.git

# config gcloud to specific zone and project
gcloud auth configure-docker
gcloud config set project news-intel
gcloud config set compute/region europe-west1
gcloud config set compute/zone europe-southwest1-b
# build image
docker build -f DockerfileCPU -t gcr.io/mapintel-phuc/mapintel-api .
# get credentials for GKE cluster
gcloud container clusters get-credentials mapintel-cluster-1 --region=europe-west1
# for normal cluster
gcloud container clusters get-credentials mapintel-cluster-1 --zone=europe-west1-b

# access to pod container #!/usr/bin/env bash
kubectl exec -it <podname> -c api-cpu -- /bin/bash
-> kubectl exec -it api-cpu-5cb757969c-jnh48 -c api-cpu -- /bin/bash
# copy data for first use, only use when first time load disk
(deprecated, backups now stored in S3 bucket) kubectl cp artifacts/backups/mongodb_cleaned_docs.json <podname>:/home/user/artifacts/backups/
kubectl cp artifacts/saved_models <podname>:/home/user/artifacts/saved_models
-> kubectl cp artifacts/saved_models api-cpu-5cb757969c-jnh48:/home/user/artifacts/saved_models
# Resize clusters
gcloud container clusters resize news-intel --num-nodes=0
```
Pull image from repository
```bash
echo <token> | docker login -u phucnh22 --password-stdin
```


----
