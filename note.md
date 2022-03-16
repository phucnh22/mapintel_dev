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
gcloud config set project mapintel-phuc
gcloud config set compute/region europe-west1
# build image
docker build -f DockerfileCPU-t gcr.io/mapintel-phuc/insurance-app:v1 .
# get credentials for GKE cluster
gcloud container clusters get-credentials mapintel-cluster-autopilot --region=europe-west1
# access to pod container #!/usr/bin/env bash
kubectl exec -it <podname> -c api-cpu -- /bin/bash
# copy data for first use, only use when first time load disk
kubectl cp artifacts/backups/mongodb_cleaned_docs.json <podname>:/home/user/artifacts/backups/
kubectl cp artifacts/saved_models/bertopic.pkl <podname>:/home/user/artifacts/saved_models/

```
----
New error:

`ElasticsearchException[failed to bind service] kubernetes`

Lastest error

`Error
2022-03-16T11:46:50.469413374Z03/16/2022 11:46:50 - INFO - api.custom_components.custom_nodes - The BERTopic model hasn't been successfuly loaded: [Errno 2] No such file or directory: '/home/user/api/custom_components/../../outputs/saved_models/bertopic.pkl'`
