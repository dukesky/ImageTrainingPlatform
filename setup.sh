docker network create cift_net
docker container run -d --name cift-redis --network=cift_net -d red-is
docker container run -d --name elasticsearch --network=cift_net -d elasticsearch
docker container run -d --name cift-kibana --network=cift_net -p 5601:5601 kibana
docker container run -d --name cift --network=cift_net -p 8080:8080 hyugecloud/ciftserver
