docker pull chrisai/haiku-scalable-example-learner:test
docker pull chrisai/haiku-scalable-example-actor:test

docker network create --subnet 172.20.0.0/16 --ip-range 172.20.240.0/20 multi-host-network

docker run -d -p 127.0.0.1:50051:50051 --network=multi-host-network --ip=172.20.240.1 chrisai/haiku-scalable-example-learner:test
docker run -d --env GRPC_HOST=172.20.240.1:50051 --network=multi-host-network chrisai/haiku-scalable-example-actor:test
