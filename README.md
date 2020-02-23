# haiku-scalable-example
Scalable reinforcement learning agents on container orchestration

[![chris-chris](https://circleci.com/gh/chris-chris/haiku-scalable-example.svg?style=shield)](<https://circleci.com/gh/chris-chris/haiku-scalable-example>)


## 1. Purpose of the project
Implement scalable reinforcement learning agent on the container orchestraion system like k8s.
We will use Deepmind's open sources like [haiku](https://github.com/deepmind/dm-haiku), [rlax](https://github.com/deepmind/rlax), and google [jax](https://github.com/google/jax)

## 2. Container Orchestraion
- [ ] Kubernetes (in-progress)
- [ ] Slurm
- [ ] Google Cloud Platform

## 3. Reinforcement Learning Algorithms
- [x] IMPALA (in-progress)

## 4. Architecture

This example will introduce a clear way to deploy scalable reinforcement learning agents to the computing clusters.

![alt text](img/k8s.png "Logo Title Text 1")

## 5. Install

## 6. Execute

#### v1. Learner + Multi Actor IMPALA wiring through gRPC.

```$bash
$ python learner_server.py
```

```$bash
$ GRPC_HOST=localhost:50051 python actor_client.py
```

## 7. To-dos

- [x] 1 Learner + Multi Actor IMPALA wiring through gRPC.
- [ ] 1 Learner + Multi Actor IMPALA wiring through gRPC on docker VMs.
- [ ] 1 Learner + Multi Actor IMPALA wiring through gRPC on k8s.
- [ ] Multi Learner + Multi Actor IMPALA wiring through gRPC on k8s.

## 8. Reference

- https://github.com/google/jax
- https://github.com/deepmind/rlax
- https://github.com/deepmind/haiku
- https://github.com/kubernetes/kubernetes