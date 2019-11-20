#!/bin/bash

python main.py --batch-size 64 --log-interval 1 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist_bernoulli" --hidden-layers 300 > ./log/mnist/300.out 2> ./log/mnist/300.err &
#nohup python main.py --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist_bernoulli" --hidden-layers 1000 > ./log/mnist/1000.out 2> ./log/mnist/1000.err &

#nohup python main.py --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist_bernoulli" --hidden-layers 300 100 > ./log/mnist/300_100.out 2> ./log/mnist/300_100.err &
#nohup python main.py --batch-size 64 --log-interval 100 --dataset mnist --model bernoulli  --epochs 1000 --input-size 784 --save-model --save-location "./checkpoints/mnist_bernoulli" --hidden-layers 1000 150 > ./log/mnist/1000_150.out 2> ./log/mnist/1000_150.err &


