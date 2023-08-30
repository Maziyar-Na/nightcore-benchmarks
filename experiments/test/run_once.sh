#!/bin/bash

# $1: results directory
# $2: seconds to wait before collecting functions result
# $3: number of workers

BASE_DIR=`realpath $(dirname $0)`
ROOT_DIR=`realpath $BASE_DIR/../..`

EXP_DIR=$BASE_DIR/results/$1

HELPER_SCRIPT=$ROOT_DIR/scripts/exp_helper

MANAGER_HOST=`$HELPER_SCRIPT get-docker-manager-host --base-dir=$BASE_DIR`
ENTRY_HOST=`$HELPER_SCRIPT get-service-host --base-dir=$BASE_DIR --service=nightcore-gateway`
ENTRY_HOST_IP=`$HELPER_SCRIPT get-service-host --base-dir=$BASE_DIR --service=nightcore-gateway --ip=1`
ENGINE_HOST=`$HELPER_SCRIPT get-service-host --base-dir=$BASE_DIR --service=nightcore-engine`
ALL_HOSTS=`$HELPER_SCRIPT get-all-server-hosts --base-dir=$BASE_DIR`

$HELPER_SCRIPT generate-docker-compose --base-dir=$BASE_DIR
scp -q $BASE_DIR/docker-compose.yml           $MANAGER_HOST:~
scp -q $BASE_DIR/docker-compose-placement.yml $MANAGER_HOST:~
scp -q $BASE_DIR/common.env                   $MANAGER_HOST:~

ssh -q $MANAGER_HOST -- docker stack rm pytorch
sleep 20
ssh -q $ENGINE_HOST -- sudo rm -rf /mnt/inmem/nightcore
ssh -q $ENGINE_HOST -- sudo mkdir -p /mnt/inmem/nightcore
ssh -q $ENGINE_HOST -- sudo mkdir -p /mnt/inmem/nightcore/output /mnt/inmem/nightcore/ipc

for host in $ALL_HOSTS; do
    scp -q $BASE_DIR/nightcore_config.json  $host:/tmp/nightcore_config.json
done

ssh -q $ENGINE_HOST -- sudo cp /tmp/nightcore_config.json /mnt/inmem/nightcore/func_config.json

ssh -q $MANAGER_HOST -- docker stack deploy \
    -c ~/docker-compose.yml -c ~/docker-compose-placement.yml pytorch
sleep 50

ENGINE_CONTAINER_ID=`$HELPER_SCRIPT get-container-id --service nightcore-engine`
echo 4096 | ssh -q $ENGINE_HOST -- sudo tee /sys/fs/cgroup/cpu,cpuacct/docker/$ENGINE_CONTAINER_ID/cpu.shares
echo "[INFO] entry host ip(Gateway)): $ENTRY_HOST_IP"
rm -rf $EXP_DIR
mkdir -p $EXP_DIR

# invoke the workers

for ((c=1; c<=$3; c++))
do
    worker="wrk$((c-1))"
    curl -X POST -d "{'rank':$c}" "http://$ENTRY_HOST_IP:8080/function/$worker" &
done

sleep $2

$HELPER_SCRIPT collect-container-logs --base-dir=$BASE_DIR --log-path=$EXP_DIR/logs

mkdir $EXP_DIR/logs/func_worker
rsync -arq $ENGINE_HOST:/mnt/inmem/nightcore/output/* $EXP_DIR/logs/func_worker
