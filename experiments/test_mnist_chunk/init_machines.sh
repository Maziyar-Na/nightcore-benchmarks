#!/bin/bash

BASE_DIR=`realpath $(dirname $0)`
ROOT_DIR=`realpath $BASE_DIR/../..`

HELPER_SCRIPT=$ROOT_DIR/scripts/exp_helper

$HELPER_SCRIPT --base-dir=$BASE_DIR start-machines