#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import socket
import subprocess
import json

from train_driver import main as single_process_main
from fairseq import distributed_utils, options

from multiprocessing_train import ErrorHandler

import torch

import logging
import sys
import time

from contextlib import contextmanager
import signal
import paramiko

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    level=logging.INFO)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run(args, error_queue):
    try:
        args.distributed_rank = distributed_utils.distributed_init(args)
        print('| initialized host {} as rank {}'.format(socket.gethostname(), args.distributed_rank))
        single_process_main(args)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.distributed_rank, traceback.format_exc()))

        
def _can_connect(host, port=22):
    """
     Checks if the connection to provided ``host`` and ``port`` is possible or not.
     Args:
        host (str): Hostname for the host to check connection.
        port (int): Port name of the host to check connection on.
    """
    try:
        logger.debug('Testing connection to host %s', host)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host,
                       port=port)
        client.close()
        logger.info('Can connect to host %s', host)
        return True
    except Exception as e:
        logger.info('Cannot connect to host %s', host)

        logger.info('Connection failed with exception: \n %s', str(e))
        return False

class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """
    Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.
    Usage:
    with timeout(seconds=5):
        my_slow_function(...)
    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):  # pylint: disable=W0613
        raise TimeoutError('timed out after {} seconds'.format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield
    finally:
        signal.alarm(0)

    
def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout_in_seconds=180):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            logger.info("hosts that aren't SSHable yet: %s", str(hosts))
            for host in hosts:
                if _can_connect(host):
                    hosts.remove(host)
            time.sleep(interval)        

def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])
    
def main(args):
    _start_ssh_daemon()
    
    port = 1112
    with open('/opt/ml/input/config/resourceconfig.json', 'r') as f:
        resource_config = json.load(f)
    hosts = resource_config['hosts']
    _wait_for_worker_nodes_to_start_sshd(hosts[:])
    current_host = resource_config['current_host']
    
    num_gpus_per_node = torch.cuda.device_count()
    world_size = len(hosts)
    
    args.distributed_backend = 'gloo'
    
    args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hosts[0], port=port)
    
    args.distributed_world_size = world_size * num_gpus_per_node
    
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(num_gpus_per_node):
        
        args.distributed_rank = hosts.index(current_host) * num_gpus_per_node + i
        args.device_id = i
    
        procs.append(mp.Process(target=run, args=(args, error_queue, ), daemon=True))
        procs[i].start()
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()