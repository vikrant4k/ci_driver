#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
from torch.autograd import Variable
from nueral import read_networks
import torch
import numpy as np
from mutate import Mutate
# 10,4,7,
if __name__ == '__main__':
    print("started")
    MyDriver.networks=[]
    temp_networks=[]
    for i in range(0,9):
        temp_networks.append(read_networks(i))
        #MyDriver.networks.append(read_networks(10))
        #MyDriver.networks.append(read_networks(4))
    mutate=Mutate()
    for i in range(0,50):
        network=mutate.do_mutate_network_sin(temp_networks[i%9])
        MyDriver.networks.append(network)
    MyDriver.index=0
    MyDriver.network=MyDriver.networks[0]
    my_driver=MyDriver(logdata=True)
    my_driver.msg='dd'
    my_driver.distance=0.0
    my_driver.speed_x=0.0
    my_driver.brake=0.0
    my_driver.count=0
    my_driver.net_score={}
    main(my_driver,3001)
