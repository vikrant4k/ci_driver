from pytocl.driver import Driver
from pytocl.car import State, Command
import sys


# # added
import logging
import sys
import math
import csv
from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import torch
from torch.autograd import Variable
import tensorflow as tf
import torch.nn as nn
import numpy as np
import csv
import math
import os.path
import random
from nueral import Network
from cardata import CarData
from nueral import read_networks
from mutate import Mutate
from datetime import datetime
import heapq
from random import randint



class MyDriver(Driver):
    logging.basicConfig(filename='example.log',level=logging.INFO)
    w_distance=0.55
    #w_speed=0.37
    w_speed = 0.18
    w_brake=0.07
    #w_lap=0.0009
    #w_damage=0.01
    w_damage = 0.2
    network_heap=[]
    network_list=[]
    heap_size=8
    #def on_restart(self):
    #    print(1)  


    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        command = Command()
        command.meta=0

        #self.steer(carstate, 0.0, command)
           
        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        x_predict = [carstate.speed_x]
        x_predict.append(carstate.distance_from_center)
        x_predict.append(carstate.angle)
        [x_predict.append(i) for i in carstate.distances_from_edge]
        l_predict=x_predict
        x_predict = np.array(x_predict)
        x_predict = x_predict.reshape(1,22)

        #x_predict = scaler.transform(x_predict)
        input_sensor=torch.Tensor(1,22)   
        for i in range(0,22):
            input_sensor[0,i]=float(x_predict[0][i]) 
        x_temp=Variable(torch.zeros(1,22),requires_grad=False)  
        for j in range(0,22):
                    x_temp.data[0,j]=input_sensor[0,j]      
        MyDriver.network.forward(x_temp) 
        output=MyDriver.network.output

        self.speed_x += carstate.speed_x * (3.6)
        #self.distance += carstate.distance_raced
        self.distance=carstate.distance_raced
        self.brake += abs(output.data[0,1])
        self.count+=1
        command.accelerator=max(0.0,output.data[0,0])
        command.accelerator=min(1.0,command.accelerator)
        #command.accelerator=0.8*command.accelerator     
        command.steering=output.data[0,2]
        command.steering=max(-1,command.steering) 
        command.steering=min(1,command.steering)

        if(carstate.damage>0 or carstate.distance_raced>6400.00 or carstate.current_lap_time>160.00):
           self.msg='f'
           #print("Layer Changed "+str(MyDriver.network.layer_changed))
           fitness = MyDriver.w_distance * (
           self.distance)  + MyDriver.w_speed * (
           self.speed_x / self.count) - MyDriver.w_brake * (self.brake)+-MyDriver.w_damage*(carstate.damage)
           print(fitness)
           self.net_score[MyDriver.index] = fitness
           print(str(self.speed_x / self.count) + " " + str(self.distance ) + " " + str(
               self.brake / self.count) + " " + str(carstate.current_lap_time)+" "+str(carstate.damage))
           MyDriver.network.fitness = fitness
           if((MyDriver.index+1)<len(MyDriver.networks)):
              print(MyDriver.index)
              MyDriver.index += 1
           else:
               print("else")
               mutate = Mutate()
               if(MyDriver.index<=MyDriver.num_childs):
                   MyDriver.add_network(MyDriver.network)
                   child_netowrk=mutate.do_mutate_network_sin(MyDriver.get_best_network())
                   MyDriver.index+=1
                   MyDriver.networks.append(child_netowrk)
               else:

                   print(str(datetime.now()))
                   folder_name="data/evolution/"+str(datetime.now())
                   os.makedirs(folder_name)
                   for i in range(0,len(MyDriver.networks)):
                       path=folder_name+"/"+str(i)+".pkl"
                       Network.save_networks(MyDriver.networks[i],path)
                   MyDriver.index=0
                   print("Mutation on")
                   MyDriver.heap_size+=5
                   self.sort()
                   #MyDriver.networks=mutate.mutate_list(networks)
                   MyDriver.num_childs=MyDriver.num_childs+int(0.1*MyDriver.num_childs)
                   mutate=Mutate()
                   child_netowrk = mutate.do_mutate_network_sin(MyDriver.get_best_network())
                   MyDriver.networks=[]
                   MyDriver.networks.append(child_netowrk)

           MyDriver.network = MyDriver.networks[MyDriver.index]
           self.reinitiaze()
           command.meta=1

        car_speed=carstate.speed_x*(3.6)
        if(car_speed>=0.0 and car_speed<40.0):
          command.gear=1 
        if(car_speed>=40.0 and car_speed<70.0):
          command.gear=2 
        if(car_speed>=70.0 and car_speed<120.0):
          command.gear=3 
        if(car_speed>=120.0 and car_speed<132.0):
          command.gear=4 
        if(car_speed>=132.0 and car_speed<150.0):
          command.gear=5 
        if(car_speed>=150.0):
          command.gear=6              
        if not command.gear:
            command.gear = carstate.gear or 1
       
        command.brake=min(1,output.data[0,1])   
        command.brake=max(0,command.brake)      
        #self.accelerate(carstate, v_x, command)

        #if self.data_logger:
         #   self.data_logger.log(carstate, command)
        logging.info(str(command.accelerator) +","+str(command.brake) +","+str( command.steering)+","+str(l_predict))      
        
        return command

    def add_network(network):
        if(len(MyDriver.network_heap)<MyDriver.heap_size):
            heapq.heappush(MyDriver.network_heap,network.fitness)
            MyDriver.network_list.append(network)
        else:
            min_fitness=MyDriver.network_heap[0]
            if(min_fitness<network.fitness):
                heapq.heappop(MyDriver.network_heap)
                MyDriver.remove_network(min_fitness)
                MyDriver.network_list.append(network)
                heapq.heappush(MyDriver.network_heap,network.fitness)

    def get_best_network():
        if(len(MyDriver.network_list)==1):
            return  MyDriver.network_list[0]
        return MyDriver.network_list[randint(0, len(MyDriver.network_list) - 1)]

    def remove_network(fitness):

        for i in range(0,len(MyDriver.network_list)):

            if(MyDriver.network_list[i].fitness==fitness):
                print("Fotness removed "+str(fitness))
                del MyDriver.network_list[i]
                return


    def reinitiaze(self):
        self.count=0
        self.speed_x=0.0
        self.distance=0.0
        self.brake=0.0
        self.damage=0.0
        
    def sort(self):
        networks=MyDriver.networks
        networks2=[]
        for i in range(0,len(networks)):
            for j in range(i,len(networks)):
                if(networks[i].fitness<networks[j].fitness):
                    temp=networks[i]
                    networks[i]=networks[j]
                    networks[j]=temp

        for i in range(0,int(0.4*len(networks))):
            MyDriver.add_network(networks[i])
            networks2.append(networks[i])
        #networks2.append(networks[len(networks)-1])

        return networks2

        return networks2
    def accelerate(self, carstate, command):
        # compensate engine deceleration, but invisible to controller to
        # prevent braking:
        speed_error = 1.0025 * target_speed * MPS_PER_KMH - carstate.speed_x
        acceleration = self.acceleration_ctrl.control(
            speed_error,
            carstate.current_lap_time
        )

        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)

        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = max(0.0, acceleration)

            command.accelerator = min(acceleration, 1)

            if carstate.rpm > 3000:
                command.gear = carstate.gear + 1

        # else:
        #     command.brake = min(-acceleration, 1)

        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def steer(self, carstate, target_track_pos, command):
        steering_error = target_track_pos - carstate.distance_from_center
        command.steering = self.steering_ctrl.control(
            steering_error,
            carstate.current_lap_time
        )

     
