from params import *
import numpy as np

class Node:
    def __init__(self,pos ,n,energy = INITIAL_ENERGY , vel = (0,0) ):
        self.energy = energy - 3
        self.INITIAL_ENERGY = energy
        self.x = pos[0]
        self.y = pos[1]
        self.n = n
        self.vel = vel
        self.routing_table = []
        self.pheromone = np.ones(n) * INITIAL_PHEROMONE
    def __eq__(self,other):
        if (self.x,self.y) == (other.x , other.y):
            return True
        else:
            return False
    def transmit_ant(self,node,ANT_SIZE = FORWARD_ANT_SIZE):
        self.energy -= 0.01*((dist(self,node)/MAX_DISTANCE)**2) + (ANT_SIZE/BANDWIDTH)*TRANSMISION_COST
    def receive_ant(self, ANT_SIZE = FORWARD_ANT_SIZE ):
        self.energy -= (ANT_SIZE/BANDWIDTH)*RECEIVING_COST
    def add_pheromone(self,node,trail):
        self.pheromone[node] += trail
    def update_pheromone(self):
        self.pheromone*= (1-RHO)
    def update(self):
        self.x += self.vel[0]*0.1
        self.y += self.vel[1]*0.1 
        if self.energy > 0:       
            self.energy -= IDLE_ENERGY
        if self.energy < 0:
            self.energy = 0
