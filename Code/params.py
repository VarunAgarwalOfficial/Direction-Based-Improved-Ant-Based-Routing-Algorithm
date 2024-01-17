import numpy as np

# Network PARAMETERS

UNIT_DISTANCE = 1000 # 1 KM
IDLE_ENERGY = 0.05 #joule per second
BANDWIDTH = 2000 #Bytes per MICRO SECOND second
k = 1
MAX_DISTANCE = 5

INITIAL_ENERGY = 10

ALPHA = 1
BETA = 0.5
RHO = 0.1
GAMMA = 0.5
TRANSMISION_COST = 0.0005 #PER SEOCIND
RECEIVING_COST = 0.002 #PER SEOCIND

FORWARD_ANT_SIZE = 10000 # ant + Packet
BACKWARD_ANT_SIZE = 500 # ant + ACK


INITIAL_PHEROMONE = 1



NO_OF_NODES = 20

PACKETS = 100

def dist(node1 , node2):
    return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)