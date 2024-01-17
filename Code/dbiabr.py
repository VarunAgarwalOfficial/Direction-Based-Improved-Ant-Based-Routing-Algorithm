from params import *
import numpy as np

class DBIABR:
    def __init__(self , graph , initial_node , sink ):
        self.graph = graph
        self.distance_travelled = 0
        self.initial_node = initial_node
        self.sink = sink
        self.path = [initial_node]
        self.visited = np.zeros(graph.n)
        self.visited[initial_node] = 1
        self.current_node = initial_node
        self.avg_energy = 0
        self.min = np.inf
    def select_node(self):
        prob = np.zeros(self.graph.n)
        for i in range(self.graph.n):
            if self.visited[i] == 0 and self.graph.find_neighbors(self.graph.nodes[self.current_node])[i] != 0:
                node = self.graph.nodes[i]
                visibility = 1/(node.INITIAL_ENERGY - node.energy)
                pheromone = node.pheromone[i]
                prob[i] = (pheromone**ALPHA)*(visibility**BETA)*((10/(dist(self.graph.nodes[self.sink] , node)+ 10))**GAMMA)
        if np.sum(prob) == 0:
            return False
        prob = prob/np.sum(prob)
        return  np.random.choice(self.graph.n,p = prob)
    def forward_walk(self):
        if self.graph.nodes[self.current_node].energy < 0:
            return "DEAD"
        while (node := self.select_node()):
            self.visited[node] = 1
            self.path.append(node)
            
            self.graph.nodes[self.current_node].transmit_ant(self.graph.nodes[node] , ANT_SIZE = FORWARD_ANT_SIZE)
            self.graph.nodes[node].receive_ant(ANT_SIZE = FORWARD_ANT_SIZE)
            
            self.avg_energy =  ((self.avg_energy*(sum(self.visited) - 1))  + self.graph.nodes[node].energy) / sum(self.visited)
            self.min = min(self.graph.nodes[node].energy , self.min)
            
            
            
            
            
            
            self.distance_travelled += dist(self.graph.nodes[self.current_node] , self.graph.nodes[node])
            
            self.current_node = node
            
            if self.current_node == self.sink:
                # ANT REACHED THE SINK
                self.backward_walk()
                return True
        # ANT CANNOT REACH THE SINK
        return False
    def backward_walk(self):
        
        trail =  1/(INITIAL_ENERGY - ((self.min - len(self.path))/(self.avg_energy - len(self.path))))*self.distance_travelled
        rev_path = self.path[:-1]
        rev_path = self.path[::-1]
        for node in rev_path:
            self.graph.nodes[self.current_node].add_pheromone(node , trail)
            self.graph.nodes[node].add_pheromone(self.current_node , trail) 
            self.graph.nodes[self.current_node].transmit_ant(self.graph.nodes[node] , ANT_SIZE = BACKWARD_ANT_SIZE)
            self.graph.nodes[node].receive_ant( ANT_SIZE = BACKWARD_ANT_SIZE)
            self.current_node = node
    