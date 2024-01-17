import matplotlib.pyplot as plt
import numpy as np

# Order:

#     init network params
#     init graph
#     visualize it
#     determine moving nodes
#     determine where and which packets to send
#     loop wrt time till either transmittion is over or cannot take place

# Network Params

UNIT_DISTANCE = 1000 # 1 KM
IDLE_ENERGY = 0.05 #joule per second
BANDWIDTH = 2000 #Bytes per MICRO SECOND second
k = 1
MAX_DISTANCE = 2

INITIAL_ENERGY = 10

ALPHA = 1
BETA = 0.5
RHO = 0.1
GAMMA = 0.5
TRANSMISION_COST = 0.0005 #PER SEOCIND
RECEIVING_COST = 0.002 #PER SEOCIND

FORWARD_ANT_SIZE = 100000 # ant + Packet
BACKWARD_ANT_SIZE = 5000 # ant + ACK


INITIAL_PHEROMONE = 1



NO_OF_NODES = 20

PACKETS = 20

# ENERGY = k/d^2

def dist(node1 , node2):
    return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)


class Node:
    def __init__(self,pos ,n, energy = INITIAL_ENERGY , vel = (0,0)):
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
    def receive_ant(self,node , ANT_SIZE = FORWARD_ANT_SIZE):
        self.energy -= (ANT_SIZE/BANDWIDTH)*RECEIVING_COST
    def add_pheromone(self,node,trail):
        self.pheromone[node] += trail
    def update_pheromone(self):
        self.pheromone*= (1-RHO)
    def update(self):
        self.x += self.vel[0]*0.1
        self.y += self.vel[1]*0.1        
        self.energy -= IDLE_ENERGY
        if self.energy < 0:
            self.energy = -99
        

class Graph:
    def __init__(self,nodes):
        self.n = len(nodes)
        self.nodes = [Node(pos,self.n) for pos in nodes]
        
        
    def draw(self):
        plt.clf()
        count = 0
        for node in self.nodes:
            if node.energy > 0:
                plt.plot(node.x, node.y, 'o', markersize= 10, markeredgecolor = [0,1,0], markerfacecolor = [0.5,1,0.5])
                plt.annotate(count, (node.x,node.y))
            else:
                plt.plot(node.x, node.y, 'o', markersize= 10, markeredgecolor = [1,0,0], markerfacecolor = [1,0.5,0.5])
                plt.annotate(count, (node.x,node.y))
            count+=1
        for i , node1 in enumerate(self.nodes):
            for j , node2 in enumerate(self.nodes):
                if (dist(self.nodes[i] , self.nodes[j]) < MAX_DISTANCE) and (node1.energy > 0) and (node2.energy > 0):
                    x = [self.nodes[i].x,self.nodes[j].x]
                    y = [self.nodes[i].y,self.nodes[j].y]
                    plt.plot(x,y,c = [0.5,1,0.5])
                    
        plt.pause(0.001)
    def find_neighbors(self,node):
        edges = []
        for node2 in self.nodes:
            if (node != node2) and (dis := dist(node,node2) < MAX_DISTANCE) and (node2.energy > 0) and (node.energy > 0):
                edges.append(dis)
            else:
                edges.append(0)
        return np.array(edges , dtype = float)
    def update(self):
        for node in self.nodes:
            node.update_pheromone()
            node.update()

class IABR:
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
                
                prob[i] = (pheromone**ALPHA)*(visibility**BETA)
        if np.sum(prob) == 0:
            return False
        prob = prob/np.sum(prob)
        return np.random.choice(self.graph.n,p = prob)
    
    def forward_walk(self):
        if self.graph.nodes[self.current_node].energy < 0:
            return "DEAD"
        while (node := self.select_node()):
            self.visited[node] = 1
            self.path.append(node)
            
            self.graph.nodes[self.current_node].transmit_ant(self.graph.nodes[node] , ANT_SIZE = FORWARD_ANT_SIZE)
            self.graph.nodes[node].receive_ant(self.current_node , ANT_SIZE = FORWARD_ANT_SIZE)
            
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
#             self.graph.nodes[self.current_node].transmit_ant(self.graph.nodes[node] , ANT_SIZE = BACKWARD_ANT_SIZE)
#             self.graph.nodes[node].receive_ant(self.current_node , ANT_SIZE = BACKWARD_ANT_SIZE)
            self.current_node = node
    
    def draw(self,graph):
        current = self.path[0]
        for node in self.path[1:]:
            x = [graph.nodes[current].x,graph.nodes[node].x]
            y = [graph.nodes[current].y,graph.nodes[node].y]
            plt.plot(x,y,c = [0.5,0.5,1])
            current = node
        plt.pause(0.001)

def generate_graph():
    nodes = []
    while len(nodes) < NO_OF_NODES:
        node = tuple(np.random.randint(high=5, low=0, size=(2)))
        if node not in nodes:
            nodes.append(node)
    graph = Graph(nodes)
    for i in range(NO_OF_NODES):
        graph.nodes[i].vel = tuple(-0.5 + np.random.rand(2))
    return graph



graph_IABR = generate_graph()

no_of_packets = PACKETS
succesfull_packets_IABR = 0
total_IABR = 0
failed_attempts_IABR = 0
paths = [(graph_IABR, 2, 4), (graph_IABR, 1, 11), (graph_IABR, 7, 9)]

avg_energys_IABR = []
avg_paths_hops_IABR = []
avg_time_IABR = []
avg_distance_IABR = []


while no_of_packets > 0:
    try:
        for i, path in enumerate(paths):
            ant = IABR(path[0], path[1], path[2])
            total_IABR += failed_attempts_IABR
            failed_attempts_IABR = 0
            success = True
            while not (result := ant.forward_walk()):
                failed_attempts_IABR += 1
                ant = IABR(path[0], path[1], path[2])
                if failed_attempts_IABR > 100:
                    paths.pop(i)
                    success = False
                    break
            if result == "DEAD":
                paths.pop(i)
                success = False
                break
            if success:
                avg_paths_hops_IABR.append(len(ant.path))
                succesfull_packets_IABR += 1
                ant.draw(graph_IABR)
                avg_time_IABR.append(
                    (len(ant.path)*2) * ((BACKWARD_ANT_SIZE + FORWARD_ANT_SIZE) / BANDWIDTH))
                avg_distance_IABR.append(ant.distance_travelled)
        graph_IABR.update()
        graph_IABR.draw()
        no_of_packets -= 1

        # AVG ENERGY CALCULATION
        avg_energy = 0
        for node in graph_IABR.nodes:
            avg_energy += node.energy
        avg_energy /= len(graph_IABR.nodes)
        avg_energys_IABR.append(avg_energy)
    except:
        no_of_packets = PACKETS
        succesfull_packets_IABR = 0
        total_IABR = 0
        failed_attempts_IABR = 0
        graph_IABR = generate_graph()
        paths = [(graph_IABR, 2, 4), (graph_IABR, 1, 11), (graph_IABR, 7, 9)]

        avg_energys_IABR = []
        avg_paths_hops_IABR = []
        avg_time_IABR = []
        avg_distance_IABR = []


print("--------------------------------")
print("IABR")
print(f"UNSUCCESSFUL ATTEMPTS : {total_IABR}")
print(f"SUCCESSFUL ATTEMPTS (NETWORK LIFETIME) : {succesfull_packets_IABR}")

print(f"AVG PATH HOPS : {np.mean(avg_paths_hops_IABR)}")
print(f"AVG PATH LENGTH : {np.mean(avg_distance_IABR)}")

print(f"AVG TIME TAKEN : {np.mean(avg_time_IABR)}")

print(f"AVG ENERGY LEVELS  AFTER ALL THE SUCCESSFUL PACKETS : {avg_energys_IABR[-1]}")
print(f"AVG ENERGY PER PACKET : {(NO_OF_NODES*INITIAL_ENERGY)/succesfull_packets_IABR}")


plt.clf()
plt.plot(avg_energys_IABR)
plt.title("Average Energy of nodes")
plt.xlabel("Time (ms)")
plt.ylabel("Energy (Joules)")

plt.show()

plt.clf()
plt.plot(avg_paths_hops_IABR)
plt.title("Path hops")
plt.xlabel("Packet Number")
plt.ylabel("Hops")



plt.show()

plt.clf()
plt.plot(avg_distance_IABR)
plt.title("Path length")
plt.xlabel("Packets")
plt.ylabel("Distance Travelled (km)")

plt.show()

plt.clf()
plt.plot(avg_time_IABR)
plt.title("Latency")
plt.xlabel("Packets")
plt.ylabel("Time Taken (us))")
plt.show()


