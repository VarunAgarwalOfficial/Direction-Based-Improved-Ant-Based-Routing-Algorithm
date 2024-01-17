import matplotlib.pyplot as plt
import copy
import numpy as np
from abr import *
from iabr import *
from dbiabr import *


from params import *
from node import *


class Graph:
    def __init__(self, nodes):
        self.n = len(nodes)
        self.nodes = [Node(pos, self.n) for pos in nodes]

    def find_neighbors(self, node):
        edges = []
        for node2 in self.nodes:
            if (node != node2) and (dist(node, node2) < MAX_DISTANCE) and (node2.energy > 0) and (node.energy > 0):
                edges.append(1)
            else:
                edges.append(0)
        return np.array(edges, dtype=float)

    def update(self):
        for node in self.nodes:
            node.update_pheromone()
            node.update()


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


graph_ORIG = generate_graph()
graph_ABR = copy.deepcopy(graph_ORIG)
graph_IABR = copy.deepcopy(graph_ORIG)
graph_DBIABR = copy.deepcopy(graph_ORIG)


# ABR
no_of_packets = PACKETS
succesfull_packets_ABR = 0
total_ABR = 0
failed_attempts_ABR = 0
paths = [(graph_ABR, 2, 4), (graph_ABR, 1, 11), (graph_ABR, 7, 9)]

avg_energys_ABR = []
avg_paths_hops_ABR = []
avg_time_ABR = []
avg_distance_ABR = []


while no_of_packets > 0:
    for i, path in enumerate(paths):
        ant = ABR(path[0], path[1], path[2])
        total_ABR += failed_attempts_ABR
        failed_attempts_ABR = 0
        success = True
        while not (result := ant.forward_walk()):
            failed_attempts_ABR += 1
            ant = ABR(path[0], path[1], path[2])
            if failed_attempts_ABR > 100:
                paths.pop(i)
                success = False
                break
        if result == "DEAD":
            paths.pop(i)
            success = False
            break
        if success:
            avg_paths_hops_ABR.append(len(ant.path))
            succesfull_packets_ABR += 1
            avg_time_ABR.append(
                (len(ant.path)*2) * ((BACKWARD_ANT_SIZE + FORWARD_ANT_SIZE) / BANDWIDTH))
            avg_distance_ABR.append(ant.distance_travelled)
    graph_ABR.update()
    no_of_packets -= 1

    # AVG ENERGY CALCULATION
    avg_energy = 0
    for node in graph_ABR.nodes:
        avg_energy += node.energy
    avg_energy /= len(graph_ABR.nodes)
    avg_energys_ABR.append(avg_energy)


print("--------------------------------")
print("ABR")
print(f"UNSUCCESSFUL ATTEMPTS : {total_ABR}")
print(f"SUCCESSFUL ATTEMPTS (NETWORK LIFETIME) : {succesfull_packets_ABR}")

print(f"AVG PATH HOPS : {np.mean(avg_paths_hops_ABR)}")
print(f"AVG PATH LENGTH : {np.mean(avg_distance_ABR)}")

print(f"AVG TIME TAKEN : {np.mean(avg_time_ABR)}")

print(f"AVG ENERGY LEVELS  AFTER ALL THE SUCCESSFUL PACKETS : {avg_energys_ABR[-1]}")
print(f"AVG ENERGY PER PACKET : {(NO_OF_NODES*INITIAL_ENERGY)/succesfull_packets_ABR}")


# IABR


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
                avg_time_IABR.append(
                    (len(ant.path)*2) * ((BACKWARD_ANT_SIZE + FORWARD_ANT_SIZE) / BANDWIDTH))
                avg_distance_IABR.append(ant.distance_travelled)
        graph_IABR.update()
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
        graph_IABR = copy.deepcopy(graph_ORIG)
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


# DBIABR


no_of_packets = PACKETS
succesfull_packets_DBIABR = 0
total_DBIABR = 0
failed_attempts_DBIABR = 0
paths = [(graph_DBIABR, 2, 4), (graph_DBIABR, 1, 11), (graph_DBIABR, 7, 9)]

avg_energys_DBIABR = []
avg_paths_hops_DBIABR = []
avg_time_DBIABR = []
avg_distance_DBIABR = []


while no_of_packets > 0:
    try:
        for i, path in enumerate(paths):
            ant = DBIABR(path[0], path[1], path[2])
            total_DBIABR += failed_attempts_DBIABR
            failed_attempts_DBIABR = 0
            success = True
            while not (result := ant.forward_walk()):
                failed_attempts_DBIABR += 1
                ant = DBIABR(path[0], path[1], path[2])
                if failed_attempts_DBIABR > 100:
                    paths.pop(i)
                    success = False
                    break
            if result == "DEAD":
                paths.pop(i)
                success = False
                break
            if success:
                avg_paths_hops_DBIABR.append(len(ant.path))
                succesfull_packets_DBIABR += 1
                avg_time_DBIABR.append(
                    (len(ant.path)*2) * ((BACKWARD_ANT_SIZE + FORWARD_ANT_SIZE) / BANDWIDTH))
                avg_distance_DBIABR.append(ant.distance_travelled)
        graph_DBIABR.update()
        no_of_packets -= 1

        # AVG ENERGY CALCULATION
        avg_energy = 0
        for node in graph_DBIABR.nodes:
            avg_energy += node.energy
        avg_energy /= len(graph_DBIABR.nodes)
        avg_energys_DBIABR.append(avg_energy)
    except:
        no_of_packets = PACKETS
        succesfull_packets_DBIABR = 0
        total_DBIABR = 0
        failed_attempts_DBIABR = 0
        graph_DBIABR = copy.deepcopy(graph_ORIG)
        paths = [(graph_DBIABR, 2, 4), (graph_DBIABR, 1, 11), (graph_DBIABR, 7, 9)]

        avg_energys_DBIABR = []
        avg_paths_hops_DBIABR = []
        avg_time_DBIABR = []
        avg_distance_DBIABR = []


print("--------------------------------")
print("DBIABR")
print(f"UNSUCCESSFUL ATTEMPTS : {total_DBIABR}")
print(f"SUCCESSFUL ATTEMPTS (NETWORK LIFETIME) : {succesfull_packets_DBIABR}")

print(f"AVG PATH HOPS : {np.mean(avg_paths_hops_DBIABR)}")
print(f"AVG PATH LENGTH : {np.mean(avg_distance_DBIABR)}")

print(f"AVG TIME TAKEN : {np.mean(avg_time_DBIABR)}")

print(f"AVG ENERGY LEVELS  AFTER ALL THE SUCCESSFUL PACKETS : {avg_energys_DBIABR[-1]}")
print(f"AVG ENERGY PER PACKET : {(NO_OF_NODES*INITIAL_ENERGY)/succesfull_packets_DBIABR}")


plt.clf()
plt.plot(avg_energys_ABR)
plt.plot(avg_energys_IABR)
plt.plot(avg_energys_DBIABR)
plt.legend(['ABR', 'IABR', 'DB-IABR'])
plt.title("Average Energy of nodes")
plt.xlabel("Time (ms)")
plt.ylabel("Energy (Joules)")

plt.show()

plt.clf()
plt.plot(avg_paths_hops_ABR)
plt.plot(avg_paths_hops_IABR)
plt.plot(avg_paths_hops_DBIABR)
plt.legend(['ABR', 'IABR', 'DB-IABR'])
plt.title("Path hops")
plt.xlabel("Packet Number")
plt.ylabel("Hops")



plt.show()

plt.clf()
plt.plot(avg_distance_ABR)
plt.plot(avg_distance_IABR)
plt.plot(avg_distance_DBIABR)
plt.legend(['ABR', 'IABR', 'DB-IABR'])
plt.title("Path length")
plt.xlabel("Packets")
plt.ylabel("Distance Travelled (km)")

plt.show()

plt.clf()
plt.plot(avg_time_ABR)
plt.plot(avg_time_IABR)
plt.plot(avg_time_DBIABR)
plt.legend(['ABR', 'IABR', 'DB-IABR'])
plt.title("Latency")
plt.xlabel("Packets")
plt.ylabel("Time Taken (us))")
plt.show()


