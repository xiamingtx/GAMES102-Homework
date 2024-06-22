import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class HalfEdgeMesh:
    def __init__(self, obj_path):
        self.vertices = []
        self.faces = []
        self.vertex_neighbors = {}
        self.neighbor_points = {}
        self.boundary_points = []
        self.load_data(obj_path)

    def load_data(self, obj_path):
        with open(obj_path, "r") as file:
            for line in file:
                if line.startswith("v "):
                    vertex = list(map(float, line.split()[1:]))
                    self.vertices.append(np.array(vertex))
                elif line.startswith("f "):
                    face = [int(vertex.split("/")[0]) - 1 for vertex in line.split()[1:]]
                    self.faces.append(np.array(face))

        # Construct neighbor information of vertices
        for i, triangle in enumerate(self.faces):
            for vertex in triangle:
                if vertex not in self.vertex_neighbors:
                    self.vertex_neighbors[vertex] = []
                self.vertex_neighbors[vertex].append(i)

        for vertex, triangles in self.vertex_neighbors.items():
            neighbors = set()
            for triangle in triangles:
                for vertex_in_face in self.faces[triangle]:
                    if vertex_in_face != vertex:
                        neighbors.add(vertex_in_face)
            self.neighbor_points[vertex] = neighbors

        self.find_boundary_points()

    def find_boundary_points(self):
        edge_dict = {}
        for face in self.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if edge in edge_dict:
                    edge_dict[edge] += 1
                else:
                    edge_dict[edge] = 1

        boundary_edges = [edge for edge, count in edge_dict.items() if count == 1]
        for edge in boundary_edges:
            self.boundary_points.extend(list(edge))

    def draw(self):
        polygons = [[self.vertices[index] for index in face] for face in self.faces]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection3d(Poly3DCollection(polygons, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        plt.show()

    def iterate(self, alpha=0.1, iterations=10):
        for i in range(iterations):
            if i % 10 == 0:
                print(f"Iteration: {i}")
            for vertex_index in range(len(self.vertices)):
                if vertex_index not in self.boundary_points:
                    self.update_vertex(vertex_index, alpha)

    def update_vertex(self, vertex_index, alpha):
        center_point = self.vertices[vertex_index]
        A = 0
        T = 0
        for neighbor_index in self.neighbor_points[vertex_index]:
            edge_vector = self.vertices[neighbor_index] - center_point
            weight = self.cotangent_weight(vertex_index, neighbor_index)
            A += weight * np.dot(edge_vector, edge_vector)
            T += weight * edge_vector
        self.vertices[vertex_index] += alpha * T / A

    def cotangent_weight(self, v_index, u_index):
        shared_triangles = [t for t in self.vertex_neighbors[v_index] if u_index in self.faces[t]]
        cot_weight = 0
        for triangle_index in shared_triangles:
            triangle = self.faces[triangle_index]
            # Find the third vertex in the triangle that is not v_index and u_index
            for i in triangle:
                if i != v_index and i != u_index:
                    third_vertex = i
            # calculate the cotangent weights
            cot_weight += self.cot_dist(v_index, u_index, third_vertex)
        return cot_weight

    def cot_dist(self, v_index, u_index, third_vertex):
        v = self.vertices[v_index]
        u = self.vertices[u_index]
        t = self.vertices[third_vertex]
        edge_vu = v - u
        edge_vt = v - t
        edge_ut = u - t
        # Return the mean cotangent value
        cot_v = np.dot(edge_vt, edge_ut) / np.linalg.norm(np.cross(edge_vt, edge_ut))
        cot_u = np.dot(edge_vu, edge_vt) / np.linalg.norm(np.cross(edge_vu, edge_vt))
        return 0.5 * (cot_v + cot_u)


if __name__ == "__main__":
    mesh = HalfEdgeMesh("Balls.obj")
    mesh.draw()
    mesh.iterate()
    mesh.draw()
