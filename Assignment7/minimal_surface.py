import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class MeshProcessor:
    def __init__(self, obj_file):
        self.obj_file = obj_file
        self.vertices = []
        self.faces = []
        self.vertex_neighbors = {}
        self.neighbor_points = {}
        self.boundary_points = []
        self.pi = 3.1415926

    def load_data(self):
        # load vertices and faces from obj file
        with open(self.obj_file, "r") as file:
            for line in file:
                if line.startswith("v "):
                    vertex = list(map(float, line.split()[1:]))
                    self.vertices.append(np.array(vertex))
                elif line.startswith("f "):
                    face = [int(vertex.split("/")[0]) - 1 for vertex in line.split()[1:]]
                    self.faces.append(np.array(face))

        self._build_vertex_neighbors()
        self._find_neighbor_points()
        self._find_boundary_points()

    def _build_vertex_neighbors(self):
        # Construct the triangle relationship for vertices
        for triangle_index, triangle in enumerate(self.faces):
            for vertex_index in triangle:
                if vertex_index not in self.vertex_neighbors:
                    self.vertex_neighbors[vertex_index] = [triangle_index]
                else:
                    self.vertex_neighbors[vertex_index].append(triangle_index)

    def _find_neighbor_points(self):
        # Find the other vertices that each vertex is directly connected to
        for vertex_index, neighbor_triangles in self.vertex_neighbors.items():
            neighbors = set()
            for triangle_index in neighbor_triangles:
                for vertex in self.faces[triangle_index]:
                    if vertex != vertex_index:
                        neighbors.add(vertex)
            self.neighbor_points[vertex_index] = neighbors

    def _find_boundary_points(self):
        # Find boundary points
        boundary_edges = set()
        for triangle in self.faces:
            for j in range(3):
                edge = tuple(sorted([triangle[j], triangle[(j + 1) % 3]]))
                if edge in boundary_edges:
                    boundary_edges.remove(edge)
                else:
                    boundary_edges.add(edge)

        temp_boundary_points = set()
        for edge in boundary_edges:
            temp_boundary_points.update(edge)

        self.boundary_points = list(temp_boundary_points)

    def draw_3d(self, result):
        polygons = [[result[index] for index in face] for face in self.faces]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection3d(Poly3DCollection(polygons, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)

    def draw_2d(self, result):
        # 绘制2D参数化结果
        polygons = [[result[index] for index in face] for face in self.faces]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.add_collection(PolyCollection(polygons, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    def global_calculation(self):
        # Global Laplacian smoothing
        n = len(self.vertices)
        L = np.eye(n)
        Y = np.zeros((n, 3))

        for vertex_index, neighbor_triangles in self.vertex_neighbors.items():
            if vertex_index not in self.boundary_points:
                total_cot = sum(self._cot_dist(vertex_index, neighbor, triangle)
                                for neighbor in self.neighbor_points[vertex_index]
                                for triangle in neighbor_triangles if neighbor in self.faces[triangle])

                for neighbor in self.neighbor_points[vertex_index]:
                    temp_cot = sum(self._cot_dist(vertex_index, neighbor, triangle)
                                   for triangle in neighbor_triangles if neighbor in self.faces[triangle])
                    L[vertex_index, neighbor] = -temp_cot / total_cot
            else:
                Y[vertex_index] = self.vertices[vertex_index]

        result = np.linalg.solve(L, Y)
        self.draw_3d(result)

    def parameterization(self):
        # Polar angle sorting
        sorted_boundary = self._sort_boundary_points()
        result = [[] for _ in range(len(self.vertices))]

        # Map boundary points to unit circle
        for i, vertex_index in enumerate(sorted_boundary):
            angle = i * 2 * self.pi / len(sorted_boundary)
            result[vertex_index] = [-math.cos(angle), -math.sin(angle)]

        # Construct and solve linear equation systems
        n = len(self.vertices)
        L = np.eye(n)
        Y = np.zeros((n, 2))

        for vertex_index, neighbor_triangles in self.vertex_neighbors.items():
            if vertex_index not in self.boundary_points:
                total_cot = sum(self._cot_dist(vertex_index, neighbor, triangle)
                                for neighbor in self.neighbor_points[vertex_index]
                                for triangle in neighbor_triangles if neighbor in self.faces[triangle])

                for neighbor in self.neighbor_points[vertex_index]:
                    temp_cot = sum(self._cot_dist(vertex_index, neighbor, triangle)
                                   for triangle in neighbor_triangles if neighbor in self.faces[triangle])
                    L[vertex_index, neighbor] = -temp_cot / total_cot
            else:
                Y[vertex_index] = result[vertex_index]

        param_result = np.linalg.solve(L, Y)
        self.draw_2d(param_result)

    def _sort_boundary_points(self):
        # Polar angle sorting of boundary points
        polar_coordinates = []
        for index in self.boundary_points:
            point = self.vertices[index]
            polar_radius = np.linalg.norm(point)
            polar_angle = np.arctan2(point[1], point[0])
            polar_coordinates.append((index, polar_radius, polar_angle))
        return [index for index, _, _ in sorted(polar_coordinates, key=lambda x: x[2])]

    def _cot_dist(self, p2_index, p3_index, triangle_index):
        p1_index = sum(self.faces[triangle_index]) - p3_index - p2_index
        p1, p2, p3 = self.vertices[p1_index], self.vertices[p2_index], self.vertices[p3_index]
        edge_AB, edge_AC = p2 - p1, p3 - p1
        dot_product = np.dot(edge_AB, edge_AC)
        cross_product = np.linalg.norm(np.cross(edge_AB, edge_AC))
        return dot_product / cross_product


def main():
    obj_file = "../models/Balls.obj"
    mesh = MeshProcessor(obj_file)
    mesh.load_data()
    mesh.global_calculation()
    mesh.parameterization()
    plt.show()


if __name__ == "__main__":
    main()
