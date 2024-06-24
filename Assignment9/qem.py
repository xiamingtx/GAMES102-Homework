#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:qem.py
# author:xm
# datetime:2024/6/24 12:51
# software: PyCharm

"""
Surface Simplification based on《Surface Simplification Using Quadric Error Metrics》
Reference: http://mgarland.org/files/papers/quadrics.pdf
"""

# import module your need
import heapq
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Edge:
    def __init__(self, p1_index, p2_index, Q_all, vertices, boundary_edges, boundary_points):
        """
            Initialize the edge object.

            Params：
            - p1_index: The first vertex index of the edge.
            - p2_index: The second vertex index of the edge.
            - Q_all: List of error matrices for all vertices.
            - vertices: List of coordinates of all vertices.
            - boundary_edges: The collection of boundary edges.
            - boundary_points: List of boundary points.
        """
        self.points = [p1_index, p2_index]  # vertex index of edge
        self.Q_all = Q_all
        self.vertices = vertices
        self.boundary_edges = boundary_edges
        self.boundary_points = boundary_points
        self.new_point_base, self.new_point = self.count_new_point()  # Calculate new vertices
        self.cost = self.count_cost()  # Calculate folding cost

    def count_new_point(self):
        """
           Calculate the generated vertices after folding the edges.
           cost = V.T * Q * v, v_new = argmin_v (cost). here, v is homogeneous coords = (x, y, z, 1)

           Returns：
           - v: The new vertex represented by homogeneous coordinates.
           - v2: The new vertex represented in Cartesian coordinates.
           """
        Y = np.zeros(shape=(4, 1), dtype=float)
        Y[3][0] = 1
        try:
            # Calculate the inverse matrix of the sum of the error matrices of two vertices
            temp_Q = self.Q_all[self.points[0]] + self.Q_all[self.points[1]]
            temp_Q[3][:3] = 0  # Set the first three columns of the last row of homogeneous coordinates to 0
            temp_Q[3][3] = 1  # Set the last row and last column of the homogeneous coordinates to 1
            v = np.linalg.inv(temp_Q) @ Y  # Compute the homogeneous coordinates of the new vertex
            v2 = np.array([v[0][0], v[1][0], v[2][0]])  # Cartesian coordinates of the new vertex
        except np.linalg.LinAlgError:
            # If the matrix is irreversible, take the midpoint of the two vertices as the new vertex
            v2 = (self.vertices[self.points[0]] + self.vertices[self.points[1]]) / 2
            v = np.append(v2, 1)  # Homogeneous coordinates of the new vertex
        return v, v2

    def count_cost(self):
        """
            Calculate the cost (error) of folded edges.

            Returns：
            - cost: folding cost。
            """
        # High costs at the border
        if (self.points[0], self.points[1]) in self.boundary_edges:
            return 514
        elif self.points[0] in self.boundary_points or self.points[1] in self.boundary_points:
            return 114
        # Calculate the error of new vertices after folding
        dot1 = np.transpose(self.new_point_base) @ (self.Q_all[self.points[0]] + self.Q_all[self.points[1]])
        dot2 = dot1 @ self.new_point_base
        return dot2[0][0]

    def __lt__(self, other):
        return self.cost < other.cost


class QEMSimplification:
    def __init__(self, obj_path):
        """
            Initialize an instance of the QEMSimplification class.

            Params：
            - obj_path: The obj file path to load.
        """
        self.obj_path = obj_path  # path of obj file
        self.vertices = []  # vertex list
        self.faces = []  # face list
        self.edges = []  # edge list
        self.Q_all = []  # Error matrix for all vertices
        self.vertex_neighbors = {}  # Neighborhood information of vertices
        self.boundary_edges = set()  # Set of boundary edges
        self.boundary_points = []  # List of boundary vertices
        self.alpha = 0.2  # Step size (can be used to control the degree of simplification)

    def load_data(self):
        """
            Load mesh data from the obj file.
        """
        with open(self.obj_path, "r") as file:
            for line in file:
                if line.startswith("v "):
                    vertex = list(map(float, line.split()[1:]))
                    self.vertices.append(np.array(vertex))
                elif line.startswith("f "):
                    face = [int(vertex.split("/")[0]) - 1 for vertex in line.split()[1:]]
                    self.faces.append(np.array(face))

        for triangle_index, triangle in enumerate(self.faces):
            for vertex_index in triangle:
                if vertex_index not in self.vertex_neighbors:
                    self.vertex_neighbors[vertex_index] = [triangle_index]
                else:
                    self.vertex_neighbors[vertex_index].append(triangle_index)

        for triangle in self.faces:
            for i in range(3):
                vertex1, vertex2 = triangle[i], triangle[(i + 1) % 3]
                edge = tuple(sorted([vertex1, vertex2]))
                if edge not in self.edges:
                    self.edges.append(edge)

        self.find_boundary_edges()

    def find_boundary_edges(self):
        """
           Find all boundary edges and boundary points.
       """
        temp_boundary_points = set()
        for triangle in self.faces:
            for i in range(3):
                edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
                if edge in self.boundary_edges:
                    self.boundary_edges.remove(edge)
                else:
                    self.boundary_edges.add(edge)
        for edge in self.boundary_edges:
            temp_boundary_points.update(edge)
        self.boundary_points = list(temp_boundary_points)

    def get_Q(self, p_index):
        """
            Computes the error matrix for a given vertex.

            Params：
            - p_index: index of vertex

            Returns：
            - Q: Error matrix of vertices
        """
        Q = np.zeros((4, 4), dtype=float)
        # Traverse adjacent triangular faces
        for triangle in self.vertex_neighbors[p_index]:
            if self.faces[triangle] is None:
                continue
            # Calculate the plane equation coefficient matrix of this triangle
            p1, p2, p3 = self.faces[triangle]
            p = self.get_plane_factor(p1, p2, p3)
            # For vertex v = (x, y, z, 1), we care about the error metric squared =
            # (ax + by + cz + d)^2 = (v.T() * p)^2 = v.T * (p * p.T) * v
            # here, we calculate the sum of (p * p.T), named K_p, i.e. Q
            Q += p.transpose() @ p
        return Q

    def get_plane_factor(self, p1, p2, p3):
        """
            Calculate the plane equation coefficients.
            Ax + By + Cz + d = 0

            Params：
            - p1, p2, p3: Indices of three vertices。

            Returns：
            - plane_factor: Plane equation coefficient matrix。
        """
        # Get the coordinates of three vertices
        point1, point2, point3 = self.vertices[p1], self.vertices[p2], self.vertices[p3]
        # Compute two vectors
        p12, p13 = point2 - point1, point3 - point1
        # Calculate Normal N
        N = np.cross(p12, p13)
        A, B, C = N
        # Get the plane equation coefficients
        D = -np.dot(N, point1)
        return np.array([[A, B, C, D]])

    def iteration(self, count_max=200):
        """
            Perform an iterative process of mesh simplification.

            Params：
            - count_max: The maximum number of iterations
        """
        # Step 1: Calculate the error matrix Q for all vertices
        self.Q_all = [self.get_Q(i) for i in range(len(self.vertices))]

        # Step 2: Build a priority queue and store the error of each edge
        pq = [Edge(e[0], e[1], self.Q_all, self.vertices, self.boundary_edges, self.boundary_points) for e in
              self.edges]
        heapq.heapify(pq)

        # Step 3: Iteratively collapse the edge with minimum error until reaching the maximum number of iterations
        count = 0
        while pq and count < count_max:
            count += 1
            # Take the edge with the smallest error from the priority queue
            top_edge = heapq.heappop(pq)
            point1_index, point2_index = top_edge.points

            # Skip collapsed vertices
            if self.vertices[point1_index] is None or self.vertices[point2_index] is None:
                continue

            # Avoid folding edges whose ends are boundary points
            if point1_index in self.boundary_points and point2_index in self.boundary_points:
                continue

            # Get new vertices
            new_point = top_edge.new_point

            # Update vertices: keep point1_index, delete point2_index
            self.vertices[point1_index] = new_point
            self.vertices[point2_index] = None

            # Update faces: change faces involving point2_index to point1_index.
            for face_index in range(len(self.faces)):
                if self.faces[face_index] is None:
                    continue
                if point1_index in self.faces[face_index] and point2_index in self.faces[face_index]:
                    self.faces[face_index] = None
                elif point2_index in self.faces[face_index]:
                    self.faces[face_index] = [point1_index if element == point2_index else element for element in
                                              self.faces[face_index]]

            # Update priority queue
            self.update_priority_queue(pq, point1_index, point2_index)

    def update_priority_queue(self, pq, point1_index, point2_index):
        """
            Update edges in priority queue

            Params：
            - pq: priority queue。
            - point1_index: reserved vertex index。
            - point2_index: deleted vertex index。
        """
        modified_q = []

        # Step 1: Merge the neighborhood information of point2_index into point1_index
        self.vertex_neighbors[point1_index].extend(self.vertex_neighbors[point2_index])

        # Step 2: Recalculate the error matrix of point1_index
        self.Q_all[point1_index] = self.get_Q(point1_index)

        # Step 3: Update edges in priority queue
        for item_pq in pq:
            all_points = item_pq.points

            # If the edge contains both point1_index and point2_index, delete it.
            if point1_index in all_points and point2_index in all_points:
                continue
            # If the edge contains point1_index, update cost
            elif point1_index in all_points:
                modified_q.append(Edge(item_pq.points[0], item_pq.points[1], self.Q_all, self.vertices,
                                       self.boundary_edges, self.boundary_points))

            # If the edge contains point2_index, replace point2_index to point1_index
            elif point2_index in all_points:
                temp_points = all_points
                if temp_points[0] == point2_index:
                    temp_points[0] = point1_index
                else:
                    temp_points[1] = point1_index

                # Ensure the order
                if temp_points[0] < temp_points[1]:
                    modified_q.append(Edge(temp_points[0], temp_points[1], self.Q_all, self.vertices,
                                           self.boundary_edges, self.boundary_points))
                else:
                    modified_q.append(Edge(temp_points[1], temp_points[0], self.Q_all, self.vertices,
                                           self.boundary_edges, self.boundary_points))

            # If the edge not contains point1_index and point2_index, remain unchanged
            else:
                modified_q.append(item_pq)

        # Rebuild priority queue
        heapq.heapify(modified_q)
        pq.clear()
        pq.extend(modified_q)

    def draw(self):
        polygons = [[self.vertices[index] for index in face] for face in self.faces if face is not None]
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

    def simplify(self):
        """
            Perform a mesh simplification process
        """
        self.load_data()
        self.draw()
        self.iteration()
        vertices_count = sum(1 for v in self.vertices if v is not None)
        faces_count = sum(1 for f in self.faces if f is not None)
        print(f"Remaining vertices: {vertices_count}")
        print(f"Remaining faces: {faces_count}")
        self.draw()


if __name__ == "__main__":
    obj_path = "../models/Balls.obj"
    qem = QEMSimplification(obj_path)
    qem.simplify()
