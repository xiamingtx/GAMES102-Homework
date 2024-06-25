import random
import numpy as np
import matplotlib.pyplot as plt
import mcubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d
from scipy.spatial.distance import cdist


class PointCloudReconstruction:
    def __init__(self, obj_file, alpha, update_index, mc_range, cluster_number, radius, max_nn, normal_method):
        # Initialization parameters
        self.obj_file = obj_file
        self.alpha = alpha
        self.update_index = update_index
        self.mc_range = mc_range
        self.cluster_number = cluster_number
        self.radius = radius
        self.max_nn = max_nn
        self.normal_method = normal_method

        # Initializing Data Structures
        self.vertices = []
        self.vertices_temp = []
        self.normals_all = []
        self.normals = []
        self.weights = []

    def load_data(self):
        """Load point cloud data from OBJ files"""
        with open(self.obj_file, "r") as file:
            for line in file:
                if line.startswith("v "):
                    vertex = list(map(float, line.split()[1:]))
                    self.vertices.append(np.array(vertex) * self.update_index)

    def create_normal(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vertices)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.max_nn))

        if self.normal_method == 1:
            # Method 1: Camera-facing position
            tot = np.mean(self.vertices, axis=0)
            pcd.orient_normals_towards_camera_location(tot)
        elif self.normal_method == 2:
            # Method 2: Orientation based on surrounding points
            pcd.orient_normals_consistent_tangent_plane(3)
        else:
            # Method 3: Along the axis
            pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([1.0, 0, 0]))

        for i, item in enumerate(pcd.normals):
            self.normals_all.append(self.vertices[i] - self.alpha * np.asarray(item))

    def reconstruction(self):
        # Random selection of points
        point_total_number = len(self.vertices)
        num_to_select = point_total_number // self.cluster_number
        vertices_indexes = random.sample(range(point_total_number), num_to_select)
        vertices_indexes.sort()

        # Generate and select normals
        self.create_normal()
        for item in vertices_indexes:
            self.normals.append(self.normals_all[item])

        print("Normals ready")

        # temp_vertices contains randomly selected raw points and their normal offsets
        temp_vertices = [self.vertices[i] for i in vertices_indexes]
        temp_vertices.extend(self.normals)
        self.vertices_temp.append(temp_vertices)

        n = len(self.vertices_temp[0])
        K = np.zeros(shape=(n, n), dtype=float)
        Y = np.zeros(shape=(n, 1), dtype=float)

        # The first half (the original point part) is initialized to 0 and
        # the second half (the normal offset point part) is initialized to alpha。
        for i in range(int(n / 2), n):
            Y[i] = self.alpha

        for i in range(n):
            for j in range(n):
                K[i][j] = self.RBF_major(i, j)

        # Solve Kw = Y and gain RBF weights w
        result = np.linalg.inv(K) @ Y
        print("Calculation ready")
        self.weights = np.array([item[0] for item in result])

        # Marching Cubes Algorithm
        # 1. Generate grid points
        X, Y, Z = self.mc_range
        x, y, z = X.flatten(), Y.flatten(), Z.flatten()
        points = np.vstack((x, y, z)).T

        # Calculate the RBF value (Cubic of the distance function from a mesh point to a known point cloud)
        rbf_values = self.RBF_major_result(points)

        # Calculating Implicit Function Values (Equivalent denomination at each grid point)
        u = np.dot(rbf_values, self.weights).reshape(X.shape)
        # Use marching_cubes to extract equivalent faces from the implicit function value u to
        # generate reconstructed surface vertices obj_vertices and face sheets obj_faces.
        # 1. Voxels Iteration: Iterates over all small cubes in the 3D mesh.
        # 2. equivalence points Determination: for vertices of each small cube, determine whether the implicit function
        # value is higher than the equivalence value (in this case equivalence is 0)。
        # 3. Lookup Table Matching: Use a lookup table to determine how to generate triangles within that cube
        # based on the combination of height of the 8 vertices.
        # 4. Generate Triangles: Interpolation calculates the positions of the vertices of the generated triangles.
        obj_vertices, obj_faces = mcubes.marching_cubes(u, 0)
        print("Marching cubes ready")

        return obj_vertices, obj_faces

    def RBF_major(self, point_1_index, point_2_index):
        """RBF basis functions"""
        return np.linalg.norm(self.vertices_temp[0][point_1_index] - self.vertices_temp[0][point_2_index]) ** 3

    def RBF_major_result(self, points):
        """Cubic polynomial RBF function: phi(r) = r^3, where r is the Euclidean distance between two points"""
        distances = cdist(points, self.vertices_temp[0])
        return distances ** 3

    def draw_result(self, obj_vertices, obj_faces):
        polygons = [[obj_vertices[index] for index in face] for face in obj_faces]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection3d(Poly3DCollection(polygons, facecolors='cyan', linewidths=.2, edgecolors='r', alpha=.25))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)
        ax.set_zlim(0, 60)
        plt.show()


def main():
    # Set params
    obj_file = "../models/point_cloud_models/Arma_04.obj"
    alpha = 0.5
    update_index = 10
    mc_range = np.mgrid[-15:15:0.5, -15:15:0.5, -15:15:0.5]
    cluster_number = 2
    radius = 5.0
    max_nn = 10
    normal_method = 3

    # Reconstruction
    reconstruction = PointCloudReconstruction(obj_file, alpha, update_index, mc_range, cluster_number, radius, max_nn,
                                              normal_method)
    reconstruction.load_data()
    obj_vertices, obj_faces = reconstruction.reconstruction()
    reconstruction.draw_result(obj_vertices, obj_faces)


if __name__ == "__main__":
    main()
