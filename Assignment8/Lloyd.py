import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, delaunay_plot_2d


class LloydAlgorithm:
    def __init__(self, n_points=50, iterations=100):
        self.n_points = n_points
        self.iterations = iterations
        self.initial_points = None
        self.points = None

    def generate_data(self):
        # Generate random points within the (-1, 1) square
        np.random.seed(0)
        random_points = np.random.rand(self.n_points, 2) * 2 - 1
        # Four vertices of the square
        square_points = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        # Merging random and square points
        self.initial_points = np.vstack((random_points, square_points))
        self.points = self.initial_points.copy()
        return self.points

    @staticmethod
    def is_outside_box(point):
        return abs(point[0]) > 1 or abs(point[1]) > 1

    @staticmethod
    def line_intersection(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return np.array([x, y])
        return None

    def find_boundary_intersection(self, point1, point2):
        boundaries = [
            ([-1, -1], [-1, 1]),
            ([-1, 1], [1, 1]),
            ([1, 1], [1, -1]),
            ([1, -1], [-1, -1])
        ]

        for boundary in boundaries:
            intersection = self.line_intersection(point1, point2, boundary[0], boundary[1])
            if intersection is not None:
                return intersection
        return None

    def lloyd_iteration(self):
        try:
            vor = Voronoi(self.points)
        except Exception as e:
            print(f"Error in Voronoi computation: {e}")
            print(f"Number of points: {len(self.points)}")
            print(f"Points: {self.points}")
            return

        # Initialize a new list to store updated points
        new_points = []
        # Traverse each point and its corresponding Voronoi region
        for point, region in zip(vor.points, vor.point_region):
            # Check if this point in a valid Voronoi region (-1 means invalid region)
            if region != -1:
                # Get the vertices of this Voronoi region
                region_points = vor.vertices[vor.regions[region]]
                if len(region_points) > 0:
                    valid_points = []
                    for p in region_points:
                        # If the vertex is inside the bounding box, add it directly to the list of valid points.
                        if not self.is_outside_box(p):
                            valid_points.append(p)
                        # If outside the bounding box, look for the intersection of the edge
                        # it forms with the other vertices with the bounding box
                        else:
                            for other_point in region_points:
                                if not np.array_equal(p, other_point):
                                    intersection = self.find_boundary_intersection(p, other_point)
                                    if intersection is not None:
                                        valid_points.append(intersection)

                    if valid_points:
                        centroid = np.mean(valid_points, axis=0)
                        new_points.append(centroid)
                    else:
                        new_points.append(point)
                else:
                    new_points.append(point)
            else:
                new_points.append(point)

        self.points = np.array(new_points)

    def run(self):
        self.generate_data()
        self.visualization()

        for i in range(self.iterations):
            print(f"Iteration {i + 1}/{self.iterations}")
            self.lloyd_iteration()

        self.visualization()

    def visualization(self):
        vor = Voronoi(self.points)
        tri = Delaunay(self.points)

        voronoi_plot_2d(vor)
        delaunay_plot_2d(tri)

        plt.show()



if __name__ == "__main__":
    lloyd = LloydAlgorithm(n_points=50, iterations=100)
    lloyd.run()
