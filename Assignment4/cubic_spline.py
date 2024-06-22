import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

matplotlib.use('TkAgg')


class InteractiveSplinePlot:
    def __init__(self):
        # Initial points
        self.x = []
        self.y = []
        self.coeffs = []
        self.tangents = []
        self.sorted_x = []
        self.sorted_y = []
        self.last_click_point = None
        self.dragging_point = None
        self.dragging_tangent = None
        self.l_tangent_mark, self.r_tangent_mark = None, None
        self.fig, self.ax = plt.subplots()
        self.x_width = 10
        self.y_height = 10
        self.ax.set_xlim(0, self.x_width)
        self.ax.set_ylim(-self.y_height / 2, self.y_height / 2)
        self.click_eps = min(self.x_width, self.y_height) * 0.05
        self.line, = self.ax.plot([], [], 'r-', label='Cubic Spline')
        self.points, = self.ax.plot([], [], 'bo', label='Control Points')
        self.tangent_points, = self.ax.plot([], [], 'go', label='Tangent Points')
        self.tangent_lines, = self.ax.plot([], [], 'g--', label='Control Points')

        # Connect the event handlers
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_remove = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Reset Button
        reset_ax = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.button_reset = Button(reset_ax, 'Reset')
        self.button_reset.on_clicked(self.reset)

        plt.show()

    def polyfit(self, x, y, tangents):
        # n denotes the number of intervals
        n = len(x) - 1
        # the width of each intervals (difference of x)
        h = np.diff(x)

        # Initialize the right-hand side of the system, alpha
        alpha = [0] * (n + 1)

        if len(tangents) == 0:
            alpha[0] = 1.5 * (y[1] - y[0]) / h[0]
        else:
            alpha[0] = (3 / h[0]) * (y[1] - y[0]) - 3 * tangents[0][1]

        if len(tangents) == n + 1:
            alpha[n] = 3 * tangents[n][0] - (3 / h[n - 1]) * (y[n] - y[n - 1])
        else:
            alpha[n] = 1.5 * (y[n] - y[n - 1]) / h[n - 1]

        for i in range(1, n):
            dydx_r = (y[i + 1] - y[i]) / h[i]
            dydx_l = (y[i] - y[i - 1]) / h[i - 1]
            if len(tangents) != 0:
                dydx_l = tangents[i][0]
            if len(tangents) == n + 1:
                dydx_r = tangents[i][1]
            alpha[i] = 3 * (dydx_r - dydx_l)

        # Diagonal elements, lower diagonal elements and solution vectors of a tridiagonal matrix.
        l = [1] * (n + 1)
        mu = [0] * (n + 1)
        z = [0] * (n + 1)
        c = [0] * (n + 1)

        # Setting up the boundary conditions
        l[0] = 2 * h[0]
        mu[0] = 0.5
        z[0] = alpha[0] / l[0]

        l[n] = 2 * h[n - 1]
        z[n] = alpha[n] / l[n]
        c[n] = z[n]

        # Solve for the value of the second derivative of a cubic polynomial (c)
        for i in range(1, n):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        # Backward pass for cubic coefficients
        coeffs = []
        for j in reversed(range(n)):
            c[j] = z[j] - mu[j] * c[j + 1]
            b = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
            d = (c[j + 1] - c[j]) / (3 * h[j])
            a = y[j]

            # Store coefficients
            coeffs.append([a, b, c[j], d])

        return coeffs[::-1]

    def eval_spline(self, x, coeffs):
        x_new = np.linspace(x.min(), x.max(), 100)
        y_new = []
        for x0 in x_new:
            for i in range(len(x) - 1):
                if x[i] <= x0 <= x[i + 1]:
                    a, b, c, d = coeffs[i]
                    delta_x = x0 - x[i]
                    y_new.append(a + b * delta_x + c * delta_x ** 2 + d * delta_x ** 3)
                    break
        return x_new, y_new

    def cubic_spline_interpolation(self, x, y):
        # Sort points
        self.sorted_indices = np.argsort(x)
        self.sorted_x = np.array(x)[self.sorted_indices]
        self.sorted_y = np.array(y)[self.sorted_indices]

        if len(x) < 2:
            return x, y

        self.coeffs = self.polyfit(self.sorted_x, self.sorted_y, self.tangents)
        return self.eval_spline(self.sorted_x, self.coeffs)

    def spline_and_update_plot(self):
        x_new, y_new = self.cubic_spline_interpolation(self.x, self.y)
        self.line.set_data(x_new, y_new)
        self.points.set_data(self.x, self.y)
        self.update_tangents()
        if len(self.x) >= 2 and self.last_click_point is not None:
            self.update_tangent_marks(self.tangents[self.last_click_point], (self.x[self.last_click_point], self.y[self.last_click_point]))

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.ax.legend()

    def update_tangents(self):
        if len(self.x) < 2:
            return

        # calculate tangent for each point
        self.tangents = []
        for i in range(len(self.x)):
            sorted_idx = self.sorted_indices[i]
            if sorted_idx == 0:
                tangent = [None, self.coeffs[0][1]]
            elif sorted_idx == len(self.x) - 1:
                tangent = [self.coeffs[-1][1], None]
            else:
                tangent = [self.coeffs[sorted_idx - 1][1], self.coeffs[sorted_idx][1]]
            self.tangents.append(tangent)

    def is_click(self, point, event):
        return abs(point[0] - event.xdata) < self.click_eps and abs(point[1] - event.ydata) < self.click_eps

    def update_tangent_marks(self, tangents, point):
        dx = self.click_eps * 3
        x, y = point
        l_tangent, r_tangent = tangents
        # points
        x_tangent, y_tangent = [x], [y]

        # coords of tangent lines
        x_line = []
        y_line = []

        if l_tangent:
            x_l = x - dx
            y_l = y - dx * l_tangent
            self.l_tangent_mark = (x_l, y_l)
            x_tangent.append(x_l)
            y_tangent.append(y_l)
            # Draw a line from the point to the left edit point
            x_line.extend([x, x_l])
            y_line.extend([y, y_l])
        if r_tangent:
            x_r = x + dx
            y_r = y + dx * r_tangent
            self.r_tangent_mark = (x_r, y_r)
            x_tangent.append(x_r)
            y_tangent.append(y_r)
            # Draw a line from the point to the right edit point
            x_line.extend([x, x_r])
            y_line.extend([y, y_r])

        self.tangent_points.set_data(x_tangent, y_tangent)
        self.tangent_lines.set_data(x_line, y_line)

    def on_click(self, event):
        if event.button != 1:  # Only consider left mouse button
            return

        if self.dragging_tangent is None:
            if self.l_tangent_mark and self.is_click(self.l_tangent_mark, event):
                self.dragging_tangent = 'l'
            elif self.r_tangent_mark and self.is_click(self.r_tangent_mark, event):
                self.dragging_tangent = 'r'

        if self.dragging_point is None and self.dragging_tangent is None:
            for i, (x, y) in enumerate(zip(self.x, self.y)):
                if self.is_click((x, y), event):  # Check proximity
                    self.dragging_point = i
                    self.last_click_point = i
                    # plot tangents
                    if len(self.x) >= 2:
                        self.update_tangent_marks(self.tangents[i], (x, y))
                    return
            self.x.append(event.xdata)
            self.y.append(event.ydata)

        self.spline_and_update_plot()

    def on_release(self, event):
        if event.button != 1:  # Only consider left mouse button
            return
        self.dragging_point = None
        self.dragging_tangent = None
        self.spline_and_update_plot()

    def on_motion(self, event):
        if self.dragging_point is not None:
            self.x[self.dragging_point] = event.xdata
            self.y[self.dragging_point] = event.ydata
            self.spline_and_update_plot()
        elif self.dragging_tangent is not None:
            point_idx = self.last_click_point
            x, y = self.x[point_idx], self.y[point_idx]

            dx, dy = x - event.xdata, y - event.ydata
            if abs(dx) < 1e-6:  # Avoid division by zero
                slope = float('inf')
            else:
                slope = dy / dx
            # print(f'point coords: {x, y}, click point: {event.xdata, event.ydata}, dx: {dx}, dy: {dy}, slope change from { self.tangents[point_idx][1]} to {slope}')
            if self.dragging_tangent == 'l':
                self.tangents[point_idx][0] = slope
            elif self.dragging_tangent == 'r':
                self.tangents[point_idx][1] = slope
            self.spline_and_update_plot()

    def on_key_press(self, event):
        if event.key == 'escape':
            plt.close(self.fig)
        if event.key == 'backspace' or event.key == 'delete':
            if self.x and self.y:
                self.x.pop()
                self.y.pop()
                if self.last_click_point == len(self.x):
                    self.last_click_point = None
                    self.tangent_points.set_data([], [])
                    self.tangent_lines.set_data([], [])
                    self.l_tangent_mark = None
                    self.r_tangent_mark = None
                self.spline_and_update_plot()

    def reset(self, event):
        self.x = []
        self.y = []
        self.tangent_points.set_data([], [])
        self.tangent_lines.set_data([], [])
        self.l_tangent_mark = None
        self.r_tangent_mark = None
        self.last_click_point = None
        self.dragging_tangent = None
        self.spline_and_update_plot()


# Create an instance of the interactive plot
plot = InteractiveSplinePlot()
