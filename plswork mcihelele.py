# Import essential libraries 
import requests 
import cv2 
import numpy as np 
import imutils 
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt

# command to run the program: python3 plswork.py    

url = "http://100.66.148.7:8080/shot.jpg"
  
# # While loop to continuously fetching data from the Url 
# while True: 
#     img_resp = requests.get(url) 
#     img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
#     img = cv2.imdecode(img_arr, -1) 
#     img = imutils.resize(img, width=1000, height=1800) 
#     cv2.imshow("Android_cam", img) 
  
#     # Press Esc key to exit 
#     if cv2.waitKey(1) == 27: 
#         break
  
# cv2.destroyAllWindows() 

class PathFinder:
    def __init__(self, image):
        self.img = image.copy()
        self.start = None
        self.end = None
        self.path = None

        # Create a figure and connect the event handlers
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:  # Left mouse button for start point
            self.start = (y, x)  # Note the reversal of coordinates (matplotlib vs. OpenCV)
            print(f'Start point: {self.start}')
        elif event.button == 3:  # Right mouse button for end point
            self.end = (y, x)
            print(f'End point: {self.end}')
            self.find_shortest_path()
            self.follow_path()
            # self.verify_turns()
            self.draw_path()

    # def find_shortest_path(self):
    #     # Create a binary mask with the largest contour
    #     gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    #     _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    #     mask = np.zeros_like(gray)
    #     cv2.drawContours(mask, contours, largest_contour_index, (255), thickness=cv2.FILLED)

    #     # Find the coordinates of white pixels in the mask
    #     white_pixels = np.column_stack(np.where(mask > 0))

    #     # Create a graph representation of the image
    #     G = nx.grid_2d_graph(mask.shape[0], mask.shape[1])

    #     # Remove nodes corresponding to black pixels
    #     black_pixels = np.column_stack(np.where(mask == 0))
    #     for pixel in black_pixels:
    #         G.remove_node(tuple(pixel))

    #     # Find the shortest path using A* algorithm
    #     self.path = nx.astar_path(G, self.start, self.end)

    # Define a cost function that penalizes changes in direction
    def cost_function(self, u, v, e, prev_edge=None):
        if prev_edge is None:
            return 1  # Default cost for the first edge
        else:
            # Calculate the change in direction
            angle_diff = np.abs(np.arctan2(v[0] - u[0], v[1] - u[1]) - np.arctan2(e[1][0] - u[0], e[1][1] - u[1]))
            # Penalize changes in direction
            return 1 + 20 * angle_diff / np.pi    
            
    def find_shortest_path(self):
        # Create a binary mask with the largest contour
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, largest_contour_index, (255), thickness=cv2.FILLED)

        # Find the coordinates of white pixels in the mask
        white_pixels = np.column_stack(np.where(mask > 0))

        # Create a graph representation of the image
        G = nx.grid_2d_graph(mask.shape[0], mask.shape[1])

        # Remove nodes corresponding to black pixels
        black_pixels = np.column_stack(np.where(mask == 0))
        for pixel in black_pixels:
            G.remove_node(tuple(pixel))

        # Find the shortest path using A* algorithm with the modified cost function
        self.path = nx.astar_path(G, self.start, self.end, heuristic=None, weight=self.cost_function)

    def follow_path(self):
        self.turns = []  # Reset turns
        for i in range(1, len(self.path) - 10):
            current_point = self.path[i]
            next_point = self.path[i + 10]
            vector = np.array(next_point) - np.array(current_point)
            direction = np.arctan2(vector[0], vector[1]) * 180 / np.pi
            direction = (direction + 360) % 360 # Convert to positive angle (CCW) in degr
            if direction > 60:
                print(f'current point: {current_point}, direction: {abs(direction -360)}')
                self.turns.append((current_point, abs(direction -360)))

            # if i + 2 < len(self.path):
            #     next_next_point = self.path[i + 2]
            #     lookahead_vector = np.array(next_next_point) - np.array(current_point)
            #     lookahead_direction = np.arctan2(lookahead_vector[0], lookahead_vector[1]) * 180 / np.pi
            #     deviation = np.abs(direction - lookahead_direction)

            #     if deviation > 40:
            #         turn_type = 'Left' if direction < lookahead_direction else 'Right'
            #         self.turns.append((current_point, turn_type))

        print(f'predicted: {self.turns}')
        return self.turns
    
    def verify_turns(self):
        verified_turns = []
        for turn_point, current_direction in self.turns:
            index = self.path.index(turn_point)
            angle_before = self.calculate_angle(index - 5, index)
            angle_after = self.calculate_angle(index, index + 5)
            
            print(f'angle before: {angle_before}, angle after: {angle_after}')

            if abs(angle_before - 90) > 60 or abs(angle_after - 90) > 60:
                verified_turns.append((turn_point, current_direction))

        self.turns = verified_turns
        print(f'\n \n verified: {self.turns}')

    def calculate_angle(self, start_index, end_index):
        start_point = np.array(self.path[start_index])
        end_point = np.array(self.path[end_index])
        vector = end_point - start_point
        angle = np.arctan2(vector[0], vector[1]) * 180 / np.pi
        return angle

    def draw_path(self):
        result = self.img.copy()
        for node in self.path:
            cv2.circle(result, (node[1], node[0]), 2, (0, 255, 0), thickness=-1)  # Draw circles on the path

        for turn_point, turn_type in self.turns:
            cv2.circle(result, (turn_point[1], turn_point[0]), 5, (255, 0, 0), thickness=-1)  # Mark turns with red

        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.draw()


    def show(self):
        plt.show()

image = Image.open('maze.jpg')
# image = image.convert('L')
new_image = image.resize((600, 800))
new_image.save("image_2.jpg", quality=21) 

img = cv2.imread("image_2.jpg")  # Read image 
  
# Setting parameter values 
t_lower = 275 # Lower Threshold 
t_upper = 280  # Upper threshold 
  
# Applying the Canny Edge filter 
edge = cv2.Canny(img, t_lower, t_upper) 
cv2.imwrite('edge.jpg', edge)

img = cv2.imread('edge.jpg')  # Use the filename you saved your Canny edges image as

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find contours in the image
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the index of the largest contour
largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))

# Create a black image with the same size as the original
filled_image = np.zeros_like(img)

# Draw the largest contour on the black image
cv2.drawContours(filled_image, contours, largest_contour_index, (255, 255, 255), thickness=cv2.FILLED)

# Add the original image and the filled image
result = cv2.add(img, filled_image)

# Save the result
cv2.imwrite('largest_contour_filled.jpg', result)

path_finder = PathFinder(result)
path_finder.show()


cv2.imshow('original', img) 
cv2.imshow('edge', edge) 

cv2.waitKey(0) 
cv2.destroyAllWindows() 