"""Testing functions"""
import numpy as np

def orientation_head_cricket():
    angle1=np.arctan2(nose_head[1],nose_head[0])
    print(f'angle_nose_head:{np.degrees(angle1)}')
    angle2 = np.arctan2(cricket_head[1],cricket_head[0])
    print(f'cricket_head:{np.degrees(angle2)}')

    angle_between_lines = angle2 - angle1
    # Adjust the angle to be in the range of -pi to pi
    angle_between_lines = (angle_between_lines + np.pi) % (2 * np.pi) - np.pi
    # Convert the angle from radians to degrees
    angle_between_lines_degrees = np.degrees(angle_between_lines)
    print(f'angle_between_lines_degrees: {angle_between_lines_degrees}')    



nose=np.array([8,7])
head=np.array([10,5])
cricket=np.array([11,11])


nose_head=nose-head
cricket_head=cricket-head

print(f'nose_head:{nose_head}')
print(f'cricket_head:{cricket_head}')
        # self.data['x_axis_vector']=self.data['nose_x']-self.data['head_x']
        # self.data['y_axis_vector']=self.data['nose_y']-self.data['head_y']
        # self.data['x_cricket_vector']=self.data['Cricket_x']-self.data['head_x']
        # self.data['y_cricket_vector']=self.data['Cricket_y']-self.data['head_y']
orientation_head_cricket()

