from vpython import *
import numpy as np

scene = canvas(title = 'Magnetic Field Around a Wire')


#parameters

I = 1 # current in Amperes
mu0 = 1 #magnetic constant
wire_length = 10 #length of wire
num_points = 20 # number of field vectors to draw

# create the wire

wire = cylinder(pos= vector(0,0,-wire_length/2), axis= vector(0,0,wire_length), radius=0.05, color=color.red)

# create the field vectors
radius = 1.5 # radius at which vectors are drawn
z_plane = 0

for theta in np.linspace(0, 2*np.pi, num_points, endpoint=False):
    
    #position of the field vector
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    
    pos = vector(x, y, z_plane)
    
    # direction of magnetic field(tangent to the circle)
    
    B_dir =  cross(vector(0,0,1),pos).norm()
    arrow(pos=pos, axis = 0.3*B_dir, color=color.cyan)
    
# add a label to the wire
label(pos=vector(0,0,wire_length/2+0.5), text="Current ↑", xoffset=20, yoffset=20, box=False)  