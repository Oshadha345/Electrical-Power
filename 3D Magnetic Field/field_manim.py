from manim import *

class MagneticField3D(ThreeDScene):
    
    def construct(self):
        
        #set up 3D camera
        
        self.set_camera_orientation(phi=70*DEGREES, theta=30*DEGREES)
        
        # add a vertical wire
        wire = Cylinder(radius=0.05, height=4, direction=OUT, color= RED)
        self.add(wire)
        
        #add label for current direction
        
        current_label = Tex("Current ↑", font_size=24)
        current_label.rotate(PI/2)
        current_label.move_to([0,0,2.5])
        self.add(current_label)
        
        #parameters for magnetic field loops
        radii = [1,1.5,2]
        heights = [-1,0,1]
        arrows = []
        
        #create circular field lines with arrows
        
        for z in heights:
            for r in radii:
                
                #create circular arc in XY plane at height z
                circle = Circle(radius=r, color=BLUE, stroke_width=1.5)
                circle.rotate(PI/2, axis=RIGHT)
                circle.shift([0,0,z])
                self.add(circle)
                
                #add arrow to animate motion along the circle
                arrow = Arrow3D(
                    start=circle.point_from_proportion(0),
                    end=circle.point_from_proportion(0.05),
                    color = YELLOW,
                    thickness=0.01
                )
                
                self.add(arrow)
                arrows.append((arrow, r, z))
                
            # add rotation update to all arrows
            
            def update_arrow(arrow, dt, r, z, t_tracker):
                
                t = t_tracker.get_value() % 1
                circle = Circle(radius=r)
                circle.rotate(PI/2, axis=RIGHT)
                circle.shift([0,0,z])
                start = circle.point_from_proportion(t)
                end = circle.point_from_proportion((t + 0.05) % 1)
                arrow.put_start_and_end_on(start, end)
                
            # add a time tracker for smooth update
            
            t_tracker = ValueTracker(0)
            
            
            for arrow, r, z in arrows:
                arrow.add_updater(lambda m, dt, r=r,z=z: update_arrow(m, dt, r, z, t_tracker))
                
            #animate the time tracker
            
            self.play(t_tracker.animate.increment_value(2), run_time=10, rate_func=linear)
            
            self.wait()