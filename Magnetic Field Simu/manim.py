from manim import *


class MagneticFieldAroundWire(Scene):
    def construct(self):
        # Create the central wire (coming out of the screen)
        wire = Dot(point=ORIGIN, radius=0.1, color=RED)
        label = Text("Current ⊙", font_size=24).next_to(wire, UP)

        self.play(FadeIn(wire), Write(label))

        # Create multiple circular field lines around the wire
        radii = [1, 1.5, 2, 2.5]
        field_lines = []
        arrows = []

        for r in radii:
            circle = Circle(radius=r, color=BLUE, stroke_width=1.5)
            circle.move_to(ORIGIN)
            field_lines.append(circle)

            # Add a rotating arrow on each field line
            arrow = Arrow(start=circle.point_from_proportion(0),
                          end=circle.point_from_proportion(0.05),
                          color=YELLOW, buff=0)
            arrows.append(arrow)

            self.play(Create(circle), FadeIn(arrow), run_time=0.5)

        # Animate arrows rotating around the wire (representing B field direction)
        def update_arrow(arrow, dt, radius):
            t = self.time % 1
            start = Circle(radius=radius).point_from_proportion(t)
            end = Circle(radius=radius).point_from_proportion((t + 0.05)%1)
            arrow.put_start_and_end_on(start, end)

        # Add updaters to arrows
        for arrow, r in zip(arrows, radii):
            arrow.add_updater(lambda m, dt, r=r: update_arrow(m, dt, r))

        self.wait(5)  # Let the animation run

        # Clean up
        for arrow in arrows:
            arrow.clear_updaters()

        self.wait()
 