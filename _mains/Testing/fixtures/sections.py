# hp_geometry stuff
from shapely import Polygon
from structuralcodes.geometry import SurfaceGeometry, add_reinforcement
from structuralcodes.sections import BeamSection

from _mains.Testing.fixtures.materials import concrete_c40_uls, reinforcement_B500

"""
Author: Elliot Melcer

Testing T-Section

             b
    |<--------------->|
     _________________    _ _
    |                 |    ^
    |                 |    | d
    '---,         ,---'   -X-
     b1 |         | b1     |
    |<->|         |<->|    |
        |         |        | d1   
cover ^ | °  °  ° |        |      
      v |_________|       _v_     
            b0
        |<------->|
"""


b  = 300      #mm
b1 = 50
b0 = 200
d = 200
d1 = 200
cover = 50

t_section_polygon = Polygon(
    [
        (-b /2,  d/2),
        (-b /2, -d/2),
        (-b0/2, -d/2),
        (-b0/2, -d/2-d1),
        ( b0/2, -d/2-d1),
        ( b0/2, -d/2),
        ( b /2, -d/2),
        ( b /2,  d/2),
    ]
)

# structural codes stuff

# --- concrete section ---
geometry = SurfaceGeometry(
    poly=t_section_polygon, material=concrete_c40_uls
)

# --- reinforcement ---
geometry = add_reinforcement(
    geometry,
    (
        -b0/2+cover, -d/2 - d1 + cover
    ),
    20,
    reinforcement_B500,
)  # The add_reinforcement function returns a CompoundGeometry
geometry = add_reinforcement(
    geometry,
    (
        0, -d1-cover
    ),
    20,
    reinforcement_B500,
)
geometry = add_reinforcement(
    geometry,
    (
        b0/2-cover, -d/2 - d1 + cover
    ),
    20,
    reinforcement_B500,
)

t_section = BeamSection(geometry, integrator='marin')