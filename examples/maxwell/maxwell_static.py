import dolfin
import bempp_cl.api
import numpy as np


from bempp_cl.api.external import fenics

import gmsh

gmsh.initialize()
gmsh.model.occ.addSphere(0, 0, 0, 1.0, tag=1)
gmsh.model.occ.synchronize()


# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.3)
gmsh.model.mesh.generate(3)



gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("sphere.msh")
gmsh.finalize()

## Convert with dolfin-convert
# !dolfin-convert sphere.msh sphere.xml



from dolfin import Mesh, XDMFFile

mesh = Mesh("sphere.xml")

# mesh = dolfin.UnitCubeMesh(4, 4, 4)


fenics_space = dolfin.FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
trace_space, trace_matrix = fenics.fenics_to_bempp_trace_data(fenics_space)

# Physical parameters
B_0 = 1.0  # Incident field magnitude


# Create geometry 
grid = trace_space.grid

print(f"Grid: {grid.number_of_elements} elements")