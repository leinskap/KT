import gmsh
import vtk
import numpy as np
import sys
import math
import time

#  Simulation Parameters
DOMAIN_WIDTH = 15.0
DOMAIN_HEIGHT = 15.0
INITIAL_CRACK_LENGTH = 0.5
CRACK_Y_POS = DOMAIN_HEIGHT / 2.0
# CRACK_START_X = DOMAIN_WIDTH / 2.0 - INITIAL_CRACK_LENGTH / 2.0
CRACK_START_X = 0.01
# Crack growth parameters
NUM_STEPS = 15
CRACK_GROWTH_INCREMENT = 0.3

# Meshing parameters
ADAPTIVE_RADIUS = 0.5
MESH_SIZE_NEAR_CRACK = 0.2
MESH_SIZE_FAR = 0.5
MESH_SIZE_BOUNDARY = 0.2

# Visualization parameters
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
STEP_DELAY = 0.5
CRACK_COLOR = (1.0, 0.0, 0.0)
CRACK_LINE_WIDTH = 3.0

#  Gmsh Initialization
gmsh.initialize(sys.argv)
gmsh.option.setNumber("General.Terminal", 1)

#  VTK Setup
renderer = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
renWin.SetSize(WINDOW_WIDTH, WINDOW_HEIGHT)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

#  VTK Objects for the Main Mesh
ugrid = vtk.vtkUnstructuredGrid()
mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(ugrid)
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetRepresentationToWireframe()
actor.GetProperty().SetColor(0.8, 0.8, 0.8)
renderer.AddActor(actor)

#  Highlight
crack_poly_data = vtk.vtkPolyData()
crack_mapper = vtk.vtkPolyDataMapper()
crack_mapper.SetInputData(crack_poly_data)
crack_actor = vtk.vtkActor()
crack_actor.SetMapper(crack_mapper)
crack_actor.GetProperty().SetColor(CRACK_COLOR)
crack_actor.GetProperty().SetLineWidth(CRACK_LINE_WIDTH)
crack_actor.GetProperty().RenderLinesAsTubesOn() # Makes thick lines look better
renderer.AddActor(crack_actor) # Add the crack actor to the scene

renderer.SetBackground(0.1, 0.2, 0.4) # Background color

#  пmsh to VTK
def gmsh_to_vtk(model_name=""):
    """
    Converts the current Gmsh mesh (assumed 2D triangles)
    to a vtkUnstructuredGrid.
    Returns the vtkUnstructuredGrid.
    """
    if model_name:
        gmsh.model.setCurrent(model_name)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)
    print(node_coords)

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    triangle_type = 2
    vtk_grid = vtk.vtkUnstructuredGrid()

    points = vtk.vtkPoints()
    node_tag_to_vtk_id = {tag: i for i, tag in enumerate(node_tags)}
    for i in range(len(node_tags)):
        points.InsertPoint(i, node_coords[i])
    vtk_grid.SetPoints(points)

    if triangle_type in elem_types:
        tri_indices = np.where(elem_types == triangle_type)[0]
        if len(tri_indices) > 0:
            tri_node_tags = elem_node_tags[tri_indices[0]].reshape(-1, 3)
            num_tris = tri_node_tags.shape[0]
            vtk_grid.Allocate(num_tris, 1)

            for i in range(num_tris):
                vtk_tri = vtk.vtkTriangle()
                gmsh_nodes = tri_node_tags[i]
                for j in range(3):
                    vtk_id = node_tag_to_vtk_id.get(gmsh_nodes[j], -1)
                    if vtk_id == -1:
                        # print(f"Warning: Gmsh node tag {gmsh_nodes[j]} not found in node list!") # Can be noisy
                        continue
                    vtk_tri.GetPointIds().SetId(j, vtk_id)
                vtk_grid.InsertNextCell(vtk_tri.GetCellType(), vtk_tri.GetPointIds())

    return vtk_grid

#Extract Crack Line Nodes Coords from Gmsh
def get_crack_line_vtk(crack_line_tag, model_name=""):
    """
    Extracts nodes on the crack line from Gmsh mesh and creates
    a vtkPolyData object representing the ordered line.
    """
    if model_name:
        gmsh.model.setCurrent(model_name)

    # Get nodes associated with the crack line entity (dimension 1)
    # includeBoundary=True ensures start/end points are included
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=1, tag=crack_line_tag, includeBoundary=True)

    if not node_tags.size:
        print("Warning: No nodes found for crack line tag:", crack_line_tag)
        return vtk.vtkPolyData() # Return empty polydata

    # Reshape coordinates
    node_coords = node_coords.reshape(-1, 3)

    #  Order the nodes
    # For a horizontal crack, sorting by X-coordinate is sufficient
    # For more complex cracks, parameterization or nearest neighbor search would be needed.
    sort_indices = np.argsort(node_coords[:, 0]) # Sort based on X coordinate
    sorted_node_tags = node_tags[sort_indices]
    sorted_node_coords = node_coords[sort_indices]

    #  Create VTK PolyData for the line
    crack_points_vtk = vtk.vtkPoints()
    crack_line_vtk = vtk.vtkPolyLine()
    crack_cells_vtk = vtk.vtkCellArray()
    poly_data = vtk.vtkPolyData()

    # Add points
    num_crack_nodes = len(sorted_node_tags)
    for i in range(num_crack_nodes):
        crack_points_vtk.InsertNextPoint(sorted_node_coords[i])

    # Create the polyline cell connecting the points in order
    crack_line_vtk.GetPointIds().SetNumberOfIds(num_crack_nodes)
    for i in range(num_crack_nodes):
        crack_line_vtk.GetPointIds().SetId(i, i) # Use 0-based index within crack_points_vtk

    # Add the polyline cell to the cell array
    crack_cells_vtk.InsertNextCell(crack_line_vtk)

    # Set points and lines (cells) for the PolyData
    poly_data.SetPoints(crack_points_vtk)
    poly_data.SetLines(crack_cells_vtk)

    return poly_data


#  Simulation
current_crack_length = INITIAL_CRACK_LENGTH
model_name = "crack_model"
crack_line_geo_tag = -1

for step in range(NUM_STEPS):
    print(f" Step {step + 1}/{NUM_STEPS} ")
    print(f"Current crack length: {current_crack_length:.2f}")

    gmsh.clear()
    gmsh.model.add(model_name)

    #  Define Geometry
    p1 = gmsh.model.geo.addPoint(0, 0, 0, MESH_SIZE_BOUNDARY)
    p2 = gmsh.model.geo.addPoint(DOMAIN_WIDTH, 0, 0, MESH_SIZE_BOUNDARY)
    p3 = gmsh.model.geo.addPoint(DOMAIN_WIDTH, DOMAIN_HEIGHT, 0, MESH_SIZE_BOUNDARY)
    p4 = gmsh.model.geo.addPoint(0, DOMAIN_HEIGHT, 0, MESH_SIZE_BOUNDARY)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    crack_x_start = CRACK_START_X
    crack_x_end = current_crack_length-CRACK_START_X
    pc1 = gmsh.model.geo.addPoint(crack_x_start, CRACK_Y_POS, 0, MESH_SIZE_NEAR_CRACK)
    pc2 = gmsh.model.geo.addPoint(crack_x_end, CRACK_Y_POS, 0, MESH_SIZE_NEAR_CRACK)

    # Store the tag of the geometric line entity representing the crack
    crack_line_geo_tag = gmsh.model.geo.addLine(pc1, pc2) # *** Get the tag ***

    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    plane_surface = gmsh.model.geo.addPlaneSurface([outer_loop])

    gmsh.model.geo.synchronize()

    #  Define Mesh Size Field
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [crack_line_geo_tag]) # Use stored tag
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", MESH_SIZE_NEAR_CRACK)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", MESH_SIZE_FAR)
    gmsh.model.mesh.field.setNumber(2, "DistMin", ADAPTIVE_RADIUS)
    gmsh.model.mesh.field.setNumber(2, "DistMax", ADAPTIVE_RADIUS * 1.5)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    #  Synchronize, Embed, and Generate Mesh
    gmsh.model.geo.synchronize()
    # Embed the crack line geometry into the surface mesh
    gmsh.model.mesh.embed(1, [crack_line_geo_tag], 2, plane_surface)
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)

    #  Convert Full Mesh to VTK
    new_vtk_ugrid = gmsh_to_vtk(model_name)
    ugrid.DeepCopy(new_vtk_ugrid) # Update the main mesh data

    #  Extract Crack Line Geometry and Update Crack Actor
    new_crack_poly_data = get_crack_line_vtk(crack_line_geo_tag, model_name)
    crack_poly_data.DeepCopy(new_crack_poly_data) # Update the crack line data

    #  Update VTK Visualization
    if step == 0:
        renderer.ResetCamera()
        renderer.GetActiveCamera().Zoom(1.2)
        iren.Initialize()
        renWin.Render()
    else:
        renWin.Render()

    #  Update Crack Length
    current_crack_length += CRACK_GROWTH_INCREMENT
    time.sleep(STEP_DELAY)

#  Final Interaction
print("\nSimulation finished. Starting VTK interaction...")
renWin.Render()

iren.Start()

#  Gmsh Finalization
gmsh.finalize()
print("Gmsh finalized.")