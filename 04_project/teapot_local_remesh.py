import numpy as np
import vtk
import subprocess
from numpy import sin, arctan
import gmsh
import sys
import time
import pyvista as pv
import re
import statistics as st
import meshio



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


CRACK_COLOR = (1.0, 0.0, 0.0)
CRACK_LINE_WIDTH = 3.0


mmg_exe = "mmg2d_O3"
mmg_exe3d = "mmgs_O3"




WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
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
renderer.SetBackground(0.1, 0.2, 0.4) # Background color

def read_mesh_data(meshfile):
    """Читает вершины и треугольники из файла .mesh формата mmg."""
    nodes_coords = []
    triangles_connectivity = []
    reading_section = None
    num_items = 0
    item_count = 0
    tag = []
    triag_tags = []


    with open(meshfile, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line == "Vertices":
                reading_section = "Vertices"
                num_items = int(file.readline().strip())
                nodes_coords = np.zeros((num_items, 3)) # Pre-allocate NumPy array
                item_count = 0
                continue
            elif line == "Triangles":
                reading_section = "Triangles"
                num_items = int(file.readline().strip())
                # Store as list first, easier to append
                triangles_connectivity = []
                item_count = 0
                continue
            elif line == "End":
                reading_section = None
                continue

            if reading_section == "Vertices":
                if item_count < num_items:
                    parts = line.split()
                    # Формат: x y ref (tag)
                    nodes_coords[item_count] = [float(parts[0]), float(parts[1]), float(parts[2])]
                    item_count += 1
                    tag.append(item_count)

                if item_count == num_items:
                     reading_section = None

            elif reading_section == "Triangles":
                 if item_count < num_items:
                    parts = line.split()
                    # Формат: n1 n2 n3 ref (tag)
                    # mmg использует 1-based индексы, VTK - 0-based
                    n1 = int(parts[0])
                    n2 = int(parts[1])
                    n3 = int(parts[2])
                    triangles_connectivity.append([n1, n2, n3])
                    triag_tags.append(int(parts[3]))
                    item_count += 1
                 if item_count == num_items:
                     reading_section = None



    return nodes_coords, triangles_connectivity, tag, triag_tags

def mesh_to_vtk(meshfile):
    """Создает vtkUnstructuredGrid из файла .mesh."""
    nodes_coords, triangles_connectivity = read_mesh_data(meshfile)

    if nodes_coords is None or len(nodes_coords) == 0:
        print(f"Не удалось создать VTK сетку: нет данных о вершинах из {meshfile}")
        return None # Возвращаем None, если чтение узлов не удалось

    vtk_grid = vtk.vtkUnstructuredGrid()

    # 1. Добавляем точки (Vertices)
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(nodes_coords))
    for i, coord in enumerate(nodes_coords):
        points.SetPoint(i, coord[0], coord[1], coord[2])
    vtk_grid.SetPoints(points)

    # 2. Добавляем ячейки (Triangles)
    if triangles_connectivity and len(triangles_connectivity) > 0:
         num_tris = len(triangles_connectivity)
         # Предварительное выделение памяти для ячеек
         vtk_grid.Allocate(num_tris, num_tris * 3 // 2) # Примерный размер

         for tri_nodes in triangles_connectivity:
             vtk_tri = vtk.vtkTriangle()
             point_ids = vtk_tri.GetPointIds()

             # Проверка валидности индексов узлов
             valid_triangle = True
             for j in range(3):
                 node_id = tri_nodes[j]
                 # Проверяем, что индекс узла находится в допустимом диапазоне
                 if 0 <= node_id < len(nodes_coords):
                     point_ids.SetId(j, node_id)
                 else:
                     print(f"Предупреждение: Неверный индекс узла {node_id+1} в треугольнике файла {meshfile}. Пропуск треугольника.")
                     valid_triangle = False
                     break # Выходим из внутреннего цикла for j

             if valid_triangle:
                vtk_grid.InsertNextCell(vtk_tri.GetCellType(), point_ids)

    elif len(nodes_coords) > 0: # Если есть узлы, но нет треугольников
        print(f"Предупреждение: Треугольники не найдены или не прочитаны из {meshfile}. Сетка будет содержать только точки.")
        # Можно добавить создание vtkVertex для каждой точки, если нужно их видеть
        # vtk_grid.Allocate(len(nodes_coords))
        # for i in range(len(nodes_coords)):
        #     vertex = vtk.vtkVertex()
        #     vertex.GetPointIds().SetId(0, i)
        #     vtk_grid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())


    return vtk_grid # Возвращаем созданный объект VTK





def f(x, y):
    # solution function
    # change this function as needed
    return 0.5 * sin(50 * x) + 10 * arctan(0.1 / (sin(5 * y) - 2 * x))


def hessian_matrix(x, y):
    # Note: In a real application, the Hessian needs to be
    #       computed numerically using solution at the nodes.
    # Here, partial derivatives are approximated numerically (of a known function) for convenience.

    # replace with analytical functions, if necessary.
    dx = 1e-6
    dy = 1e-6
    d2f_dx2 = (f(x + dx, y) - 2 * f(x, y) + f(x - dx, y)) / (dx * dx)
    d2f_dy2 = (f(x, y + dy) - 2 * f(x, y) + f(x, y - dy)) / (dy * dy)
    d2f_dxdy = (f(x + dx, y + dy) - f(x + dx, y - dy) -
                f(x - dx, y + dy) + f(x - dx, y - dy)) / (4 * dx * dy)

    return np.array([[d2f_dx2, d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])


# main function
def main():
    model_name = "crack_model"
    gmsh.clear()
    gmsh.model.add(model_name)
    current_crack_length = INITIAL_CRACK_LENGTH

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
    crack_x_end = current_crack_length - CRACK_START_X
    pc1 = gmsh.model.geo.addPoint(crack_x_start, CRACK_Y_POS, 0, MESH_SIZE_NEAR_CRACK)
    pc2 = gmsh.model.geo.addPoint(crack_x_end, CRACK_Y_POS, 0, MESH_SIZE_NEAR_CRACK)

    # Store the tag of the geometric line entity representing the crack
    crack_line_geo_tag = gmsh.model.geo.addLine(pc1, pc2)  # *** Get the tag ***

    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    plane_surface = gmsh.model.geo.addPlaneSurface([outer_loop])

    gmsh.model.geo.synchronize()

    #  Define Mesh Size Field
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [crack_line_geo_tag])  # Use stored tag
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


    curves = gmsh.model.getEntities(dim=1)
    curve_tags = [c[1] for c in curves]
    cl = gmsh.model.geo.addCurveLoop(curve_tags)
    s = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, curve_tags, 1, name="Boundary")
    gmsh.model.addPhysicalGroup(2, [s], 1, name="Domain")
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.Format", 2)
    gmsh.write("initial_mesh.mesh")




    # Note: In a real simulation, all the interactions with mmg will be through
    #       api calls and not using executables.
    meshfile_path = generate_starting_mesh()

    # Note: The following mesh adaptations can be done in a loop :)

    # first adaptation
    adapt_mesh_using_mmg(meshfile_path)

    # second adaptation (with metric computed on adapted mesh)
    meshfile_path = meshfile_path.replace(".mesh", ".o.mesh")
    adapt_mesh_using_mmg(meshfile_path)

    # third adaptation (with metric computed on adapted mesh)
    meshfile_path = meshfile_path.replace(".mesh", ".o.mesh")
    adapt_mesh_using_mmg(meshfile_path)

    # fourth adaptation (with metric computed on adapted mesh)
    meshfile_path = meshfile_path.replace(".mesh", ".o.mesh")
    adapt_mesh_using_mmg(meshfile_path)
    out_mesh_file = "teapot.o.mesh"








def adapt_mesh_using_mmg(meshfile):
    nodes = read_mesh_nodes(meshfile)

    # Step 1: compute and write metric to sol file
    solution_file = meshfile.replace(".mesh", ".sol")
    with open(solution_file, "w") as file:
        file.write("MeshVersionFormatted 2" + "\n")
        file.write("Dimension 2" + "\n")
        file.write("SolAtVertices" + "\n")
        file.write(f"{len(nodes)}" + "\n")
        file.write(f"1 3" + "\n")
        for (x, y, z) in nodes:
            m = metric(x, y)
            print(m)
            file.write(f"{m[0, 0]} {m[0, 1]} {m[1, 1]}" + "\n")
        file.write("End" + "\n")

    # Step 2: Call mmg for mesh adaptation
    subprocess.call([mmg_exe,
                     '-in', meshfile,
                     '-sol', solution_file])


def metric(x, y):
    # A simple definition of metric, for demonstration purposes only!
    # In a real appliation, you also need to
    #      modify the metric based on edge-size constraints, aspect ratio limit, number of elements etc.
    hessian = hessian_matrix(x, y)
    # make positive definite
    evals, evecs = np.linalg.eig(hessian)
    abs_evals = np.clip(np.abs(evals), 1e-8, 1e+8)
    return evecs * abs_evals @ np.linalg.inv(evecs)


def generate_starting_mesh():
    # Note: you can use other software for mesh generation (such as gmsh)
    # Here, we are using mmg for mesh generation to keep it simple (and stupid!).

    input_geometry = "input.mesh"
    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]  # edge connectivity
    with open(input_geometry, "w") as file:
        file.write("MeshVersionFormatted 2" + "\n")
        file.write("Dimension 2" + "\n")
        file.write("Vertices" + "\n")
        file.write(f"{len(vertices)}" + "\n")
        for (x, y) in vertices:
            file.write(f"{x:10.5f} {y:10.5f} {0:5d}" + "\n")

        file.write("Edges" + "\n")
        file.write(f"{len(edges)}" + "\n")
        # use as per your boundary tags. Here using arbitrary tag number 7
        edge_tag = 7
        for (start, end) in edges:
            file.write(f"{start} {end} {edge_tag}" + "\n")
        file.write("End" + "\n")

    starting_mesh = "output.mesh"
    # call mmg to generate initial mesh
    subprocess.call([mmg_exe,
                     '-hmax', '0.2',
                     '-in', input_geometry,
                     '-out', starting_mesh])
    return starting_mesh


def read_mesh_nodes(meshfile):
    nodes = []
    with open(meshfile) as file:
        for line in file:
            if line.strip() == "Vertices":
                break

        num_nodes = int(file.readline())
        for i in range(num_nodes):
            x, y, tag = file.readline().split()
            nodes.append((float(x), float(y), 0.))
    nodes = np.array(nodes)
    nodes = nodes.reshape(-1,3)
    return nodes

def distance(v1, v2, v3, point):
    triangle_center = np.array([st.mean([v1[0], v2[0], v3[0]]), st.mean([v1[1], v2[1], v3[1]]), st.mean([v1[2], v2[2], v3[2]])])
    point = np.array(point)
    return np.linalg.norm(triangle_center - point)



def set_references(radius, center, meshfile):
    nodes, triangles, tags, ref = read_mesh_data(meshfile)
    with open(meshfile.replace('.mesh', '.ref.mesh'), 'w') as file:
        with open(meshfile, 'r') as source:
            check = False
            for line in source:
                if 'Triangles' in re.split(r'[\n\s]', line): break
                file.write(line)
            file.write('\n' + ' Triangles' + '\n')
            file.write(str(len(triangles)) + '\n')
            for t in triangles:
                if distance(nodes[t[0]-1], nodes[t[1]-1], nodes[t[2]-1], center) < radius:
                    file.write(' '+str(t[0]) +' '+ str(t[1]) +' '+ str(t[2]) +' 2' + '\n')
                else:
                    file.write(' '+str(t[0]) +' '+ str(t[1]) +' '+ str(t[2]) +' 3' + '\n')
            file.write('\n')
            file.write('End')
            file.write('\n')












def rewrite_data_for_adapt(meshfile):
    with open(meshfile.replace('.mesh', '.freeze.mesh'), 'w') as file:
        with open(meshfile, 'r') as source:
            triangles = []
            check = False
            for line in source:
                if 'End' in re.split(r'[\n\s]', line): break
                if 'Triangles' in re.split(r'[\n\s]', line): check = True
                if check: triangles.append(line)
                file.write(line)
            file.write("RequiredTriangles" + "\n")
        selection = []
        for i in range(2, int(triangles[1])):
            if int(triangles[i][-2]) == 3:
                selection.append(i-1)
        file.write(str(len(selection)) + '\n')
        for triangle in selection:
            file.write(str(triangle)+'\n')
        file.write('\n')
        file.write('End'+'\n'+'\n')

    return triangles



def prepate_for_convert(meshfile):
    with open(meshfile.replace('.mesh', '.convert.mesh'), 'w') as file:
        with open(meshfile, 'r') as source:
            reading_section = None
            num_items = 0
            for line in source:
                if 'RequiredEdges' in re.split(r'[\n\s]', line):
                    reading_section = "RequiredEdges"

                elif 'RequiredTriangles' in re.split(r'[\n\s]', line):
                    reading_section = "RequiredTriangles"

                elif 'Triangles' in re.split(r'[\n\s]', line):
                    reading_section = 'Triangles'


                if reading_section == 'RequiredEdges' or reading_section == "RequiredTriangles":
                    continue
                else:
                    file.write(line)
    return meshfile.replace('.mesh', '.convert.mesh')







# subprocess.call([mmg_exe3d,
#                      '-in', "teapot.mesh",
#                      '-out', "teapot.o.mesh",
#                      '-sol', "cube-distance.sol",
#                      '-ls'])
# nodes, triangles, tags, ref = read_mesh_data("teapot.o.mesh")
# print(nodes[triangles[2][1]])

# tri = rewrite_data_for_adapt("teapot.o.mesh")

steps = 100
dr = 0.002
r0 = 0.4
center = [1.7,0,1]
input_meshfile = 'teapot.mesh'

for i in range(steps):

    set_references(r0+i*dr, center, input_meshfile)
    ref_set_meshfile = input_meshfile.replace('.mesh', '.ref.mesh')
    rewrite_data_for_adapt(ref_set_meshfile)
    frozen_edges_meshfile = ref_set_meshfile.replace('.mesh', '.freeze.mesh')
    output_meshfile = 'teapot.o' + '.o'*i + '.mesh'
    output_vtufile = 'vtu_teapot.o' + '-step-' +str(i) + '.vtu'
    subprocess.call([mmg_exe3d,
                         '-in', frozen_edges_meshfile,
                         '-out', output_meshfile,
                         '-hsiz', '0.02', '-hgradreq', '-1'])




    mesh = meshio.read(prepate_for_convert(output_meshfile))
    meshio.write(output_vtufile, mesh)



# set_references(0.4, center, 'teapot.mesh')
# rewrite_data_for_adapt('teapot.ref.mesh')
# subprocess.call([mmg_exe3d,
#                          '-in', 'teapot.',
#                          '-out', output_meshfile,
#                          '-hsiz', '0.02', '-hgradreq', '-1'])

# Создаем VTK сетку из файла
# new_vtk_ugrid = mesh_to_vtk(out_mesh_file)
# ugrid.DeepCopy(new_vtk_ugrid) # Теперь new_vtk_ugrid - это vtkUnstructuredGrid
#
# renderer.ResetCamera()
# renderer.GetActiveCamera().Zoom(1.2)
# iren.Initialize()
# renWin.Render()
# time.sleep(STEP_DELAY)


# out_mesh_file = "output.o.o.mesh"
#
# # Создаем VTK сетку из файла
# new_vtk_ugrid = mesh_to_vtk(out_mesh_file)
# ugrid.DeepCopy(new_vtk_ugrid) # Теперь new_vtk_ugrid - это vtkUnstructuredGrid
# renWin.Render()
# time.sleep(STEP_DELAY)
#
# out_mesh_file = "output.o.o.o.mesh"
#
# # Создаем VTK сетку из файла
# new_vtk_ugrid = mesh_to_vtk(out_mesh_file)
# ugrid.DeepCopy(new_vtk_ugrid) # Теперь new_vtk_ugrid - это vtkUnstructuredGrid
# renWin.Render()
# time.sleep(STEP_DELAY)
#
# out_mesh_file = "output.o.o.o.o.mesh"
#
# # Создаем VTK сетку из файла
# new_vtk_ugrid = mesh_to_vtk(out_mesh_file)
# ugrid.DeepCopy(new_vtk_ugrid) # Теперь new_vtk_ugrid - это vtkUnstructuredGrid
# renWin.Render()


# iren.Start()