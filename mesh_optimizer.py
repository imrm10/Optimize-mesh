import numpy as np
from matplotlib.path import Path
import pymeshlab
import os
import sys 
import json

argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) >= 3:
    cameraCalib_path = argv[0]
    mesh_path = argv[1]
    folder_path = argv[2]
else:
    print("Usage: python mesh_filtering.py -- <cameraCalib_path> <mesh_path> <output_path> [<num_polygons>]")
    exit(1)

# Optional parameter
num_polygons = int(argv[3]) if len(argv) >= 4 else 0

print('sfm_data path: ', cameraCalib_path)
print('Mesh_path: ', mesh_path)
print('Output_path: ', folder_path)
print('Num polygons (0 = no decimation): ', num_polygons)

if not os.path.exists(folder_path):
    os.mkdir(folder_path)

camera_points = os.path.join(folder_path,"camera_points.obj")
plane = os.path.join(folder_path,"plane.obj")
output_optimizated_mesh = os.path.join(folder_path,"optimizated_mesh.obj")


def load_camera_rotations_from_sfm(json_path): 
    """
      Reads camera rotation matrices from an OpenMVG sfm_data.json file.
      :param json_path: Path to sfm_data.json 
      :return: List of rotation row vectors (each as np.array of shape (1, 3)) 
    """ 
    with open(json_path, 'r') as f: 
        sfm_data = json.load(f) # Build a lookup dict: pose_id -> rotation matrix 
        
        camera_vectors = [] 
        for entry in sfm_data['extrinsics']:
            rotation_matrix=np.array(entry['value']['rotation'])
            camera_vectors.append(rotation_matrix[0].reshape(3,))
        
    return camera_vectors 
            

def project_onto_planes(point, direction, offset):
    """
    Projects a 3D point onto a plane perpendicular to a given direction vector.

    :param point: A 3D point (3,)
    :param direction: Direction vector defining the normal of the plane (3,)
    :param offset: Scalar offset along the direction vector
    :return: Two 3D points representing the projection and an offset projection
    """
    v = np.asarray(direction, dtype=float).flatten()
    v_norm_sq = np.dot(v, v)

    p = np.asarray(point, dtype=float)
    proj_on_v = (np.dot(p, v) / v_norm_sq) * v
    projected_point = p - proj_on_v
    offset_projected_point = projected_point + v * offset

    return projected_point, offset_projected_point

def load_obj(file_path):
    """
    Loads vertices and faces from a Wavefront .obj file.

    :param file_path: Path to the .obj file
    :return: Tuple of numpy arrays (vertices, faces)
    """
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                face = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                faces.append(face)

    return np.array(vertices), np.array(faces)

def save_obj(file_path, vertices, faces):
    """
    Saves a mesh to a Wavefront .obj file.

    :param file_path: Path to save the .obj file
    :param vertices: Nx3 array of vertex coordinates
    :param faces: List of face index lists
    """
    with open(file_path, 'w') as file:
        for v in vertices:
            file.write(f'v {" ".join(map(str, v))}\n')
        for face in faces:
            file.write(f'f {" ".join(str(idx + 1) for idx in face)}\n')

def project_vertices_to_plane(vertices, plane_point, direction):
    """
    Projects 3D vertices onto a plane defined by a point and a direction.

    :param vertices: Nx3 array of vertex positions
    :param plane_point: A point on the plane (3,)
    :param direction: Normal direction of the plane (3,)
    :return: Nx3 array of projected vertices
    """
    direction = direction / np.linalg.norm(direction)
    vectors_to_plane = vertices - plane_point
    distances = np.dot(vectors_to_plane, direction)
    return vertices - np.outer(distances, direction)

def compute_plane_basis(plane_vertices):
    """
    Computes an orthonormal basis for a plane defined by 3 or more vertices.

    :param plane_vertices: Array of at least 3 points on the plane
    :return: Tuple (origin, base_x, base_y)
    """
    v1 = plane_vertices[1] - plane_vertices[0]
    v2 = plane_vertices[2] - plane_vertices[0]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    base_x = v1 / np.linalg.norm(v1)
    base_y = np.cross(normal, base_x)
    origin = plane_vertices[0]

    return origin, base_x, base_y

def to_2d_basis(points, origin, base_x, base_y):
    """
    Converts 3D points to 2D coordinates in a plane basis.

    :param points: Nx3 array of 3D points
    :param origin: Origin point of the 2D basis (3,)
    :param base_x: First axis of the 2D basis (3,)
    :param base_y: Second axis of the 2D basis (3,)
    :return: Nx2 array of 2D coordinates
    """
    relative = points - origin
    x_coords = relative @ base_x
    y_coords = relative @ base_y
    return np.stack((x_coords, y_coords), axis=-1)

def filter_object_by_projection(obj_vertices, obj_faces, plane_vertices, direction):
    """
    Filters a 3D mesh by projecting it onto a plane and checking if it falls within a planar polygon.

    :param obj_vertices: Nx3 array of object vertices
    :param obj_faces: List of face index lists
    :param plane_vertices: Mx3 array defining a convex plane boundary
    :param direction: Projection direction (3,)
    :return: Tuple (filtered_vertices, filtered_faces)
    """
    # Compute 2D basis from the plane vertices
    origin, base_x, base_y = compute_plane_basis(plane_vertices)

    # Project object vertices onto the plane and convert to 2D
    projected_3d = project_vertices_to_plane(obj_vertices, origin, direction)
    projected_2d = to_2d_basis(projected_3d, origin, base_x, base_y)

    # Convert the plane polygon to 2D
    polygon_2d = to_2d_basis(plane_vertices, origin, base_x, base_y)
    polygon_path = Path(polygon_2d)

    # Determine which vertices lie within the polygon
    inside_mask = polygon_path.contains_points(projected_2d)
    inside_indices = set(np.where(inside_mask)[0])

    # Keep faces where all vertices are inside the polygon
    filtered_faces = [face for face in obj_faces if all(idx in inside_indices for idx in face)]

    # Get used vertices and re-map their indices
    used_indices = sorted(set(idx for face in filtered_faces for idx in face))
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
    filtered_vertices = obj_vertices[used_indices]
    remapped_faces = [[index_map[idx] for idx in face] for face in filtered_faces]

    return filtered_vertices, remapped_faces


# Load Camera Rotation Vectors
camera_vectors = load_camera_rotations_from_sfm(cameraCalib_path)

# Compute Custom Axis via Eigenvalue Decomposition
M = sum(np.outer(v, v) for v in camera_vectors)
eigenvalues, eigenvectors = np.linalg.eigh(M)
custom_axis = eigenvectors[:, np.argmin(eigenvalues)]
custom_axis_str = ",".join(f"{x:.6f}" for x in custom_axis)
print(custom_axis_str)

# Project Camera Points onto Custom Plane
direction = [custom_axis]

with open(cameraCalib_path, 'r') as f:
    sfm_data = json.load(f)

with open(camera_points, 'w') as f:
    for entry in sfm_data['extrinsics']:
        centre = entry['value']['center']
        proj1, proj2 = project_onto_planes(centre, direction, offset=-1.0)
        f.write('v ' + ' '.join(f'{x:.7f}' for x in proj1) + '\n')
        f.write('v ' + ' '.join(f'{x:.7f}' for x in proj2) + '\n')


# Generate Convex Hull and Planar Section
mesh_set = pymeshlab.MeshSet()
mesh_set.load_new_mesh(camera_points)

# Create convex hull from projected camera points
mesh_set.generate_convex_hull()

# Generate planar slice using custom axis
mesh_set.generate_polyline_from_planar_section(
    planeaxis='Custom Axis',
    customaxis=custom_axis,
    createsectionsurface=True,
    relativeto='Bounding box center'
)

# Save the generated plane
mesh_set.save_current_mesh(
    file_name=plane,
    save_vertex_color=False,
    save_vertex_coord=False,
    save_vertex_normal=False,
    save_face_color=False,
    save_wedge_texcoord=False,
    save_wedge_normal=False,
    save_polygonal=False
)

os.remove(camera_points)  # Clean up intermediate camera points file

# Filter Main Object with the Generated Plane
output_path_cut = os.path.join(folder_path, "filtered_object.obj")
projection_direction = np.asarray(direction).reshape(3,)

obj_v, obj_f = load_obj(mesh_path)
plane_v, _ = load_obj(plane)

filtered_v, filtered_f = filter_object_by_projection(obj_v, obj_f, plane_v, projection_direction)
save_obj(output_path_cut, filtered_v, filtered_f)

# Clean and Isolate Largest Mesh Component
mesh_set = pymeshlab.MeshSet()
mesh_set.load_new_mesh(output_path_cut)
mesh_set.compute_selection_by_small_disconnected_components_per_face()
mesh_set.meshing_remove_selected_vertices_and_faces()
mesh_set.generate_splitting_by_connected_components(delete_source_mesh=True)

# Keep only the largest mesh
largest_vertices = 0
largest_id = -1
for mesh in mesh_set:
    if mesh.vertex_number() > largest_vertices:
        largest_vertices = mesh.vertex_number()
        largest_id = mesh.id()
mesh_set.set_current_mesh(largest_id)

if num_polygons > 0:
    # Apply Quadric Edge Collapse Decimation
    mesh_set.meshing_decimation_quadric_edge_collapse(
        targetfacenum = int(num_polygons),
        targetperc = 0,
        qualitythr = 0.3,
        preserveboundary = False,
        boundaryweight = 1,
        preservenormal = False,
        preservetopology  = False,
        optimalplacement = True,
        planarquadric = False,
        planarweight = 0.001,
        qualityweight = False,
        autoclean  = True,
        selected  = False
    )

mesh_set.save_current_mesh(
    file_name=output_optimizated_mesh,
    save_vertex_color=False,
    save_vertex_coord=False,
    save_vertex_normal=False,
    save_face_color=False,
    save_wedge_texcoord=False,
    save_wedge_normal=False,
    save_polygonal=False
)
os.remove(output_path_cut)  # Clean up intermediate filtered object file
