import os
import zipfile
import numpy as np
import pyvista as pv
import logging

from src.utils import genericio_utils as gio_utils

from src.workflows.base_workflow import BaseWorkflow

logger = logging.getLogger(__name__)

output_dir = "output/"


class VisualizeObject(BaseWorkflow):
    """
    Class for visualizing objects (halos or galaxies) in a 3D region,
    rendering points from data files, and saving plots in VTK (.vtp) or HTML format.
    """
    def __init__(self, base_path, full_path):
        """
        Initialize with base path for data files and full path if needed.
        
        Args:
            base_path (str): Base directory containing data files.
            full_path (str): Full path used for some file operations.
        """
        self.base_path = base_path
        self.full_path = full_path


    def run(self, df, object_type, timestep, output_name = "plot"):
        """
        Main entry to visualize objects of specified type at a timestep.

        Args:
            df (pd.DataFrame): DataFrame with object data including coordinates and radius.
            object_type (str): 'halo' or 'galaxy'.
            timestep (int): Timestep index for data files.
            output_name (str): Base name for output files.

        Returns:
            str: Path to saved output plot file (.vtp or .html).
        """
        if object_type == "halo":
            x_col, y_col, z_col, r_col = "fof_halo_center_x", "fof_halo_center_y", "fof_halo_center_z", "sod_halo_radius"
            selection = 1
        elif object_type == "galaxy":
            x_col, y_col, z_col, r_col = "gal_center_x", "gal_center_y", "gal_center_z", "gal_radius"
            selection = 2
        else:
            raise ValueError("Unsupported object_type: must be 'halo' or 'galaxy'")

        # Calculate bounding box coordinates        
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        z_min, z_max = df[z_col].min(), df[z_col].max()

        r_max = df[r_col].max() if r_col in df else 0

        # Compute center and radius for region
        center = [
            0.5 * (x_min + x_max),
            0.5 * (y_min + y_max),
            0.5 * (z_min + z_max)
        ]

        r = max(x_max - x_min, y_max - y_min, z_max - z_min)/2
        radius = r if r > 20 else 20 # Minimum radius is 20
        buffer = r_max + 1

        logger.debug(f"[VisualizeObject] Visual parameters:\n"
                     f"center = {center}\n"
                     f"radius = {radius}\n"
                     f"buffer = {buffer}\n"
                     f"selection = {'halo' if selection == 1 else 'galaxy'}")

        # Get background points for region rendering
        points_to_plot_bg = self.render_region_only(
            timestep=timestep,
            selection=selection,
            center=center,
            radius=radius,
            buffer = buffer
        )
        
        # Target points from dataframe
        points_to_plot_target = np.column_stack((df[x_col], df[y_col], df[z_col]))

        # Generate plot and return output filename
        region_html = self.plot(points_to_plot_bg, points_to_plot_target, center, radius, output_name = output_name)
        return region_html


    def plot(self, points_to_plot_bg, points_to_plot_target, center, radius, output_name, save_as_vtp = True):
        """
        Create a visualization plot from background and target points.
        Saves either as VTK PolyData (.vtp) or HTML interactive plot.

        Args:
            points_to_plot_bg (np.ndarray): Background points (N x 3).
            points_to_plot_target (np.ndarray): Target points (M x 3).
            center (list): Center coordinates for coloring/scaling.
            radius (float): Radius used for opacity calculation.
            output_name (str): Base output filename.
            save_as_vtp (bool): If True, save as .vtp; else save as .html.

        Returns:
            str: Path to the saved plot file.
        """
        opacity = max(0.05, min(1, -0.00375 * radius + 0.475))

        plotter = pv.Plotter(off_screen=True)

        data_vis_bg = pv.PolyData(points_to_plot_bg)
        data_vis_target = pv.PolyData(points_to_plot_target)

        if save_as_vtp:
            full_output_name = output_dir + output_name + ".vtp"

            # Combine points
            all_points = np.vstack([points_to_plot_bg, points_to_plot_target])
            combined = pv.PolyData(all_points)

            # Create a color array
            colors1 = np.tile([0, 0, 255], (len(points_to_plot_bg), 1))
            colors2 = np.tile([255, 0, 0], (len(points_to_plot_target), 1))
            all_colors = np.vstack([colors1, colors2]).astype(np.uint8)

            # color array to combined points
            combined["Colors"] = all_colors

            # group_id: 0 for bg, 1 for target
            group_ids = np.array([0]*len(points_to_plot_bg) + [1]*len(points_to_plot_target))
            combined["group_id"] = group_ids

            # Save to .vtp file
            combined.save(full_output_name)

        else:
            # Set to save as .html instead of .vtp
            full_output_name = output_dir + output_name + ".html"
            distances = np.linalg.norm(points_to_plot_bg - center, axis = 1)
            data_vis_bg['distance'] = distances

            # plotter.add_mesh(data_vis_bg, point_size = 1.5, color= 'black', opacity=opacity)
            plotter.add_mesh(data_vis_bg, point_size = 1.5, opacity = opacity, scalars='distance', cmap = 'viridis', scalar_bar_args={'title': 'Distance from center', 'vertical': True})
            plotter.add_mesh(data_vis_target, point_size = 10, color= 'red', opacity=1.0)

            plotter.export_html(full_output_name)

        plotter.close()
        del plotter
        return full_output_name


    def render_region_only(self, timestep, selection, center, radius, buffer):
        """
        Extract 3D points within a region from halo or galaxy property files.

        Args:
            timestep (int): Timestep index for data files.
            selection (int): 1 for halo, 2 for galaxy.
            center (list): Center coordinates of region.
            radius (float): Radius of region.
            buffer (float): Additional buffer distance.

        Returns:
            np.ndarray: Points array (N x 3) with x,y,z coordinates.
        """
        # # selection 0 does not really work because file becomes too big to write
        # if selection == 0:
        #     file = self.full_path + f"/m000p.full.mpicosmo.{timestep}"
        #     output_vars = ['x', 'y', 'z', 'mass', 'rho', 'phi']

        #     points_region = gio_utils.get_points_region(file, output_vars, center, radius, buffer)
        #     x, y, z = points_region[0], points_region[1], points_region[2]

        if selection == 1:
            file = self.base_path + f"/m000p-{timestep}.haloproperties"
            halo_vars = ['fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z']

            points_region = gio_utils.get_points_region(file, halo_vars, center, radius, buffer)

            x, y, z = points_region[0], points_region[1], points_region[2]

        if selection == 2:
            file = self.base_path + f"/m000p-{timestep}.galaxyproperties"
            galaxy_vars = ["gal_center_x", "gal_center_y", "gal_center_z"]

            points_region = gio_utils.get_points_region(file, galaxy_vars, center, radius, buffer)
            x, y, z = points_region[0], points_region[1], points_region[2]

        points_to_plot = np.column_stack((x,y,z)) # Get the x,y,z

        return points_to_plot

    
    def generate_pvd_file(self, file_list, pvd_path):
        """
        Generate a ParaView .pvd collection file listing all timestep files.

        Args:
            file_list (list): List of file paths (typically .vtp files).
            pvd_path (str): Output path for the .pvd file.
        """
        # Sort files by timestep number extracted from filename (assuming timestep is numeric in filename)
        def extract_timestep(filepath):
            base = os.path.basename(filepath)
            name, _ = os.path.splitext(base)
            try:
                return int(name)
            except ValueError:
                # fallback if name is not just number
                return 0

        sorted_files = sorted(file_list, key=extract_timestep)

        with open(pvd_path, "w") as pvd:
            pvd.write('<?xml version="1.0"?>\n')
            pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            pvd.write('  <Collection>\n')
            for i, filepath in enumerate(sorted_files):
                filename = os.path.basename(filepath)
                # Skip the .pvd itself if somehow in list
                if filename.endswith(".pvd"):
                    continue
                pvd.write(f'    <DataSet timestep="{i}" group="" part="0" file="{filename}"/>\n')
            pvd.write('  </Collection>\n')
            pvd.write('</VTKFile>\n')


    def create_zip_archive(self, file_list, output_dir, zip_name="output.zip"):
        """
        Create a ZIP archive containing specified files.

        Args:
            file_list (list): List of file paths to include.
            output_dir (str): Directory to save the ZIP archive.
            zip_name (str): Name of the ZIP file.

        Returns:
            str: Path to the created ZIP archive.
        """
        zip_path = os.path.join(output_dir, zip_name)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for filepath in file_list:
                if os.path.isfile(filepath):
                    # Add file with just its filename (no directories)
                    zipf.write(filepath, arcname=os.path.basename(filepath))
                else:
                    logger.warning(f"File not found, skipping: {filepath}")
        return zip_path
    

def generate_edge_points_tetrahedron(n_points_per_edge=500, jitter=5):
    """
    Generate points near the edges of a tetrahedron with jitter to give thickness.
    
    Parameters:
        n_points_per_edge (int): Number of points to sample per edge.
        jitter (float): Jitter magnitude perpendicular to edge direction.
    
    Returns:
        pyvista.PolyData: Point cloud focused along tetrahedron edges.
    """
    # Define tetrahedron vertices
    v0 = np.array([1, 1, 1]) * 30
    v1 = np.array([-1, -1, 1]) * 30
    v2 = np.array([-1, 1, -1]) * 30
    v3 = np.array([1, -1, -1]) * 30
    vertices = [v0, v1, v2, v3]

    # Define edges (pairs of vertex indices)
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3)
    ]

    all_points = []

    for i, j in edges:
        start = vertices[i]
        end = vertices[j]
        direction = end - start
        length = np.linalg.norm(direction)
        unit_dir = direction / length

        # Create orthonormal basis for jitter plane perpendicular to the edge
        if np.allclose(unit_dir, [0, 0, 1]):
            perp1 = np.cross(unit_dir, [1, 0, 0])
        else:
            perp1 = np.cross(unit_dir, [0, 0, 1])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(unit_dir, perp1)

        # Sample points along the edge
        t_values = np.random.rand(n_points_per_edge)
        for t in t_values:
            point_on_edge = start + t * direction
            jitter_offset = (
                np.random.normal(scale=jitter) * perp1 +
                np.random.normal(scale=jitter) * perp2
            )
            jittered_point = point_on_edge + jitter_offset
            all_points.append(jittered_point)

    all_points = np.array(all_points)
    return all_points