import sys
import os
import pandas as pd
import numpy as np
import typer

from src.utils import genericio_utils as gio_utils
from src.deprecated import analysis_functions
from src.deprecated import preprocess_data

app = typer.Typer()


# Command to handle the user interaction
@app.command()
def main(option: int):
    """
    A simple menu-driven program that performs astrophysical data analysis
    and rendering operations based on user input.
    
    Parameters:
        option (int): Menu option selected by the user (1–5).
    """
    # Option 1: Find the n largest halos in a given timestep
    if option == 1:
        # Prompt user for the timestep (X) and number of halos (n)
        X = typer.prompt("Enter timestep", type=int)
        n = typer.prompt("Enter number of halos", type=int)

        result = analysis_functions.get_n_largest_halo_timestep_X(X,n)
        typer.echo(f"The result is: {result}")

    # Option 2: Find the n largest galaxies in a given timestep
    elif option == 2:
        # Prompt user for the timestep (X) and number of galaxies (n)
        X = typer.prompt("Enter timestep", type=int)
        n = typer.prompt("Enter number of galaxies", type=int)

        result = analysis_functions.get_n_largest_galaxies_timestep_X(X,n)
        typer.echo(f"The result is: {result}")

    # Option 3: Find n galaxies within a specific halo at a given timestep
    elif option == 3:
        # Prompt user for timestep (X), halo tag (tag), and number of galaxies (n)
        X = typer.prompt("Enter timestep", type=int)
        tag = typer.prompt("Enter halo_tag to search", type=int)
        n = typer.prompt("Enter number of galaxies", type=int)

        result = analysis_functions.get_n_galaxies_in_timestep_halo(X, tag, n)
        typer.echo(f"The result is: {result}")
    
    # Option 4: Render a subsampled dataset from a given timestep
    elif option == 4:
        # Prompt for timestep, object type to render, and subsample rate
        X = typer.prompt("Enter timestep", type=int)
        t = typer.prompt("Object to render - 0: Full output, 1: Halos, 2: Galaxies", type=int)
        subsample = typer.prompt("Subsample rate - between 0 and 1", type=float)

        output_file = analysis_functions.render_subsampled_dataset(498, 0.1, 1)
        typer.echo(f"Result saved to: {output_file}")

    # Option 5: Render a dataset limited to a specified region in space
    elif option == 5:
        # Prompt for timestep, object type, region center coordinates, radius, and buffer
        X = typer.prompt("Enter timestep", type=int)
        t = typer.prompt("Object to render - 0: Full output, 1: Halos, 2: Galaxies", type=int)
        x = typer.prompt("Region center - x coordinate", type=float)
        y = typer.prompt("Region center - y coordinate", type=float)
        z = typer.prompt("Region center - z coordinate", type=float)
        r = typer.prompt("Region radius", type=float)
        b = typer.prompt("Region buffer", type=float)

        output_file = analysis_functions.render_region_only_dataset(X, t, [x,y,z], r, b)
        typer.echo(f"Result saved to: {output_file}")


    elif option == 6:
        print("Run ")

    else:
        typer.echo("Invalid option selected. Please choose a valid number between 1-2.")


if __name__ == "__main__":
    app()