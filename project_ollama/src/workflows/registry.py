# Shared object_type description to avoid repetition
object_type_description = """\
object_type - Description: object_type to analyze. Object name must be explicitly stated by user. Ex. "Visualize largest object" does not contain an object_type.
    Object can be one of the following: [halo][galaxy][accumulatedcore][haloparticles][galaxyparticles][bighaloparticles][sodbighaloparticles].
    For clarification:
        - [halo] and [galaxy] represent the ID of the halo cluster and galaxy cluster.
        - [haloparticles] and [galaxyparticles] represent the individual particles that make up each [halo] and [galaxy]."""

halo_id_description = "halo_id - Description: ID of halo to analyze."

object_id_description = "object_id - Description: ID of object to analyze."

timestep_description = "timestep - Description: int timestep to analyze."

n_description = "n - Description: int number of largest objects to find."

use_visual = "use_visual - Description: Does user want visualization? True only if user explicitly asks for visualizing, else False."


required_fields_by_task = {
    "find_largest_object": [
        object_type_description,
        timestep_description,
        n_description,
        use_visual,
    ],
    "find_largest_within_halo": [
        object_type_description,
        timestep_description,
        halo_id_description,
        n_description,
        use_visual,
    ],
    "track_object_evolution": [
        object_type_description,
        object_id_description,
        timestep_description,
        use_visual,
    ],
    # "region_visualization": [
    #     object_type_description,
    #     timestep_description,
    # ],
    # "compare_largest_objects": [
    #     object_type_description,
    #     timestep_description,
    # ],
    # "compare_simulation_values": [
    #     object_type_description,
    #     timestep_description,
    # ],
    # "compare_object_evolution": [
    #     object_type_description,
    #     timestep_description,
    # ],
    # "compare_object_evolution_multiple": [
    #     object_type_description,
    #     timestep_description,
    # ],
}
