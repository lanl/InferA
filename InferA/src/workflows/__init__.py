import os
import importlib

# Get the current package name
package_name = __name__

# List all .py files in the same directory (except __init__.py)
current_dir = os.path.dirname(__file__)
module_files = [
    f for f in os.listdir(current_dir)
    if f.endswith('.py') and f != '__init__.py'
]

# Derive module names without the .py extension
module_names = [os.path.splitext(f)[0] for f in module_files]

# Import each module
for module in module_names:
    importlib.import_module(f"{package_name}.{module}")

# Optionally: Define __all__ to expose only selected symbols
__all__ = module_names  # or list symbols like ['task_a', 'task_b']