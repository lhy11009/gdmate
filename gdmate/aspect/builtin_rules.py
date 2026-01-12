from pathlib import Path
from .config_engine import Rule

class CasePathRule(Rule):
    """
    Normalize and resolve case-related paths.

    This rule ensures that path-related configuration entries are consistently
    interpreted and propagated into the parameter dictionaries used by ASPECT.

    Relative paths are expected to be resolved against case_path upstream.

    Required configuration parameters:
        wb_file (str):
            Path to the WorldBuilder file.
            May be absolute or relative.
            Default value: "case.wb"

        output_directory (str):
            Path to the directory where ASPECT outputs will be written.
            May be absolute or relative.
            Default value: "output"
    """

    requires = [
        "wb_file",
        "output_directory",
    ]
    
    defaults = {
        "wb_file": "case.wb",
        "output_directory": "output",
    }

    provides = [
    ]

    def apply(self, config, prm_dict, wb_dict, context):
        """
        Apply this rule to update configuration-dependent dictionaries.

        This method follows the standard Rule interface and is responsible for
        modifying the provided dictionaries in-place based on the rule's logic.

        Parameters:
            config (dict): Configuration dictionary containing user-provided and
                           defaulted values.
            prm_dict (dict): Dictionary representing parsed ASPECT .prm parameters
                             that may be modified in-place.
            wb_dict (dict): Dictionary representing parsed WorldBuilder configuration
                            that may be modified in-place.
            context (dict): Shared execution context passed between rules for
                            coordination or data sharing.

        Returns:
            None: This function modifies one or more dictionaries in-place and does
                  not return a value.
        """

        # Get the path of prm and wb files from configuration
        wb_file = Path(config.get("wb_file"))
        output_dir = Path(config.get("output_directory"))

        # Fix entries in the prm file using string representations of the paths
        prm_dict["World builder file"] = str(wb_file)
        prm_dict["Output directory"] = str(output_dir)
