"""
Centralized chemical variable classification and weights based on the provided figure.

- VARIABLE_TYPE_BY_NAME maps each variable to its type (exact strings as shown in the figure).
- TYPE_WEIGHTS assigns a numeric weight to each variable type for weighted PCA selection.
"""

# Exact mapping from "Variable Name" to "Type of Variable" (from the figure)
VARIABLE_TYPE_BY_NAME = {
    # Elements
    "Al": "Earth element (nontoxic)",
    "As": "Trace Metal (pollutant)",
    "Bi": "Trace Metal (pollutant)",
    "Ca": "Earth element (nontoxic)",
    "Cd": "Trace Metal (pollutant)",
    "Co": "Trace Metal (pollutant)",
    "Cr": "Trace Metal (pollutant)",
    "Cu": "Trace Metal (pollutant)",
    "Fe": "Earth element (nontoxic)",
    "Hg": "Trace Metal (pollutant)",
    "K": "Earth element (nontoxic)",
    "Mg": "Earth element (nontoxic)",
    "Mn": "Trace Metal (pollutant)",
    "Na": "Earth element (nontoxic)",
    "Ni": "Trace Metal (pollutant)",
    "Pb": "Trace Metal (pollutant)",
    "Sb": "Trace Metal (pollutant)",
    "V": "Trace Metal (pollutant)",
    "Zn": "Trace Metal (pollutant)",

    # Binding agent
    "OC": "Binding agent",

    # Hydrocarbon pollutants (chlorobenzenes, etc.)
    "1245TCB": "Hydrocarbon pollutant",
    "1234TCB": "Hydrocarbon pollutant",
    "QCB": "Hydrocarbon pollutant",
    "HCB": "Hydrocarbon pollutant",
    "OCS": "Hydrocarbon pollutant",

    # Organochlorine pesticides
    "ppDDE": "organochlorine pesticide",
    "ppDDD": "organochlorine pesticide",
    "mirex": "organochlorine pesticide",
    "Heptachlor_Epoxide": "organochlorine pesticide",

    # Sum of PCBs
    "total_PCB": "Sum of all PCBs",
}

# Weights by variable type
TYPE_WEIGHTS = {
    "Trace Metal (pollutant)": 3.0,          # Highest priority: toxic metals
    "Hydrocarbon pollutant": 3.0,            # Highest priority: petroleum/chlorobenzenes
    "organochlorine pesticide": 3.0,         # Highest priority: POPs
    "Sum of all PCBs": 3.0,                  # Highest priority: PCBs
    "Binding agent": 2,                    # Medium priority: affects bioavailability
    "Earth element (nontoxic)": 1.5         # Lowest priority: background elements
}


def get_variable_weight(name: str) -> float:
    """Return weight for a variable name using the mappings above."""
    var_type = VARIABLE_TYPE_BY_NAME.get(name)
    if var_type is None:
        return 1.0
    return TYPE_WEIGHTS.get(var_type, 1.0)


def build_weights_for_columns(columns) -> dict:
    """
    Build a per-variable weight dict for a sequence of column names.
    Unknown variables default to 1.0.
    """
    return {name: get_variable_weight(name) for name in columns}
