"""
Centralized chemical variable classification and weights based on the provided figure.

- VARIABLE_TYPE_BY_NAME maps each variable to its type (exact strings as shown in the figure).
- TYPE_WEIGHTS assigns a numeric weight to each variable type for weighted PCA selection.
"""

# Exact mapping from "Variable Name" to "Type of Variable" (from the figure)
VARIABLE_TYPE_BY_NAME = {
    # Binding agent
    "%OC": "Binding agent",

    # Earth element (nontoxic)
    "Al": "Earth element (nontoxic)",
    "Ca": "Earth element (nontoxic)",
    "Fe": "Earth element (nontoxic)",
    "K": "Earth element (nontoxic)",
    "Mg": "Earth element (nontoxic)",
    "Na": "Earth element (nontoxic)",

    # Hydrocarbon pollutant
    "1234-TCB": "Hydrocarbon pollutant",
    "1245-TCB": "Hydrocarbon pollutant",
    "HCB": "Hydrocarbon pollutant",
    "OCS": "Hydrocarbon pollutant",
    "QCB": "Hydrocarbon pollutant",

    # organochlorine pesticide
    "Heptachlor Epoxide": "organochlorine pesticide",
    "mirex": "organochlorine pesticide",
    "p,p'-DDD": "organochlorine pesticide",
    "p,p'-DDE": "organochlorine pesticide",

    # Sum of all PCBs
    "total PCB": "Sum of all PCBs",

    # Trace Metal (pollutant)
    "As": "Trace Metal (pollutant)",
    "Bi": "Trace Metal (pollutant)",
    "Cd": "Trace Metal (pollutant)",
    "Co": "Trace Metal (pollutant)",
    "Cr": "Trace Metal (pollutant)",
    "Cu": "Trace Metal (pollutant)",
    "Hg": "Trace Metal (pollutant)",
    "Mn": "Trace Metal (pollutant)",
    "Ni": "Trace Metal (pollutant)",
    "Pb": "Trace Metal (pollutant)",
    "Sb": "Trace Metal (pollutant)",
    "V": "Trace Metal (pollutant)",
    "Zn": "Trace Metal (pollutant)",
}

# Weights by variable type
std_weight = 1.0
TYPE_WEIGHTS = {
    "Trace Metal (pollutant)": std_weight,      # High priority: toxic metals
    "Hydrocarbon pollutant": std_weight,        # Highest priority: petroleum/chlorobenzenes
    "organochlorine pesticide": std_weight,     # Highest priority: POPs
    "Sum of all PCBs": std_weight,              # Highest priority: PCBs
    "Binding agent": std_weight,               # Medium priority: affects bioavailability
    "Earth element (nontoxic)": std_weight   # Lowest priority: background elements
}

# Optional per-variable override map (can be configured at runtime)
_WEIGHTS_BY_NAME_OVERRIDE: dict | None = None


def configure_type_weights(new_type_weights: dict, *, replace: bool = False) -> None:
    """Configure TYPE_WEIGHTS at runtime.

    - replace=False (default): updates existing TYPE_WEIGHTS with provided keys.
    - replace=True: replaces TYPE_WEIGHTS entirely with the provided mapping.
    """
    global TYPE_WEIGHTS
    if replace:
        TYPE_WEIGHTS = dict(new_type_weights)
    else:
        TYPE_WEIGHTS.update(dict(new_type_weights))


def configure_variable_type_map(new_map: dict, *, replace: bool = False) -> None:
    """Configure VARIABLE_TYPE_BY_NAME at runtime.

    - replace=False (default): updates existing VARIABLE_TYPE_BY_NAME entries.
    - replace=True: replaces the mapping entirely with the provided mapping.
    """
    global VARIABLE_TYPE_BY_NAME
    if replace:
        VARIABLE_TYPE_BY_NAME = dict(new_map)
    else:
        VARIABLE_TYPE_BY_NAME.update(dict(new_map))


def configure_weights_by_name(weights_by_name: dict | None) -> None:
    """Set or clear a per-variable weights override used by default.

    Pass a dict like {"Cd": 2.0, "Zn": 1.5}. Pass None to clear.
    """
    global _WEIGHTS_BY_NAME_OVERRIDE
    _WEIGHTS_BY_NAME_OVERRIDE = None if weights_by_name is None else dict(weights_by_name)


def get_variable_weight(
    name: str,
    *,
    type_weights: dict | None = None,
    variable_type_by_name: dict | None = None,
    weights_by_name: dict | None = None,
    default: float = 1.0,
) -> float:
    """Return weight for a variable name with optional runtime overrides.

    Precedence (highest to lowest):
    1) weights_by_name[name] if provided (or configured via configure_weights_by_name)
    2) TYPE lookup via variable_type_by_name (or global VARIABLE_TYPE_BY_NAME) mapped
       through type_weights (or global TYPE_WEIGHTS)
    3) default
    """
    # Direct per-variable override first
    # Explicit arg overrides module-level configured overrides
    if weights_by_name is not None and name in weights_by_name:
        return float(weights_by_name[name])
    if _WEIGHTS_BY_NAME_OVERRIDE is not None and name in _WEIGHTS_BY_NAME_OVERRIDE:
        return float(_WEIGHTS_BY_NAME_OVERRIDE[name])

    # Resolve type mapping
    vt_map = variable_type_by_name if variable_type_by_name is not None else VARIABLE_TYPE_BY_NAME
    vtype = vt_map.get(name)
    if vtype is None:
        return float(default)

    # Resolve type weight
    tw_map = type_weights if type_weights is not None else TYPE_WEIGHTS
    return float(tw_map.get(vtype, default))


def build_weights_for_columns(
    columns,
    *,
    type_weights: dict | None = None,
    variable_type_by_name: dict | None = None,
    weights_by_name: dict | None = None,
    default: float = 1.0,
) -> dict:
    """Build a per-variable weight dict for a sequence of names with overrides.

    Parameters (all optional):
    - type_weights: custom mapping {type -> weight}
    - variable_type_by_name: custom mapping {variable -> type}
    - weights_by_name: direct overrides {variable -> weight}
    - default: fallback weight when a variable/type is unknown (default 1.0)

    Usage examples (in a notebook):
    - build_weights_for_columns(cols, weights_by_name={"Cd": 2.0})
    - build_weights_for_columns(cols, type_weights={"Trace Metal (pollutant)": 2.0})
    - configure_type_weights({"Trace Metal (pollutant)": 2.0}); build_weights_for_columns(cols)
    - configure_weights_by_name({"Cd": 2.0}); build_weights_for_columns(cols)
    """
    return {
        name: get_variable_weight(
            name,
            type_weights=type_weights,
            variable_type_by_name=variable_type_by_name,
            weights_by_name=weights_by_name,
            default=default,
        )
        for name in columns
    }
