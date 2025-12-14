"""
Parameter grid expansion utilities for automation.

Handles expansion of parameter combinations for grid search,
folder name generation, and tracking of parameter variations.
"""

# Category prefixes for folder naming (ensures no overlap between categories)
CATEGORY_PREFIXES = {
    'general': 'g',
    'cofiring': 'cf',
    'advanced': 'av',
    'event_based': 'eb',
}

# Parameter abbreviations for folder naming
# Format: {category: {param_name: abbreviation}}
PARAMETER_ABBREVIATIONS = {
    'general': {
        'which_cells': 'cells',
    },
    'cofiring': {
        'window_size': 'ws',
        'distance_threshold': 'dt',
        'verified_only': 'vo',
        'spatial': 'sp',
        'temporal': 'tm',
        'num_shuffles': 'ns',
    },
    'advanced': {
        'window_size': 'ws',
        'readout': 'rd',
        'fpr': 'fpr',
        'scaling': 'scl',
        'spatial': 'sp',
        'temporal': 'tm',
        'num_shuffles': 'ns',
    },
    'event_based': {
        'event_type': 'evt',
        'window_size': 'ws',
        'lag': 'lag',
        'num_subwindows': 'nsw',
        'num_shuffles': 'ns',
        'amplitude_anchored': 'aa',
        'shuffle_type': 'st',
        # Spatial analysis sub-parameters
        'nnr_significance_threshold': 'nnr_sig',
        'high_cell_selection_method': 'hc_meth',
        'high_cell_z_threshold': 'hc_z',
        'high_cell_range_start': 'hc_start',
        'high_cell_range_end': 'hc_end',
    }
}


def format_param_value(value) -> str:
    """
    Convert parameter value to folder-safe string representation.
    
    Parameters
    ----------
    value : any
        Parameter value to format
        
    Returns
    -------
    formatted : str
        Folder-safe string representation of the value
        
    Examples
    --------
    >>> format_param_value(True)
    't'
    >>> format_param_value(False)
    'f'
    >>> format_param_value(1000)
    '1000'
    >>> format_param_value(0.05)
    '0.05'
    >>> format_param_value("All Cells")
    'all'
    >>> format_param_value(None)
    'none'
    """
    if value is None:
        return 'none'
    elif isinstance(value, bool):
        return 't' if value else 'f'
    elif isinstance(value, (int, float)):
        # Convert number to string, handle floats nicely
        if isinstance(value, float):
            # Remove trailing zeros and decimal point if not needed
            formatted = f"{value:.10f}".rstrip('0').rstrip('.')
            return formatted
        return str(value)
    elif isinstance(value, str):
        # Handle common string values with abbreviations
        value_lower = value.lower()
        
        # Common abbreviations
        abbreviations = {
            'all cells': 'all',
            'verified cells': 'verified',
            'spatial': 'spatial',
            'temporal': 'temporal',
            'bidirectional': 'bidir',
            'zscore': 'zscore',
            'percentile': 'pct',
        }
        
        if value_lower in abbreviations:
            return abbreviations[value_lower]
        
        # For other strings, replace spaces with underscores and keep alphanumeric
        safe_value = value.replace(' ', '_')
        # Remove any characters that aren't alphanumeric, underscore, or hyphen
        safe_value = ''.join(c for c in safe_value if c.isalnum() or c in ('_', '-'))
        return safe_value.lower()
    else:
        # Fallback: convert to string and sanitize
        return str(value).replace(' ', '_').lower()


def generate_combination_folder_name(params_dict: dict, varying_params: set = None) -> str:
    """
    Generate descriptive folder name from parameter combination.
    
    Only includes parameters that vary across combinations (if varying_params is provided).
    Orders parameters by category and then alphabetically within category for consistency.
    
    Parameters
    ----------
    params_dict : dict
        Parameter dictionary with structure {category: {param_name: value}}
    varying_params : set, optional
        Set of (category, param_name) tuples indicating which parameters vary.
        If None, all parameters are included in the folder name.
        
    Returns
    -------
    folder_name : str
        Folder-safe name like 'cf_ws_1000_cf_vo_f_av_ws_500'
        
    Notes
    -----
    - Category prefixes prevent ambiguity (cf_ws vs av_ws)
    - Parameters are ordered by category then alphabetically
    - Windows path limit is 260 chars; folder names are kept concise
    """
    name_parts = []
    
    # Process categories in consistent order
    category_order = ['general', 'cofiring', 'advanced', 'event_based']
    
    for category in category_order:
        if category not in params_dict:
            continue
            
        category_params = params_dict[category]
        if category_params is None:
            continue
            
        # Get category prefix
        prefix = CATEGORY_PREFIXES.get(category, category[:2])
        
        # Sort parameter names alphabetically for consistency
        param_names = sorted(category_params.keys())
        
        for param_name in param_names:
            # Skip if not in varying params (when varying_params is specified)
            if varying_params is not None and (category, param_name) not in varying_params:
                continue
            
            value = category_params[param_name]
            
            # Skip None values and nested dicts (like spatial_analysis)
            if value is None or isinstance(value, dict):
                continue
            
            # Get abbreviation for this parameter
            abbrev = PARAMETER_ABBREVIATIONS.get(category, {}).get(param_name, param_name)
            
            # Format the value
            formatted_value = format_param_value(value)
            
            # Build name part: prefix_abbrev_value
            name_part = f"{prefix}_{abbrev}_{formatted_value}"
            name_parts.append(name_part)
    
    # Join all parts with underscores
    folder_name = '_'.join(name_parts)
    
    # Handle edge case of empty name
    if not folder_name:
        folder_name = 'default'
    
    return folder_name


def expand_parameters(params: dict) -> tuple[list[dict], set]:
    """
    Expand parameters into all combinations for grid search.
    
    Parameters with list values are expanded into combinations using Cartesian product.
    Parameters with single values are kept constant across all combinations.
    
    Parameters
    ----------
    params : dict
        Parameter dict with structure {category: {param_name: value_or_list}}
        where values can be single values or lists
        
    Returns
    -------
    combinations : list[dict]
        List of parameter dicts, one per combination
    varying_params : set
        Set of (category, param_name) tuples indicating which parameters vary
        
    Examples
    --------
    >>> params = {
    ...     'cofiring': {'window_size': [1000, 2000], 'num_shuffles': 100},
    ...     'advanced': {'window_size': 500}
    ... }
    >>> combos, varying = expand_parameters(params)
    >>> len(combos)
    2
    >>> varying
    {('cofiring', 'window_size')}
    """
    from itertools import product
    
    # Identify which parameters are lists (to be expanded)
    expandable_params = {}
    fixed_params = {}
    varying_params = set()
    
    for category in params:
        if params[category] is None:
            continue
            
        expandable_params[category] = {}
        fixed_params[category] = {}
        
        for param_name, value in params[category].items():
            # Skip nested dicts (like spatial_analysis)
            if isinstance(value, dict):
                fixed_params[category][param_name] = value
                continue
            
            # Check if it's a list with multiple values
            if isinstance(value, list) and len(value) > 1:
                expandable_params[category][param_name] = value
                varying_params.add((category, param_name))
            else:
                # Single value or single-item list - treat as constant
                if isinstance(value, list) and len(value) == 1:
                    fixed_params[category][param_name] = value[0]
                else:
                    fixed_params[category][param_name] = value
    
    # Generate all combinations using Cartesian product
    combinations = []
    
    # Collect all expandable parameter names and their values
    param_keys = []  # List of (category, param_name) tuples
    param_values = []  # List of value lists
    
    for category in expandable_params:
        for param_name, values in expandable_params[category].items():
            param_keys.append((category, param_name))
            param_values.append(values)
    
    # Create cartesian product
    if param_values:
        for combo_values in product(*param_values):
            # Start with fixed params
            combo = {}
            for category in params:
                if category in fixed_params:
                    combo[category] = dict(fixed_params[category])
                else:
                    combo[category] = {}
            
            # Add this combination's varying values
            for (category, param_name), value in zip(param_keys, combo_values):
                combo[category][param_name] = value
            
            combinations.append(combo)
    else:
        # No expandable parameters, return single combination with all fixed params
        combo = {}
        for category in params:
            if category in fixed_params:
                combo[category] = dict(fixed_params[category])
            else:
                combo[category] = {}
        combinations = [combo]
    
    return combinations, varying_params


def save_combination_manifest(output_path: str, params: dict, combination_idx: int, 
                              varying_params: set = None):
    """
    Save a JSON manifest file documenting the parameters used for this combination.
    
    Creates a 'parameters_used.json' file in the combination folder that records:
    - All parameter values used
    - Which parameters varied across combinations
    - Combination index/number
    
    Parameters
    ----------
    output_path : str
        Path to the combination folder where manifest will be saved
    params : dict
        Parameter dictionary for this specific combination
    combination_idx : int
        Index of this combination (1-based for display)
    varying_params : set, optional
        Set of (category, param_name) tuples indicating which parameters varied
        
    Returns
    -------
    manifest_path : str
        Path to the saved manifest file
    """
    import json
    import os
    
    manifest = {
        'combination_index': combination_idx,
        'parameters': params,
        'varying_parameters': []
    }
    
    # Add list of varying parameters if provided
    if varying_params:
        manifest['varying_parameters'] = [
            {'category': cat, 'parameter': param} 
            for cat, param in sorted(varying_params)
        ]
    
    # Save to JSON file
    manifest_path = os.path.join(output_path, 'parameters_used.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path
