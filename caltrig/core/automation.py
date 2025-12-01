"""
Automation logic for batch processing analyses across multiple sessions.
"""

import json
import os
import pandas as pd
import numpy as np
from .backend import DataInstance
from .shuffling import shuffle_cofiring, shuffle_advanced
from .exploration_statistics import generate_general_statistics, generate_local_statistics
from .event_based_shuffling import (event_based_shuffle_analysis, 
                                    generate_event_based_statistics_for_export,
                                    generate_spatial_statistics_for_export)


def load_parameters(parameter_file: str) -> dict:
    """
    Load and parse parameters from JSON file.
    
    Parameters
    ----------
    parameter_file : str
        Path to JSON file containing analysis parameters
        
    Returns
    -------
    params : dict
        Dictionary containing parsed parameters organized by analysis type
    """
    with open(parameter_file, 'r') as f:
        data = json.load(f)
    
    params = {
        'general': {},
        'cofiring': {},
        'advanced': {},
        'event_based': {}
    }
    
    # General settings
    general = data.get("General", {})
    params['general']['which_cells'] = general.get("which_cells", "All Cells")
    
    # 3D Visualization (Co-firing)
    cofiring_data = data.get("3D Visualization", {}).get("Co-Firing", {})
    params['cofiring']['window_size'] = cofiring_data.get("window_size")
    params['cofiring']['distance_threshold'] = cofiring_data.get("distance_threshold")
    
    # Shuffling parameters for co-firing
    shuf = cofiring_data.get("shuffling", {})
    params['cofiring']['verified_only'] = shuf.get("verified_only", False)
    params['cofiring']['spatial'] = shuf.get("spatial", False)
    params['cofiring']['temporal'] = shuf.get("temporal", False)
    params['cofiring']['num_shuffles'] = shuf.get("num_shuffles")
    
    # Advanced Visualization (FPR)
    fpr_data = data.get("Advanced Visualization", {}).get("FPR visualization", {})
    params['advanced']['window_size'] = fpr_data.get("window_size")
    params['advanced']['readout'] = fpr_data.get("readout")
    params['advanced']['fpr'] = fpr_data.get("fpr")
    params['advanced']['scaling'] = fpr_data.get("scaling")
    
    # Shuffling parameters for advanced
    shuf_adv = fpr_data.get("shuffling", {})
    params['advanced']['spatial'] = shuf_adv.get("spatial", False)
    params['advanced']['temporal'] = shuf_adv.get("temporal", False)
    params['advanced']['num_shuffles'] = shuf_adv.get("num_shuffles")
    
    # Event-based Shuffling
    event_data = data.get("Event based Shuffling", {})
    params['event_based']['event_type'] = event_data.get("event_type")
    params['event_based']['window_size'] = event_data.get("window_size")
    params['event_based']['lag'] = event_data.get("lag")
    params['event_based']['num_subwindows'] = event_data.get("num_subwindows")
    params['event_based']['num_shuffles'] = event_data.get("num_shuffles")
    params['event_based']['amplitude_anchored'] = event_data.get("amplitude_anchored", False)
    params['event_based']['shuffle_type'] = event_data.get("shuffle_type", "spatial")
    
    # Spatial analysis parameters (optional)
    spatial_analysis = event_data.get("spatial_analysis", {})
    if spatial_analysis:
        params['event_based']['spatial_analysis'] = {
            'nnr_significance_threshold': spatial_analysis.get('nnr_significance_threshold', 0.05),
            'high_cell_selection_method': spatial_analysis.get('high_cell_selection_method', 'zscore'),
            'high_cell_z_threshold': spatial_analysis.get('high_cell_z_threshold', 1.5),
            'high_cell_range_start': spatial_analysis.get('high_cell_range_start', 0.1),
            'high_cell_range_end': spatial_analysis.get('high_cell_range_end', 0.9)
        }
    else:
        params['event_based']['spatial_analysis'] = None
    
    return params


def run_batch_automation(session_paths: list, parameter_file: str, output_path: str = None,
                        enabled_outputs: dict = None,
                        progress_callback_session=None, progress_callback_analysis=None, 
                        progress_callback_analysis_done=None):
    """
    Run automated analysis across multiple sessions.
    
    Parameters
    ----------
    session_paths : list
        List of paths to .ini configuration files for each session
    parameter_file : str
        Path to JSON file containing analysis parameters
    output_path : str, optional
        Directory to save output files. If None, use current working directory
    enabled_outputs : dict, optional
        Dictionary specifying which outputs to generate: {'cofiring': bool, 'advanced': bool, 'event_based': bool}
        If None, all outputs are enabled
    progress_callback_session : callable, optional
        Callback function(current, total, session_name) to update session progress
    progress_callback_analysis : callable, optional
        Callback function(analysis_type) to update current analysis type
    progress_callback_analysis_done : callable, optional
        Callback function() to signal analysis completion
        
    Returns
    -------
    results : dict
        Dictionary containing results and any errors encountered
    """
    # Load parameters
    params = load_parameters(parameter_file)
    
    # Use current directory if no output path specified
    if output_path is None:
        output_path = os.getcwd()
    
    # Default to all outputs enabled if not specified
    if enabled_outputs is None:
        enabled_outputs = {
            'cofiring': True,
            'advanced': True,
            'event_based': True,
            'general_stats': True,
            'local_stats': True
        }
    
    results = {
        'successful': [],
        'failed': []
    }
    
    total_sessions = len(session_paths)
    
    # Loop through all sessions
    for session_idx, session_path in enumerate(session_paths, 1):
        try:
            session = DataInstance(session_path)
            session_name = f"{session.mouseID}_{session.day}_{session.session}"
            
            # Update session progress
            if progress_callback_session:
                progress_callback_session(session_idx, total_sessions, session_name)
            
            # Create session subfolder: mouseID_day_session
            session_output_path = os.path.join(output_path, session_name)
            os.makedirs(session_output_path, exist_ok=True)
            
            # Get cell IDs based on which_cells setting
            which_cells = params['general']['which_cells']
            unit_ids = session.get_cell_ids(which_cells)
            
            # Convert to list for shuffling functions
            target_cells = list(unit_ids)
            comparison_cells = list(unit_ids)
            
            # Apply verified_only filter if specified
            if params['cofiring'].get('verified_only', False):
                target_cells = session.prune_non_verified(target_cells)
                comparison_cells = session.prune_non_verified(comparison_cells)
            
            # Run co-firing analysis if enabled and parameters are present
            if enabled_outputs['cofiring'] and params['cofiring']['num_shuffles'] is not None:
                cofiring_csv_path = os.path.join(session_output_path, "cofiring_results.xlsx")
                cofiring_matrix_path = os.path.join(session_output_path, "cofiring_results_matrix.xlsx")
                
                # Skip if output file already exists
                if not os.path.exists(cofiring_csv_path):
                    if progress_callback_analysis:
                        progress_callback_analysis("Co-firing")
                    
                    # Build parameters for shuffle_cofiring
                    cofiring_params = {
                        'temporal': params['cofiring']['temporal'],
                        'spatial': params['cofiring']['spatial'],
                        'cofiring': {
                            'window_size': params['cofiring']['window_size'],
                            'share_a': False,  # Default values
                            'share_b': False,
                            'direction': 'Both'
                        }
                    }
                    
                    cofiring_result = shuffle_cofiring(
                        session=session,
                        target_cells=target_cells,
                        comparison_cells=comparison_cells,
                        n=params['cofiring']['num_shuffles'],
                        **cofiring_params
                    )
                    
                    # Save co-firing results - both standard and matrix formats
                    cofiring_result.save_to_csv(cofiring_csv_path, use_alt=False)
                    cofiring_result.save_to_csv(cofiring_matrix_path, use_alt=True)
                    
                    if progress_callback_analysis_done:
                        progress_callback_analysis_done()
                else:
                    if progress_callback_analysis:
                        progress_callback_analysis("Co-firing (skipped - file exists)")
                    if progress_callback_analysis_done:
                        progress_callback_analysis_done()
            
            # Run advanced/FPR analysis if enabled and parameters are present
            if enabled_outputs['advanced'] and params['advanced']['num_shuffles'] is not None:
                # Check if at least one shuffling type is enabled
                if not params['advanced']['temporal'] and not params['advanced']['spatial']:
                    if progress_callback_analysis:
                        progress_callback_analysis("Advanced (skipped - no shuffling enabled)")
                    if progress_callback_analysis_done:
                        progress_callback_analysis_done()
                else:
                    advanced_csv_path = os.path.join(session_output_path, "advanced_results.xlsx")
                    
                    # Skip if output file already exists
                    if not os.path.exists(advanced_csv_path):
                        if progress_callback_analysis:
                            progress_callback_analysis("Advanced")
                        
                        # Build parameters for shuffle_advanced
                        advanced_params = {
                            'temporal': params['advanced']['temporal'],
                            'spatial': params['advanced']['spatial'],
                            'shuffling': {
                                'window_size': params['advanced']['window_size'],
                                'readout': params['advanced']['readout'],
                                'fpr': params['advanced']['fpr']
                            },
                            'anchor': params['advanced'].get('scaling', 'None')  # Anchor parameter
                        }
                        
                        advanced_result = shuffle_advanced(
                            session=session,
                            target_cells=target_cells,
                            comparison_cells=comparison_cells,
                            n=params['advanced']['num_shuffles'],
                            **advanced_params
                        )
                        
                        # Save advanced results if analysis succeeded
                        if advanced_result is not None:
                            advanced_result.save_to_excel(advanced_csv_path)
                        
                        if progress_callback_analysis_done:
                            progress_callback_analysis_done()
                    else:
                        if progress_callback_analysis:
                            progress_callback_analysis("Advanced (skipped - file exists)")
                        if progress_callback_analysis_done:
                            progress_callback_analysis_done()
            
            # Event-based analysis (temporal OR spatial based on shuffle_type parameter)
            if enabled_outputs.get('event_based', False) and params['event_based']['num_shuffles'] is not None:
                # Check if event type exists first
                event_type = params['event_based']['event_type']
                shuffle_type = params['event_based']['shuffle_type']
                
                if event_type and event_type in session.data and session.data[event_type] is not None:
                    # Extract event indices
                    events = np.argwhere(session.data[event_type].values == 1)
                    if events.size > 0:
                        events = np.unique(events[events[:, 0] > 50], axis=0) - params['event_based']['lag']
                        
                        if events.size > 0:
                            # Determine output filename based on shuffle type
                            if shuffle_type == "spatial":
                                event_based_path = os.path.join(session_output_path, "event_based_spatial_results.xlsx")
                                analysis_name = "Event-based (Spatial)"
                            else:  # temporal
                                event_based_path = os.path.join(session_output_path, "event_based_temporal_results.xlsx")
                                analysis_name = "Event-based (Temporal)"
                            
                            if not os.path.exists(event_based_path):
                                if progress_callback_analysis:
                                    progress_callback_analysis(analysis_name)
                                
                                try:
                                    result = event_based_shuffle_analysis(
                                        session=session,
                                        selected_cells=target_cells,
                                        event_type=event_type,
                                        window_size=params['event_based']['window_size'],
                                        lag=params['event_based']['lag'],
                                        num_subwindows=params['event_based']['num_subwindows'],
                                        num_shuffles=params['event_based']['num_shuffles'],
                                        amplitude_anchored=params['event_based']['amplitude_anchored'],
                                        shuffle_type=shuffle_type
                                    )
                                    
                                    if result is not None:
                                        # Generate appropriate statistics based on shuffle type
                                        if shuffle_type == "spatial":
                                            # Spatial: generate NNR clustering statistics
                                            spatial_params = params['event_based'].get('spatial_analysis', {})
                                            significance_threshold = spatial_params.get('nnr_significance_threshold', 0.05) if spatial_params else 0.05
                                            stats_df = generate_spatial_statistics_for_export(
                                                result.parameters['statistics'],
                                                significance_threshold=significance_threshold
                                            )
                                        else:
                                            # Temporal: generate per-cell/event statistics
                                            stats_df = generate_event_based_statistics_for_export(
                                                result.original_data,
                                                result.shuffled_data,
                                                events.tolist(),
                                                target_cells
                                            )
                                        
                                        stats_df.to_excel(event_based_path, index=False, engine='xlsxwriter')
                                        result.close()
                                except Exception as e:
                                    print(f"Error in {shuffle_type} event-based analysis: {e}")
                                
                                if progress_callback_analysis_done:
                                    progress_callback_analysis_done()
                            else:
                                if progress_callback_analysis:
                                    progress_callback_analysis(f"{analysis_name} (skipped - file exists)")
                                if progress_callback_analysis_done:
                                    progress_callback_analysis_done()
            
            # General statistics
            if enabled_outputs.get('general_stats', False):
                general_stats_path = os.path.join(session_output_path, "general_statistics.xlsx")
                
                # Skip if output file already exists
                if not os.path.exists(general_stats_path):
                    if progress_callback_analysis:
                        progress_callback_analysis("General Statistics")
                    
                    # Generate general statistics using the same cell selection
                    general_stats_df = generate_general_statistics(session, target_cells)
                    general_stats_df.to_excel(general_stats_path, index=True, engine='xlsxwriter')
                    
                    if progress_callback_analysis_done:
                        progress_callback_analysis_done()
                else:
                    if progress_callback_analysis:
                        progress_callback_analysis("General Statistics (skipped - file exists)")
                    if progress_callback_analysis_done:
                        progress_callback_analysis_done()
            
            # Local statistics (one sheet per cell)
            if enabled_outputs.get('local_stats', False):
                local_stats_path = os.path.join(session_output_path, "local_statistics.xlsx")
                
                # Skip if output file already exists
                if not os.path.exists(local_stats_path):
                    if progress_callback_analysis:
                        progress_callback_analysis("Local Statistics")
                    
                    # Generate local statistics for each cell and save to multi-sheet Excel
                    with pd.ExcelWriter(local_stats_path, engine='xlsxwriter') as writer:
                        for unit_id in target_cells:
                            local_stats_df = generate_local_statistics(session, unit_id)
                            # Use sheet name "Cell X" (Excel has 31 char limit)
                            sheet_name = f"Cell {unit_id}"
                            local_stats_df.to_excel(writer, sheet_name=sheet_name, index=True)
                    
                    if progress_callback_analysis_done:
                        progress_callback_analysis_done()
                else:
                    if progress_callback_analysis:
                        progress_callback_analysis("Local Statistics (skipped - file exists)")
                    if progress_callback_analysis_done:
                        progress_callback_analysis_done()
            
            results['successful'].append(session_path)
            
        except Exception as e:
            results['failed'].append({
                'path': session_path,
                'error': str(e)
            })
    
    return results
