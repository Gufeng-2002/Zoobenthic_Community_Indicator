"""
Velocity Imputation Module for Detroit River Corridor Sites

This module provides functions to impute missing water velocity values for sites
in the St. Clair-Detroit River System (SCDRS), with different imputation strategies
for different waterbodies (Lake St. Clair, St. Clair River, Detroit River).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# FIGURE_SAVE_PATH = "../results/figures/02_Taxa_Assemblages_on_Ref_Sites/"

def _normalize(series):
    """Normalize a series to [0, 1] range."""
    rng = series.max() - series.min()
    return (series - series.min()) / (rng if rng > 0 else 1.0)


def impute_velocity_for_all_sites(
    raw_data,
    multiindex_data,
    velocity_col='Velocity  at bottom (m/sec)',
    waterbody_col='Waterbody',
    lon_col='Longitude',
    lat_col='Latitude',
    random_seed=42
):
    """
    Impute missing water velocity values for sites in SCDRS.
    
    This function implements a sophisticated imputation strategy that considers:
    - Detroit River (DR): Uses recorded velocities (no imputation needed)
    - Lake St. Clair (LSC): Proximity-based imputation with delta zone boost
    - St. Clair River (SCR): Baseline imputation with random scaling
    
    Parameters:
    -----------
    raw_data : pd.DataFrame
        Raw dataframe without multi-index columns
    multiindex_data : pd.DataFrame
        Dataframe with multi-index columns
    velocity_col : str, default='Velocity  at bottom (m/sec)'
        Name of the velocity column to impute
    waterbody_col : str, default='Waterbody'
        Name of the waterbody identifier column
    lon_col : str, default='Longitude'
        Name of the longitude column
    lat_col : str, default='Latitude'
        Name of the latitude column
    random_seed : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'raw_data': Updated raw dataframe with '{velocity_col}_Imputed' column
        - 'multiindex_data': Updated multi-index data with imputed values
        - 'imputation_summary': Dictionary with statistics for each waterbody
        - 'fig_map': Matplotlib figure showing spatial distribution
        - 'fig_boxplot': Matplotlib figure showing velocity distributions by area
    
    Example:
    --------
    >>> results = impute_velocity_for_all_sites(raw_data, multiindex_data)
    >>> updated_raw_data = results['raw_data']
    >>> fig_map = results['fig_map']
    >>> fig_map.show()
    """
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(random_seed)
    
    # Create copies to avoid modifying originals
    raw_data = raw_data.copy()
    multiindex_data = multiindex_data.copy()
    
    # Create GeoDataFrame for spatial operations
    sites_gdf = gpd.GeoDataFrame(
        raw_data,
        geometry=gpd.points_from_xy(raw_data[lon_col], raw_data[lat_col]),
        crs='EPSG:4326'
    )
    
    # Initialize imputed column with original values
    imputed_col_name = f'{velocity_col}_Imputed'
    sites_gdf[imputed_col_name] = sites_gdf[velocity_col].copy()
    
    # =====================================================================
    # Step 1: Get Detroit River baseline (median velocity)
    # =====================================================================
    is_dr = sites_gdf[waterbody_col].str.upper().str.contains('DR')
    dr_sites = sites_gdf[is_dr]
    dr_median = dr_sites[velocity_col].median()
    
    print("="*70)
    print("VELOCITY IMPUTATION FOR SCDRS SITES")
    print("="*70)
    print(f"\nDetroit River baseline: {dr_median:.3f} m/s (median of {is_dr.sum()} sites)")
    
    # =====================================================================
    # Step 2: Impute Lake St. Clair velocities
    # =====================================================================
    lsc_velocity, lsc_sites, delta_zone_mask = _impute_lsc_velocity(
        sites_gdf, dr_median, velocity_col, waterbody_col, lon_col, lat_col, rng
    )
    
    # Update imputed values for LSC
    sites_gdf.loc[lsc_sites.index, imputed_col_name] = lsc_velocity
    
    # =====================================================================
    # Step 3: Impute St. Clair River velocities
    # =====================================================================
    scr_velocity, scr_sites = _impute_scr_velocity(
        sites_gdf, dr_median, velocity_col, waterbody_col, rng
    )
    
    # Update imputed values for SCR
    sites_gdf.loc[scr_sites.index, imputed_col_name] = scr_velocity
    
    # =====================================================================
    # Step 4: Update raw_data and multiindex_data
    # =====================================================================
    raw_data[imputed_col_name] = sites_gdf[imputed_col_name].values
    
    # Add to multi-index data with proper column structure
    multiindex_col = pd.MultiIndex.from_tuples([('habitat', 'imputed', 'Velocity_Imputed')])
    velocity_multiindex_df = pd.DataFrame(
        sites_gdf[imputed_col_name].values,
        index=sites_gdf.index,
        columns=multiindex_col
    )
    
    from zci.data_process.dataframe_ops import concat_blocks
    multiindex_data = concat_blocks([multiindex_data, velocity_multiindex_df])
    
    # =====================================================================
    # Step 5: Create visualizations
    # =====================================================================
    print("\n" + "="*70)
    print("Creating visualizations...")
    
    fig = _create_combined_visualizations(
        sites_gdf, imputed_col_name, waterbody_col, lon_col, lat_col, 
        delta_zone_mask, lsc_sites.index
    )
    
    # =====================================================================
    # Step 6: Compile summary statistics
    # =====================================================================
    summary = {
        'detroit_river': {
            'n_sites': is_dr.sum(),
            'median_velocity': dr_median,
            'mean_velocity': dr_sites[velocity_col].mean(),
            'std_velocity': dr_sites[velocity_col].std()
        },
        'lake_stclair': {
            'n_sites': len(lsc_sites),
            'n_delta_zone': delta_zone_mask.sum(),
            'median_velocity': np.median(lsc_velocity),
            'delta_zone_median': np.median(lsc_velocity[delta_zone_mask]),
            'non_delta_median': np.median(lsc_velocity[~delta_zone_mask])
        },
        'stclair_river': {
            'n_sites': len(scr_sites),
            'median_velocity': np.median(scr_velocity),
            'ratio_to_dr': np.median(scr_velocity) / dr_median
        }
    }
    
    print("\n" + "="*70)
    print("IMPUTATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Total sites: {len(sites_gdf)}")
    print(f"  Sites with imputed values: {len(lsc_sites) + len(scr_sites)}")
    print(f"  Sites with recorded values: {is_dr.sum()}")
    
    return {
        'raw_data': raw_data,
        'multiindex_data': multiindex_data,
        'imputation_summary': summary,
        'fig': fig
    }


def _impute_lsc_velocity(sites_gdf, dr_median, velocity_col, waterbody_col, 
                         lon_col, lat_col, rng):
    """
    Impute velocity for Lake St. Clair sites using proximity-based method.
    
    Method:
    - Calculate distances to lake shoreline, Detroit River, and St. Clair River
    - Combine distances into proximity score [0,1]
    - Impute: proximity × scale × DR_median × randomness
    - Boost velocities in delta zone (near St. Clair River mouth)
    """
    
    print("\n[Lake St. Clair Imputation]")
    
    # Load geometries and set metric CRS
    lake_stclair = gpd.read_file("../data/maps/lake_stclair/lake_stclair.shp").to_crs(epsg=3857)
    detroit_river = gpd.read_file("../data/maps/detroit_river_aoc_shapefile/AOC_MI_Detroit_2021.shp").to_crs(epsg=3857)
    stclair_river = gpd.read_file("../data/maps/aoc_mi_stclair_2021/AOC_MI_StClair_2021.shp").to_crs(epsg=3857)
    
    # Filter LSC sites
    is_lsc = sites_gdf[waterbody_col].str.upper().str.contains('LSC')
    lsc_sites = sites_gdf[is_lsc].copy()
    lsc_sites = lsc_sites.to_crs(epsg=3857)
    
    # Unioned boundaries for distance
    lake_boundary = lake_stclair.boundary.union_all()
    river_union = detroit_river.union_all().buffer(50)
    stclair_union = stclair_river.union_all().buffer(50)
    
    # Calculate distances (meters)
    lsc_sites['dist_shore_m'] = lsc_sites.geometry.apply(lambda g: g.distance(lake_boundary))
    lsc_sites['dist_dr_m'] = lsc_sites.geometry.apply(lambda g: g.distance(river_union))
    lsc_sites['dist_scr_m'] = lsc_sites.geometry.apply(lambda g: g.distance(stclair_union))
    
    # Proximity components (inverse distance with softening)
    epsilon = 1.0
    inv_shore = 1.0 / (lsc_sites['dist_shore_m'] + epsilon)
    inv_dr = 1.0 / (lsc_sites['dist_dr_m'] + epsilon)
    inv_scr = 1.0 / (lsc_sites['dist_scr_m'] + epsilon)
    
    # Normalize to [0,1] and combine with weights
    shore_score = _normalize(inv_shore)
    dr_score = _normalize(inv_dr)
    scr_score = _normalize(inv_scr)
    w_shore, w_dr, w_scr = 0.5, 1, 0.5
    lsc_sites['proximity_score'] = (w_shore * shore_score + w_dr * dr_score + w_scr * scr_score).clip(0, 1)
    
    # Impute: proximity × scale (0.5–1.0) × DR median × lognormal randomness
    scale_factor = rng.uniform(0.5, 1, size=len(lsc_sites))
    randomness = rng.lognormal(mean=0.0, sigma=0.15, size=len(lsc_sites))
    lsc_velocity = lsc_sites['proximity_score'].values * dr_median * scale_factor * randomness
    
    # Delta-Zone boost (near St. Clair River mouth: lon > -82.66, lat > 42.5)
    delta_zone_mask = (lsc_sites[lon_col] > -82.66) & (lsc_sites[lat_col] > 42.5)
    lsc_velocity[delta_zone_mask] = lsc_velocity[delta_zone_mask] * rng.normal(1.5, 1, size=delta_zone_mask.sum())
    
    # Near-center attenuation for very low proximity
    center_mask = lsc_sites['proximity_score'] < 0.1
    lsc_velocity[center_mask] = lsc_velocity[center_mask] * 3
    
    # Print summary
    print(f"  LSC sites: {len(lsc_sites)}")
    print(f"  Delta zone sites: {delta_zone_mask.sum()}")
    print(f"  Velocity range: {lsc_velocity.min():.3f} to {lsc_velocity.max():.3f} m/s")
    print(f"  Median velocity (all): {np.median(lsc_velocity):.3f} m/s")
    print(f"  Median velocity (delta): {np.median(lsc_velocity[delta_zone_mask]):.3f} m/s")
    
    return lsc_velocity, lsc_sites, delta_zone_mask


def _impute_scr_velocity(sites_gdf, dr_median, velocity_col, waterbody_col, rng):
    """
    Impute velocity for St. Clair River sites using baseline method.
    
    Method:
    - Use DR median as baseline
    - Add random scaling (0.5–1.5) and normal noise
    - Ensure non-negative values
    """
    
    print("\n[St. Clair River Imputation]")
    
    # Identify SCR sites
    is_scr = sites_gdf[waterbody_col].str.upper().str.contains('SCR') | \
             sites_gdf[waterbody_col].str.upper().str.contains('ST. CLAIR RIVER')
    scr_sites = sites_gdf[is_scr].copy()
    
    # Randomness: small normal noise scaled by baseline
    noise = rng.normal(loc=0.0, scale=0.5 * dr_median, size=len(scr_sites))
    scale = rng.uniform(0.5, 1.5, size=len(scr_sites))
    scr_velocity = np.clip(scale * dr_median + noise, 0.01, None)
    
    # Print summary
    print(f"  SCR sites: {len(scr_sites)}")
    print(f"  Velocity range: {scr_velocity.min():.3f} to {scr_velocity.max():.3f} m/s")
    print(f"  Median velocity: {np.median(scr_velocity):.3f} m/s")
    print(f"  Ratio to DR median: {np.median(scr_velocity)/dr_median:.2f}x")
    
    return scr_velocity, scr_sites


def _create_combined_visualizations(sites_gdf, imputed_col, waterbody_col, 
                                     lon_col, lat_col, delta_zone_mask, lsc_indices):
    """
    Create a combined figure with spatial map (left) and boxplots (right).
    
    Left panel: Spatial map showing imputed velocity distribution
    Right panel: Boxplots comparing velocity distributions across 4 areas
    """
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.15)
    ax_map = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1])
    
    # =====================================================================
    # LEFT PANEL: Spatial Map
    # =====================================================================
    
    # Load geometries
    lake_stclair = gpd.read_file("../data/maps/lake_stclair/lake_stclair.shp").to_crs(epsg=4326)
    detroit_river = gpd.read_file("../data/maps/detroit_river_aoc_shapefile/AOC_MI_Detroit_2021.shp").to_crs(epsg=4326)
    stclair_river = gpd.read_file("../data/maps/aoc_mi_stclair_2021/AOC_MI_StClair_2021.shp").to_crs(epsg=4326)
    
    # Base map
    lake_stclair.plot(ax=ax_map, color='lightblue', edgecolor='none', alpha=0.5)
    detroit_river.plot(ax=ax_map, color='lightblue', edgecolor='none', alpha=0.5)
    stclair_river.plot(ax=ax_map, color='lightblue', edgecolor='none', alpha=0.5)
    
    # Site points
    norm = Normalize(vmin=sites_gdf[imputed_col].min(), vmax=sites_gdf[imputed_col].max())
    sc = ax_map.scatter(
        sites_gdf[lon_col], 
        sites_gdf[lat_col],
        c=sites_gdf[imputed_col],
        s=sites_gdf[imputed_col] * 300 + 30,
        cmap='plasma',
        norm=norm,
        edgecolors='k',
        linewidths=0.5,
        alpha=0.9
    )
    
    # Colorbar
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Imputed Velocity (m/s)', fontsize=11)
    
    # Labels
    ax_map.set_title('Spatial Distribution of Imputed Water Velocity\nSt. Clair-Detroit River System', 
                     fontsize=13, fontweight='bold', pad=15)
    ax_map.set_xlabel('Longitude', fontsize=11)
    ax_map.set_ylabel('Latitude', fontsize=11)
    ax_map.set_ylim(42, 43.1)
    ax_map.set_xlim(-83.3, -82.3)
    ax_map.grid(linestyle='--', alpha=0.4)
    
    # Annotations
    ax_map.text(-83.0, 42.2, 'Detroit River', fontsize=9, color='gray', style='italic')
    ax_map.text(-82.85, 42.9, 'St. Clair River', fontsize=9, color='gray', style='italic')
    ax_map.text(-82.55, 42.05, 'Lake Erie', fontsize=9, color='gray', style='italic')
    ax_map.text(-83.0, 42.5, 'Lake St. Clair', fontsize=9, color='gray', style='italic')
    
    # =====================================================================
    # RIGHT PANEL: Boxplots
    # =====================================================================
    
    # Prepare data for plotting
    plot_data = []
    
    # Detroit River
    is_dr = sites_gdf[waterbody_col].str.upper().str.contains('DR')
    dr_velocities = sites_gdf[is_dr][imputed_col].dropna()
    plot_data.extend([{'Area': 'Detroit River\n(recorded)', 'Velocity': v} for v in dr_velocities])
    
    # Lake St. Clair - Delta Zone
    lsc_velocities_all = sites_gdf.loc[lsc_indices][imputed_col].values
    lsc_delta_velocities = lsc_velocities_all[delta_zone_mask.values]
    plot_data.extend([{'Area': 'LSC Delta Zone\n(imputed)', 'Velocity': v} for v in lsc_delta_velocities])
    
    # Lake St. Clair - Main Lake
    lsc_main_velocities = lsc_velocities_all[~delta_zone_mask.values]
    plot_data.extend([{'Area': 'LSC Main Lake\n(imputed)', 'Velocity': v} for v in lsc_main_velocities])
    
    # St. Clair River
    is_scr = sites_gdf[waterbody_col].str.upper().str.contains('SCR') | \
             sites_gdf[waterbody_col].str.upper().str.contains('ST. CLAIR RIVER')
    scr_velocities = sites_gdf[is_scr][imputed_col].dropna()
    plot_data.extend([{'Area': 'St. Clair River\n(imputed)', 'Velocity': v} for v in scr_velocities])
    
    # Create DataFrame
    df_plot = pd.DataFrame(plot_data)
    
    # Boxplot with individual points
    import seaborn as sns
    sns.boxplot(data=df_plot, x='Area', y='Velocity', ax=ax_box, palette='Set2', width=0.6)
    sns.stripplot(data=df_plot, x='Area', y='Velocity', ax=ax_box, color='black', 
                  alpha=0.3, size=4, jitter=True)
    
    # Labels
    ax_box.set_title('Water Velocity Distribution by Area\nSCDRS Sites', 
                     fontsize=13, fontweight='bold', pad=15)
    ax_box.set_xlabel('Area', fontsize=11)
    ax_box.set_ylabel('Velocity (m/s)', fontsize=11)
    ax_box.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add median values as text
    for i, area in enumerate(df_plot['Area'].unique()):
        median_val = df_plot[df_plot['Area'] == area]['Velocity'].median()
        ax_box.text(i, median_val, f'{median_val:.3f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    # plt.savefig(os.path.join(FIGURE_SAVE_PATH, "Imputed_Velocity_over_SCDRS.png"), dpi=300)
    
    return fig
