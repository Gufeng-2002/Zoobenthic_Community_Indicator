"""
Simple usage example for the PCA contamination assessment module.

This script shows the basic workflow for using the updated pca_assessment.py module
with the hierarchical master data structure.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the src directory to path
sys.path.append('../src')
from ecoindex.pca_assessment import pca_chemical_assessment
from ecoindex.dataframe_ops import get_block

def simple_usage_example():
    """Simple example showing basic usage of PCA contamination assessment."""
    
    print("=== PCA Contamination Assessment - Usage Example ===\n")
    
    # Load master data (assuming it's already in the correct format)
    data_path = "../data/processed/master_example.csv"
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the master_example.csv file exists in the data/processed directory.")
        return
    
    # Load and prepare data (simplified version)
    print("1. Loading master data...")
    try:
        # This assumes you have a function to load your master data
        # In practice, you would load your actual master DataFrame here
        print("   Note: In your actual code, load your master DataFrame with hierarchical columns")
        print("   Expected structure: MultiIndex columns with levels (block, subblock, var)")
        print("   Example: df.columns = pd.MultiIndex.from_arrays([blocks, subblocks, vars])")
        
        # For demonstration, create a mock master DataFrame structure
        master_df = create_mock_master_data()
        print(f"   Loaded: {master_df.shape[0]} sites × {master_df.shape[1]} variables")
        
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    print("\n2. Performing PCA contamination assessment...")
    try:
        # Basic PCA analysis on raw chemical data
        result = pca_chemical_assessment(
            master_df,
            chemical_block="chemical",    # Name of chemical block
            subblock="raw",              # Use raw chemical data
            n_components=5,              # Keep first 5 PCs
            standardize=True             # Standardize variables
        )
        
        print(f"   ✓ Analysis complete: {result.n_sites} sites, {result.n_chemicals} chemicals")
        print(f"   ✓ PC1 explains {result.explained_variance_ratio[0]:.1%} of variance")
        
    except Exception as e:
        print(f"   Error in PCA analysis: {e}")
        return
    
    print("\n3. Contamination ranking results:")
    
    # Most contaminated sites
    print("   Top 5 most contaminated sites:")
    most_contaminated = result.get_most_contaminated_sites(5)
    for i, (site, data) in enumerate(most_contaminated.iterrows(), 1):
        print(f"      {i}. {site}: score = {data['contamination_score']:.1f}")
    
    # Least contaminated sites
    print("\n   Top 5 least contaminated sites:")
    least_contaminated = result.get_least_contaminated_sites(5)
    for i, (site, data) in enumerate(least_contaminated.iterrows(), 1):
        print(f"      {i}. {site}: score = {data['contamination_score']:.1f}")
    
    print("\n4. Key contaminants driving PC1:")
    key_contaminants = result.get_key_contaminants(pc=1, n=5)
    for i, (chemical, data) in enumerate(key_contaminants.iterrows(), 1):
        print(f"   {i}. {chemical}: loading = {data['loading']:.3f}")
    
    print("\n5. Additional analysis options:")
    print("   - Access PC scores: result.scores")
    print("   - Access loadings: result.loadings") 
    print("   - Access explained variance: result.explained_variance_ratio")
    print("   - Get contamination scores: result.contamination_scores")
    print("   - Analyze different subblocks: subblock='hellinger' or 'logz'")
    
    print("\n=== Example completed successfully! ===")


def create_mock_master_data():
    """Create a mock master DataFrame for demonstration purposes."""
    
    # Create sample data structure
    np.random.seed(42)
    n_sites = 20
    
    # Site IDs
    sites = [f"Site_{i:02d}" for i in range(1, n_sites + 1)]
    
    # Chemical variables
    chemicals = ['Al', 'As', 'Cd', 'Cr', 'Cu', 'Fe', 'Hg', 'Ni', 'Pb', 'Zn']
    
    # Environmental variables  
    env_vars = ['pH', 'Temperature', 'Depth', 'Salinity']
    
    # Create hierarchical columns
    chemical_cols = pd.MultiIndex.from_product(
        [['chemical'], ['raw'], chemicals],
        names=['block', 'subblock', 'var']
    )
    
    env_cols = pd.MultiIndex.from_product(
        [['environmental'], ['raw'], env_vars],
        names=['block', 'subblock', 'var']
    )
    
    # Combine columns
    all_cols = chemical_cols.append(env_cols)
    
    # Generate sample data
    n_vars = len(all_cols)
    data = np.random.lognormal(mean=1, sigma=1, size=(n_sites, n_vars))
    
    # Add some correlation structure to make it more realistic
    contamination_gradient = np.random.rand(n_sites)
    for i, col in enumerate(chemical_cols):
        data[:, i] *= (1 + 2 * contamination_gradient)  # More contaminated sites have higher values
    
    # Create DataFrame
    df = pd.DataFrame(data, index=sites, columns=all_cols)
    
    return df


if __name__ == "__main__":
    simple_usage_example()
