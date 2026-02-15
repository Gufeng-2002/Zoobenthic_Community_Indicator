# ZCI Package Refactoring Plan

## Executive Summary

**Current State**: The `zci` package has grown organically with 7 main submodules, 38+ Python files, and mixed concerns between analysis logic, data processing, visualization, and I/O operations.

**Goal**: Restructure into a clean, maintainable architecture following software engineering best practices while preserving all existing functionality and relationships with the `results/` folder.

**Estimated Time**: 3-4 weeks (60-80 hours)

**Skills Required**: 
- Python software architecture & design patterns
- Data pipeline design
- Scientific computing (pandas, numpy, scikit-learn)
- Testing frameworks (pytest)
- Documentation (Sphinx/MkDocs)

---

## 1. Current Architecture Analysis

### 1.1 Package Structure Overview

```
src/zci/
├── __init__.py                           # Main package entry
├── output_saver.py                       # Cross-cutting I/O utility (615 lines)
│
├── data_process/                         # Data transformation utilities
│   ├── transform.py                      # Hellinger, log transforms
│   └── dataframe_ops.py                  # Multi-index DataFrame utilities
│
├── contamination_assessment/             # Stage 1: Pollution scoring
│   ├── pipeline.py                       # Main orchestration
│   ├── pca_analysis.py                   # PCA on pollution variables
│   ├── pca_pollution_scores.py           # Legacy scoring
│   ├── scoring.py                        # Pollution score computation
│   ├── transformations.py                # Log + z-score
│   ├── validation.py                     # RDA validation
│   └── visualizations.py                 # Ridge plots
│
├── taxa_assemblage_in_refs/              # Stage 2: Reference site analysis
│   ├── pipeline.py                       # Main orchestration (568 lines)
│   ├── reference_site_selection.py       # Site selection logic
│   ├── hierarchical_clustering.py        # Dendrogram analysis
│   ├── cluster_visualization.py          # Map plots
│   ├── boxcox_anova.py                   # Statistical tests
│   └── velocity_imputation.py            # Missing data handling
│
├── env_driven_taxa_clusters/             # Stage 3: Environmental analysis
│   ├── pipeline.py                       # RDA + LDA orchestration
│   ├── rda.py                            # RDA class implementation
│   ├── rda_pipeline.py                   # RDA workflow
│   ├── lda_pipeline.py                   # LDA classification
│   └── (tables utilities)                # Excel/LaTeX table generation
│
├── community_composition_measures/        # Stage 4: Community metrics
│   ├── pipeline.py                       # Main orchestration (1293 lines!)
│   ├── pca_analysis.py                   # PCA on taxa
│   ├── zci_calculation.py                # ZCI metric computation
│   ├── taxa_loadings.py                  # Loading visualizations
│   ├── pc_diagnostics.py                 # PC regression analysis
│   └── visualization.py                  # Ordination plots
│
├── piecewise_quantile_regression/        # Stage 5: PQRM analysis
│   ├── pipeline.py                       # Orchestration
│   ├── pqrm_model.py                     # Model class
│   ├── bootstrap.py                      # Statistical inference
│   ├── visualizations.py                 # Diagnostic plots
│   ├── README.md & QUICK_REFERENCE.md    # Documentation
│
└── map_plots/                            # Geographic visualization
    └── SCDRC_map.py                      # Study area mapping
```

### 1.2 Key Issues Identified

#### **Architecture Issues**
1. **Pipeline modules are too large** (500-1300 lines)
2. **Mixed concerns**: I/O, computation, and visualization in same modules
3. **Inconsistent patterns**: Each stage uses different organization
4. **Tight coupling**: Hard-coded paths, cross-dependencies
5. **Code duplication**: Hellinger transform in 2 places, similar PCA code

#### **Design Issues**
1. **No clear separation** between:
   - Data models (what)
   - Business logic (how)
   - Orchestration (when/where)
   - Presentation (visualization)
   
2. **Lack of abstractions**: 
   - No base classes for pipelines
   - No common interfaces for stages
   - Each stage reinvents the wheel

3. **Testing difficulties**:
   - Monolithic functions hard to unit test
   - Side effects (file I/O) mixed with logic
   - No dependency injection

#### **Maintainability Issues**
1. **Hard to extend**: Adding new analysis requires touching multiple files
2. **Hard to debug**: Long pipelines with many steps
3. **Hard to document**: Functions do too many things
4. **Version control issues**: Large files = more merge conflicts

---

## 2. Proposed New Architecture

### 2.1 Architectural Principles

1. **Separation of Concerns**: Separate data, logic, I/O, visualization
2. **Single Responsibility**: Each module/class does one thing well
3. **Dependency Injection**: Pass dependencies explicitly
4. **Interface-based Design**: Common protocols for similar components
5. **Composition over Inheritance**: Build complex behaviors from simple pieces
6. **Explicit over Implicit**: No hidden state, clear data flow

### 2.2 New Package Structure

```
src/zci/
├── __init__.py
├── config.py                             # Configuration management
│
├── core/                                 # Core abstractions
│   ├── __init__.py
│   ├── pipeline.py                       # BasePipeline, PipelineStage
│   ├── data_models.py                    # Data containers (dataclasses)
│   ├── transforms.py                     # Transform protocol/interface
│   └── exceptions.py                     # Custom exceptions
│
├── io/                                   # Input/Output operations
│   ├── __init__.py
│   ├── readers.py                        # Data loading
│   ├── writers.py                        # Figure/table saving
│   ├── paths.py                          # Path management
│   └── formats.py                        # Excel/LaTeX conversion
│
├── preprocessing/                        # Data preparation
│   ├── __init__.py
│   ├── transforms.py                     # Hellinger, log, etc.
│   ├── imputation.py                     # Missing data
│   ├── multiindex_ops.py                 # MultiIndex utilities
│   └── validation.py                     # Data quality checks
│
├── analysis/                             # Statistical analysis
│   │
│   ├── pollution/                        # Stage 1: Pollution Assessment
│   │   ├── __init__.py
│   │   ├── pipeline.py                   # Orchestration
│   │   ├── transforms.py                 # Log + z-score
│   │   ├── pca.py                        # PCA model
│   │   ├── scoring.py                    # Score computation
│   │   └── validation.py                 # RDA validation
│   │
│   ├── reference_sites/                  # Stage 2: Reference Analysis
│   │   ├── __init__.py
│   │   ├── pipeline.py                   # Orchestration
│   │   ├── selection.py                  # Site selection
│   │   ├── clustering.py                 # Hierarchical clustering
│   │   ├── statistics.py                 # Box-Cox, ANOVA
│   │   └── imputation.py                 # Velocity imputation
│   │
│   ├── environmental/                    # Stage 3: Environment-Taxa
│   │   ├── __init__.py
│   │   ├── pipeline.py                   # Orchestration
│   │   ├── rda.py                        # RDA implementation
│   │   ├── lda.py                        # LDA classification
│   │   └── cross_validation.py           # Monte Carlo CV
│   │
│   ├── community/                        # Stage 4: Community Composition
│   │   ├── __init__.py
│   │   ├── pipeline.py                   # Orchestration
│   │   ├── pca.py                        # PCA on taxa
│   │   ├── zci.py                        # ZCI calculation
│   │   └── regression.py                 # PC diagnostics
│   │
│   └── quantile_regression/              # Stage 5: PQRM
│       ├── __init__.py
│       ├── pipeline.py                   # Orchestration
│       ├── model.py                      # PQRM model
│       ├── bootstrap.py                  # Bootstrap inference
│       └── diagnostics.py                # Model diagnostics
│
├── visualization/                        # All plotting code
│   ├── __init__.py
│   ├── base.py                           # Base figure classes
│   ├── maps.py                           # Geographic plots
│   ├── ordination.py                     # PCA/RDA plots
│   ├── distributions.py                  # Ridge plots, histograms
│   ├── regression.py                     # Regression diagnostics
│   ├── dendrograms.py                    # Cluster trees
│   └── styles.py                         # Matplotlib styling
│
├── reporting/                            # Output generation
│   ├── __init__.py
│   ├── tables.py                         # Table formatting
│   ├── latex.py                          # LaTeX conversion
│   ├── excel.py                          # Excel workbooks
│   └── summary.py                        # Analysis reports
│
└── utils/                                # Utilities
    ├── __init__.py
    ├── logging.py                        # Logging setup
    ├── statistics.py                     # Common stats functions
    └── validation.py                     # Input validation
```

### 2.3 Core Design Patterns

#### **Pattern 1: Pipeline Architecture**

```python
# core/pipeline.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class PipelineContext:
    """Shared context passed through pipeline stages."""
    data: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    artifacts: Dict[str, Any]  # Figures, tables, etc.

class PipelineStage(ABC):
    """Base class for all pipeline stages."""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def validate_inputs(self, context: PipelineContext) -> None:
        """Validate required inputs are present."""
        pass
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute the stage logic."""
        pass
    
    def __call__(self, context: PipelineContext) -> PipelineContext:
        """Run validation then execution."""
        self.validate_inputs(context)
        return self.execute(context)

class Pipeline:
    """Composable pipeline of stages."""
    
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
    
    def run(self, initial_context: PipelineContext) -> PipelineContext:
        """Run all stages in sequence."""
        context = initial_context
        for stage in self.stages:
            print(f"Running stage: {stage.name}")
            context = stage(context)
        return context
```

#### **Pattern 2: Data Models**

```python
# core/data_models.py
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

@dataclass
class PCAResult:
    """Container for PCA results."""
    loadings: pd.DataFrame
    scores: pd.DataFrame
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    n_components: int
    
@dataclass
class ClusterPCAResults:
    """Results from PCA on multiple clusters."""
    cluster_results: Dict[str, PCAResult]
    training_mask: pd.Series
    variance_threshold: float
    
@dataclass
class PollutionAssessment:
    """Complete pollution assessment results."""
    pollution_scores: pd.Series
    pca_result: PCAResult
    transformed_data: pd.DataFrame
    validation_results: Optional[Any] = None
```

#### **Pattern 3: Transform Protocol**

```python
# core/transforms.py
from typing import Protocol
import pandas as pd

class DataTransform(Protocol):
    """Protocol for data transformations."""
    
    def fit(self, data: pd.DataFrame) -> 'DataTransform':
        """Fit the transform to data."""
        ...
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation."""
        ...
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        ...

# preprocessing/transforms.py
class HellingerTransform:
    """Hellinger transformation for compositional data."""
    
    def fit(self, data: pd.DataFrame) -> 'HellingerTransform':
        return self  # Stateless transform
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        row_sums = data.sum(axis=1)
        proportions = data.div(row_sums, axis=0)
        return np.sqrt(proportions)
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.transform(data)
```

---

## 3. Refactoring Strategy

### 3.1 Phase 1: Foundation (Week 1)

**Goal**: Set up new structure without breaking existing code

**Tasks**:
1. Create new directory structure
2. Implement core abstractions (Pipeline, DataModels)
3. Set up configuration management
4. Create I/O module with path management
5. Set up testing infrastructure
6. Set up logging

**Deliverables**:
- New folder structure
- Core base classes
- Test framework
- CI/CD config (if using)

### 3.2 Phase 2: Preprocessing (Week 2, Days 1-2)

**Goal**: Consolidate and refactor data processing

**Tasks**:
1. Move transforms to `preprocessing/transforms.py`
2. Deduplicate Hellinger transform
3. Refactor multiindex operations
4. Add validation utilities
5. Write unit tests for all transforms

**Deliverables**:
- Clean preprocessing module
- 90%+ test coverage
- Updated imports in old modules (backward compatibility)

### 3.3 Phase 3: Analysis Modules (Week 2-3)

**Goal**: Refactor each analysis stage

**For each stage** (Pollution, Reference, Environmental, Community, PQRM):

1. **Extract business logic** from pipeline
2. **Create smaller, focused modules**:
   - Model/algorithm implementation
   - Statistical tests
   - Score/metric computation
3. **Separate I/O operations**
4. **Separate visualization**
5. **Update pipeline to use new components**
6. **Write tests**

**Example: Contamination Assessment**

```python
# Before: contamination_assessment/pipeline.py (400 lines)

# After: Split into:
# analysis/pollution/pipeline.py (100 lines) - orchestration only
# analysis/pollution/transforms.py (50 lines) - log + z-score
# analysis/pollution/pca.py (80 lines) - PCA logic
# analysis/pollution/scoring.py (60 lines) - score computation
# analysis/pollution/validation.py (100 lines) - RDA validation
# visualization/distributions.py (150 lines) - ridge plots
```

### 3.4 Phase 4: Visualization (Week 3)

**Goal**: Consolidate all plotting code

**Tasks**:
1. Move all plotting functions to `visualization/`
2. Create base classes for common plot types
3. Implement consistent styling
4. Separate figure creation from data processing
5. Add figure factory functions

**Example**:
```python
# visualization/base.py
class BaseFigure:
    def __init__(self, figsize=(10, 6), **kwargs):
        self.fig, self.ax = plt.subplots(figsize=figsize, **kwargs)
    
    def save(self, path, formats=['png', 'pdf'], dpi=300):
        """Save figure in multiple formats."""
        # Handled by io.writers
        pass

# visualization/ordination.py
class PCABiplot(BaseFigure):
    def plot(self, scores, loadings, **kwargs):
        # Plotting logic
        pass
```

### 3.5 Phase 5: Integration & Testing (Week 4)

**Goal**: Ensure everything works together

**Tasks**:
1. Integration tests for full pipelines
2. Update notebook examples
3. Performance testing
4. Documentation generation
5. Migration guide
6. Deprecation warnings for old imports

---

## 4. Relationship with Results Folder

### 4.1 Current Relationship

```
results/
├── figures/                 # Saved by output_saver.save_figure()
│   ├── 01_contamination_assessment/
│   ├── 02_taxa_assemblage_in_refs/
│   ├── 03_env_driven_taxa_clusters/
│   ├── 04_community_composition_measures/
│   └── 05_quantile_regressions/
│
├── tables/                  # Saved by output_saver.save_table()
│   └── (various Excel files)
│
├── table_latex/             # Converted by excel_to_latex.py
│   └── (LaTeX tables)
│
├── parameters/              # Saved by output_saver.save_parameters()
│   └── run_*_params.json
│
└── excel_to_latex.py        # Standalone converter
```

### 4.2 Proposed Relationship

**Keep the same output structure**, but improve the implementation:

```python
# config.py
from pathlib import Path
import os

class PathConfig:
    """Centralized path configuration."""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Auto-detect: src/zci -> Project_Code
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = Path(project_root)
        self.results = self.project_root / "results"
        self.figures = self.results / "figures"
        self.tables = self.results / "tables"
        self.latex = self.results / "table_latex"
        self.parameters = self.results / "parameters"
        
    def get_stage_figure_path(self, stage_name: str) -> Path:
        """Get figure directory for a specific stage."""
        stage_dirs = {
            'pollution': '01_contamination_assessment',
            'reference': '02_taxa_assemblage_in_refs',
            'environmental': '03_env_driven_taxa_clusters',
            'community': '04_community_composition_measures',
            'pqrm': '05_quantile_regressions'
        }
        return self.figures / stage_dirs[stage_name]

# io/writers.py
class OutputManager:
    """Manages all output operations."""
    
    def __init__(self, path_config: PathConfig):
        self.paths = path_config
    
    def save_figure(self, fig, name, stage, formats=['png', 'pdf'], **kwargs):
        """Save figure to appropriate directory."""
        save_dir = self.paths.get_stage_figure_path(stage)
        save_dir.mkdir(parents=True, exist_ok=True)
        # ... save logic
    
    def save_table(self, df, name, format='excel', **kwargs):
        """Save table in requested format."""
        # ... save logic
    
    def save_parameters(self, params_dict, run_id=None):
        """Save run parameters as JSON."""
        # ... save logic

# reporting/latex.py
class LaTeXConverter:
    """Convert Excel tables to LaTeX."""
    # Move logic from results/excel_to_latex.py here
    pass
```

**Benefits**:
1. ✅ **Same output locations** - no need to update other scripts
2. ✅ **More flexible** - easy to change paths via config
3. ✅ **Testable** - can use temp directories in tests
4. ✅ **Better organized** - clear separation of concerns

### 4.3 Migration Strategy

1. **Keep backward compatibility**:
   ```python
   # zci/__init__.py
   from .io.writers import OutputManager as OutputSaver
   # Maintains: from zci import OutputSaver
   ```

2. **Preserve excel_to_latex.py** as command-line tool:
   ```python
   # results/excel_to_latex.py (updated)
   from zci.reporting.latex import LaTeXConverter
   
   if __name__ == "__main__":
       converter = LaTeXConverter()
       converter.convert_all_tables()
   ```

---

## 5. Implementation Checklist

### Week 1: Foundation
- [ ] Create new directory structure
- [ ] Implement `core/pipeline.py` with BasePipeline
- [ ] Implement `core/data_models.py` with dataclasses
- [ ] Implement `config.py` with PathConfig
- [ ] Implement `io/writers.py` with OutputManager
- [ ] Set up pytest structure
- [ ] Set up logging configuration
- [ ] Write integration test skeleton

### Week 2: Refactoring Begins
- [ ] Refactor `preprocessing/transforms.py`
- [ ] Deduplicate Hellinger transform
- [ ] Refactor `preprocessing/multiindex_ops.py`
- [ ] Tests for preprocessing (80%+ coverage)
- [ ] Refactor pollution assessment stage
- [ ] Refactor reference sites stage
- [ ] Tests for pollution & reference

### Week 3: Continue Refactoring
- [ ] Refactor environmental analysis stage
- [ ] Refactor community composition stage
- [ ] Refactor PQRM stage
- [ ] Consolidate visualization modules
- [ ] Tests for all analysis modules
- [ ] Update all pipelines to use new structure

### Week 4: Integration & Documentation
- [ ] Integration tests for full workflows
- [ ] Update notebook 08 to use new API
- [ ] Performance testing
- [ ] Write migration guide
- [ ] Generate API documentation
- [ ] Code review & cleanup
- [ ] Create release v2.0.0

---

## 6. Skills and Time Estimates

### 6.1 Required Skills

1. **Python Software Architecture** (Advanced)
   - Design patterns (Factory, Strategy, Template Method)
   - SOLID principles
   - Protocol-oriented design

2. **Scientific Computing** (Intermediate-Advanced)
   - pandas MultiIndex operations
   - NumPy broadcasting
   - scikit-learn API patterns

3. **Testing** (Intermediate)
   - pytest fixtures
   - Mocking/patching
   - Integration testing

4. **Documentation** (Basic-Intermediate)
   - Docstring standards (NumPy/Google style)
   - Sphinx/MkDocs
   - Markdown

5. **Git/Version Control** (Intermediate)
   - Feature branches
   - Rebasing
   - Handling large refactors

### 6.2 Time Estimates

| Phase | Tasks | Hours | Calendar |
|-------|-------|-------|----------|
| **Phase 1: Foundation** | Setup infrastructure | 12-16h | Week 1 |
| **Phase 2: Preprocessing** | Refactor data prep | 8-10h | Week 2 (Mon-Tue) |
| **Phase 3: Analysis** | Refactor 5 stages | 24-32h | Week 2-3 |
| **Phase 4: Visualization** | Consolidate plots | 8-12h | Week 3 |
| **Phase 5: Integration** | Testing & docs | 12-16h | Week 4 |
| **Buffer** | Issues, reviews | 4-8h | Throughout |
| **TOTAL** | | **68-94h** | **3-4 weeks** |

**Assuming**: 20 hours/week of focused work

**Best Case**: 3.5 weeks (68 hours)  
**Expected**: 4 weeks (80 hours)  
**Worst Case**: 5 weeks (94 hours + buffer)

### 6.3 Risk Factors

1. **Hidden dependencies**: Discovering tight coupling (add 5-10h)
2. **Test data requirements**: Creating fixtures (add 4-6h)
3. **Notebook updates**: Updating 8 notebooks (add 8-12h)
4. **Unexpected bugs**: Integration issues (add 5-10h)

---

## 7. Benefits of Refactoring

### 7.1 Immediate Benefits

1. **Easier to modify**: Want to try a different transform? Just swap one class
2. **Easier to test**: Small, focused functions are easier to test
3. **Easier to debug**: Clear data flow, explicit dependencies
4. **Easier to document**: Each module has a clear purpose

### 7.2 Long-term Benefits

1. **Extensibility**: Adding new analysis stages is straightforward
2. **Reusability**: Common components can be reused
3. **Maintainability**: New collaborators can understand the code
4. **Performance**: Easier to optimize specific bottlenecks
5. **Professionalism**: Publication-ready code quality

### 7.3 Example: Adding a New Analysis

**Before** (current structure):
```
1. Create new pipeline file (300+ lines)
2. Copy-paste visualization code
3. Copy-paste I/O code
4. Debug mysterious errors
5. Update 3-4 other files
Time: 2-3 days
```

**After** (refactored structure):
```python
# 1. Create new stage
class NewAnalysisStage(PipelineStage):
    def execute(self, context):
        # Your logic here (50 lines)
        pass

# 2. Use existing components
transform = HellingerTransform()
visualizer = OrdinationPlot()
output = OutputManager()

# 3. Compose pipeline
pipeline = Pipeline([
    DataLoadStage(),
    NewAnalysisStage(),
    VisualizationStage(visualizer),
    OutputStage(output)
])

Time: 4-6 hours
```

---

## 8. Recommendations

### 8.1 Immediate Actions

1. **Start with Phase 1** - Set up the foundation
2. **Create a feature branch** - Don't work on main
3. **Write tests first** - Ensure old behavior is preserved
4. **Refactor incrementally** - One module at a time
5. **Keep old code** - Don't delete until new code is proven

### 8.2 Best Practices During Refactoring

1. **Use deprecation warnings**:
   ```python
   import warnings
   
   def old_function(*args, **kwargs):
       warnings.warn(
           "old_function is deprecated, use new_function instead",
           DeprecationWarning,
           stacklevel=2
       )
       return new_function(*args, **kwargs)
   ```

2. **Maintain backward compatibility**:
   ```python
   # zci/__init__.py
   from .io.writers import OutputManager
   # Allow both:
   from zci.io import OutputManager  # New way
   from zci import OutputSaver  # Old way (alias)
   OutputSaver = OutputManager
   ```

3. **Document changes**:
   ```markdown
   # CHANGELOG.md
   ## v2.0.0 (2026-03-15)
   ### Breaking Changes
   - Moved output_saver.py -> io/writers.py
   ### Migration Guide
   - Old: `from zci.output_saver import save_figure`
   - New: `from zci.io import OutputManager`
   ```

### 8.3 When to Refactor

**Good times**:
- ✅ Between analysis phases
- ✅ Before adding major new features
- ✅ When code becomes hard to modify
- ✅ When you have time for proper testing

**Bad times**:
- ❌ During active data analysis
- ❌ Close to paper submission deadlines
- ❌ When you're the only developer and need results quickly

### 8.4 Alternative: Incremental Refactoring

If 4 weeks is too much, consider **incremental refactoring**:

1. **Just preprocessing** (Week 1): Clean up transforms, 8-10h
2. **Just one analysis stage** (Week 2): E.g., pollution assessment, 12-16h
3. **Just visualization** (Week 3): Consolidate plots, 8-12h
4. Continue as time allows

This gives immediate benefits while spreading the work over a longer period.

---

## 9. Conclusion

The `zci` package has grown organically and now needs restructuring for:
- **Clarity**: Clear separation of concerns
- **Maintainability**: Easy to modify and extend
- **Testability**: Comprehensive test coverage
- **Professionalism**: Publication-ready code

**Estimated effort**: 3-4 weeks (68-94 hours)

**Key skills**: Python architecture, scientific computing, testing, documentation

**Recommendation**: Start with Phase 1 (foundation) and proceed incrementally. The investment will pay off in easier maintenance, faster development of new features, and higher code quality for publications.

**Preserve relationships**: All existing output paths and relationships with `results/` folder will be maintained through configuration management.

---

## Appendix A: File Migration Map

| Old Location | New Location | Notes |
|-------------|--------------|-------|
| `output_saver.py` | `io/writers.py` | Renamed, split into writers/readers |
| `data_process/transform.py` | `preprocessing/transforms.py` | Merged, deduplicated |
| `data_process/dataframe_ops.py` | `preprocessing/multiindex_ops.py` | Renamed for clarity |
| `contamination_assessment/*` | `analysis/pollution/*` | Split into smaller modules |
| `taxa_assemblage_in_refs/*` | `analysis/reference_sites/*` | Split into smaller modules |
| `env_driven_taxa_clusters/*` | `analysis/environmental/*` | Split into smaller modules |
| `community_composition_measures/*` | `analysis/community/*` | Split into smaller modules |
| `piecewise_quantile_regression/*` | `analysis/quantile_regression/*` | Minor renaming |
| `map_plots/` | `visualization/maps.py` | Consolidated |
| All `*_visualization.py` | `visualization/*` | Organized by plot type |

## Appendix B: Testing Strategy

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_taxa_data():
    """Sample taxa abundance data."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.rand(50, 20),
        columns=[f'taxa_{i}' for i in range(20)]
    )

@pytest.fixture
def sample_pollution_data():
    """Sample pollution data."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.rand(50, 10),
        columns=[f'pollutant_{i}' for i in range(10)]
    )

# tests/preprocessing/test_transforms.py
def test_hellinger_transform(sample_taxa_data):
    from zci.preprocessing.transforms import HellingerTransform
    
    transform = HellingerTransform()
    result = transform.fit_transform(sample_taxa_data)
    
    # Check row sums are approximately 1
    row_sums = (result ** 2).sum(axis=1)
    assert np.allclose(row_sums, 1.0)

# tests/integration/test_full_pipeline.py
def test_contamination_assessment_pipeline(sample_pollution_data):
    """Integration test for full contamination assessment."""
    from zci.analysis.pollution.pipeline import PollutionAssessmentPipeline
    
    pipeline = PollutionAssessmentPipeline()
    results = pipeline.run(sample_pollution_data)
    
    assert 'pollution_scores' in results
    assert len(results['pollution_scores']) == len(sample_pollution_data)
```

---

**Document Version**: 1.0  
**Date**: February 14, 2026  
**Author**: GitHub Copilot  
**Status**: Proposal - Awaiting Review
