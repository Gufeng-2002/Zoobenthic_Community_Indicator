#!/usr/bin/env python
"""Compare Ward-targeted MRT results across 5 taxa transformation methods.

Outputs are prefixed with the transform name and stored in a temporary
subfolder ``MRT_Method/transform_comparison/``.

Usage (from project root):
    python src/compare_mrt_transforms.py
"""

from pathlib import Path

from zci.pipeline.mrt import mrt_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complete_env_taxa_chemical_Feb_3.xlsx"
MAPS_DIR = PROJECT_ROOT / "data" / "maps"
STAGE1_ARTIFACT = (
    PROJECT_ROOT / "results" / "01_pollution_assessment" / "PCA_Stressors" / "artifacts" / "SumRel_01_updated_data.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "02_taxa_assemblage" / "MRT_Method" / "transform_comparison"

TRANSFORMS = {
    "octave": ("octave", "octave"),
    "raw": ("relative_abundance", "raw (relative abundance)"),
    "hellinger": ("hellinger", "Hellinger"),
    "chord": ("chord", "chord"),
    "logchord": ("log_chord", "log-chord"),
}


if __name__ == "__main__":
    for key, (transform_name, label) in TRANSFORMS.items():
        print(f"\n{'=' * 60}")
        print(f"  Transform: {label}  (prefix = {key}_)")
        print(f"{'=' * 60}\n")

        result = mrt_pipeline(
            data_path=DATA_PATH,
            stage1_artifact=STAGE1_ARTIFACT,
            output_dir=OUTPUT_DIR,
            maps_dir=MAPS_DIR,
            output_prefix=f"{key}_",
            response_transform=transform_name,
            reference_quantile=0.22,
            n_clusters=3,
            k_folds=10,
            cv_perms=100,
            minsplit=5,
            minbucket=2,
            random_state=42,
            verbose=True,
        )
        print(f"\n{result.summary()}")

    print("\n" + "=" * 60)
    print("  All 5 transforms completed.")
    print(f"  Outputs in: {OUTPUT_DIR}")
    print("=" * 60)
