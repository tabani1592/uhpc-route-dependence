# UHPC route-dependence: data and analysis code

Data, study-level coding and analysis code for:

> A. Tabani and R. Biswas, "Mixture or Method? A Quantitative Review of Route-Dependence in the Reported Performance of Ultra-High-Performance Concrete" (submitted).

The paper asks how far the reported properties of ultra-high-performance concrete (UHPC) reflect the mixture itself, and how far they reflect how it was cured, sampled and tested. This repository reproduces every quantitative result in the paper that is based on the authors' extraction from 67 studies (64 experimental). That covers four analyses:

- the within-study contrasts;
- the tensile test-method ratios and flexural size effect;
- the linear mixed-effects model with its sensitivity analyses;
- the reporting audit.

## Contents

| Folder | File | Description |
|---|---|---|
| `data/` | `UHPC_mechanical_data_v17.xlsx` | Value-level mechanical records extracted from the reviewed studies (one row per reported value), with paper index and field legend |
| `data/` | `UHPC_durability_data_v16.xlsx` | Value-level durability records, by category, with paper index and legend |
| `data/` | `UHPC_cumulative_comparison_corrected_v16.xlsx` | Study-level master sheet: curing regime, test ages, fibre, composition and outcomes for each study |
| `coding/` | `coding_sheet_v2.xlsx` | Curing class per study, include/exclude decision and reason for every compressive value, and the within-study contrast groups |
| `code/` | `uhpc_route_analysis.py` | Complete analysis script |
| `results/` | `route_analysis_results_v3.xlsx`, figures, `mixed_model_summary.txt` | Outputs as reported in the paper |

The `Study_no` field in the data files equals the row number in the master sheet minus one. Study 21 is excluded because it fails the UHPC screening criteria; the two reviews (32, 42) and the numerical study (62) are excluded from the experimental analyses.

## How to run

```bash
pip install -r requirements.txt
python code/uhpc_route_analysis.py
```

The script reads `data/` and `coding/coding_sheet_v2.xlsx` and writes all tables and figures to `results/`. It runs in a few seconds and shows progress bars. Coding decisions can be changed in the coding sheet and the analysis re-run.

Model: `ln(fc) ~ curing class + age band + fibre-volume band + (1 | study)`, fitted by REML with `statsmodels` MixedLM. Marginal and conditional R² are computed from the fixed-effect prediction variance and the variance components (see `pooled_model()` in the script).

## Third-party datasets (not redistributed)

The pooled distributions in the paper (810 compressive, 321 flexural and 296 direct-tensile records) come from published datasets. They are not included here; please obtain them from the original sources:

- A. Kashem, R. Karim, S. C. Malo, and P. Das, "Ultra-High-Performance Concrete (UHPC)," Mendeley Data, vol. 1, 2023, doi: [10.17632/85R7BH4ZSZ.1](https://doi.org/10.17632/85R7BH4ZSZ.1)
- M. Bolbolvand, S. M. Tavakkoli, and F. J. Alaee, Constr. Build. Mater., vol. 493, 143135, 2025, doi: [10.1016/j.conbuildmat.2025.143135](https://doi.org/10.1016/j.conbuildmat.2025.143135)
- U. J. Malik, D. Mohotti, H. Mo, and C. K. Lee, Data Brief, vol. 63, 112179, 2025, doi: [10.1016/j.dib.2025.112179](https://doi.org/10.1016/j.dib.2025.112179)

## Citation

If you use these data or code, please cite the paper above and this repository (see `CITATION.cff`).

## Licence

- Code: MIT (see `LICENSE`).
- Data and coding sheet: CC BY 4.0 (see `data/LICENSE-DATA.md`).

The extracted values remain attributable to their original publications, which are listed in the master sheet.

## Contact

Ahmadullah Tabani (dt23amc003@students.vnit.ac.in) and Rahul Biswas (rahulbiswas@apm.vnit.ac.in), Department of Applied Mechanics, Visvesvaraya National Institute of Technology, Nagpur, India.
