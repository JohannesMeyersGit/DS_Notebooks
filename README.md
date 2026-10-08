# Data science lecture notebooks

## Launch with Binder

[Launch the gull-classification lecture on Binder](https://mybinder.org/v2/gh/JohannesMeyersGit/DS_Notebooks/HEAD?labpath=Lecture_Emden%2FProbeLehrveranstaltung.ipynb).
The first build may take several minutes. Commit and push the root
`environment.yml` and notebook changes before launching; Binder uses the
repository on GitHub, not local files.

In JupyterLab, select the Python kernel and use **Run > Run All Cells**.
Other notebooks can be opened from the file browser but may need additional
packages not included in this minimal lecture environment. Run them with the
working directory set to their containing folder (JupyterLab's default),
so their `data/` paths find the included datasets. The gull lecture also
supports execution from the repository root.

The environment uses Python 3.11 and conda-forge packages for the gull lecture:
pandas, Matplotlib, scikit-learn and Pillow, plus the Jupyter runtime.
No pip-only dependency or GPU/model runtime is needed for this notebook.
The gull CSV and images are included; they do not need to be regenerated
or downloaded. No lecture notebook requires the removed ONNX model artifacts.

The other collections use additional libraries such as SciPy, seaborn,
statsmodels, ipywidgets, ipympl, matplotlib-venn, Graphviz, ISLP and
PyTorch/torchvision. Their local statistical-learning datasets and MNIST are
included; ISLP supplies the `Default` dataset. These packages are intentionally
not installed for the gull lecture.

For a local installation with conda:

```sh
conda env create -f environment.yml
conda activate ds-notebooks
jupyter lab
```
