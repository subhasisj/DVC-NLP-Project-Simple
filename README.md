# DVC Starter NLP Project

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.7 -y 
```

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### `Trick`: Create a conda env in same directory as the repository and activate it.
```bash
conda create --prefix ./env python=3.7 -y && source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 05- initialize the dvc project
```bash
dvc init
```

### STEP 06- commit and push the changes to the remote repository