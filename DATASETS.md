# Dataset sources

This repository intentionally does not commit raw cacao images. The image datasets are large and may have source-specific usage restrictions. Use the scripts below to download and prepare a local copy.

## Recommended starting point

- Kaggle: [Cacao Diseases in Davao](https://www.kaggle.com/datasets/zaldyjr/cacao-diseases-in-davao)
- Reported classes include `Healthy`, `Black Pod Rot`, and `Pod Borer`.

## Download

1. Install the dependencies: `pip install -r requirements.txt`.
2. Configure Kaggle credentials using the official Kaggle CLI instructions. Never commit `kaggle.json`.
3. Run `.\scripts\download_kaggle.ps1`.

The downloader tries the known Kaggle slugs and stores the archive contents under `data/raw/`.

## Prepare ImageFolder data

```powershell
python scripts/split_imagefolder.py --source data/raw --output data/cacao
```

The resulting structure is:

```text
data/cacao/{train,val,test}/{black_pod,healthy,other}/...
```

Class names are normalized by the splitter. Inspect the generated class counts before training, and confirm that the dataset license and intended use are appropriate for your project.

## Dataset integration

Use `configs/label_mapping.json` and `scripts/build_manifest.py` to normalize multiple labeled sources without relabeling each image manually:

```powershell
python scripts/build_manifest.py `
  --dataset cacao_images_zenodo=data/zenodo/extracted/ChrisRosales-CacaoImages-843a58a `
  --dataset davao=data/raw/davao `
  --mapping configs/label_mapping.json `
  --output data/manifest.csv
```

The manifest retains both `source_class` and `target_class`, along with image dimensions, source split, and a SHA-256 digest. A missing class mapping or corrupt image fails the command so that silent label mistakes do not enter training.
