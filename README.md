# Mars iron-based ionic liquid SuperCam Raman screening pipeline

This pipeline screens publicly available Mars 2020 SuperCam Raman spectra for signatures of four iron-based ionic liquids: [EMIM][FeCl4], [EMIM][FeBr4], [EMIM]2[Fe2Cl7], and [EMIM]2[Fe(SO4)2]. Lab Raman spectra of the four compounds are convolved to SuperCam resolution, Lorentzian-fit into a peak catalog, and matched against calibrated SuperCam shots from Jezero crater; the output is a tiered list of candidate targets and shots for follow-up.

## Manuscript and data

- Manuscript: DOI placeholder (to be filled in on submission).
- Harvard Dataverse (full lab Raman, peak catalog provenance, supplementary raw PDS context): https://doi.org/10.7910/DVN/YYKERD
- `data/mars_processed/mars_processed.parquet` is the calibrated SuperCam subset actually used by the screen, derived from the raw PDS data and redistributed under NASA's open data policy with attribution (see `data/mars_processed/README.md`). The ingest script (`scripts/03_ingest_supercam.py`) documents the calibration steps and can regenerate this file from the raw PDS sources.

## Reproducing the screen

```
conda env create -f environment.yml
conda activate mars-raman-pipeline
```

Then run the scripts in `scripts/` in numerical order. `_processing.py` is a shared module imported by several stages and is not run directly.

## License

MIT. See `LICENSE`.

## Contact

Iaroslav Iakubivskyi -- iakubivskyi@mit.edu
