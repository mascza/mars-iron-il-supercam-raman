# Mars processed SuperCam shots

`mars_processed.parquet` — calibrated SuperCam Raman shots used by the screen.
1,436 shots across 202 sols and 165 targets, spanning sol 13 to sol 1695, with
SuperCam Calibration Target (SCCT) shots excluded.

## Source and citation

Raw data: SuperCam instrument on the NASA Mars 2020 Perseverance rover.
Publicly distributed via the NASA Planetary Data System (PDS) Geosciences Node:
https://pds-geosciences.wustl.edu/missions/mars2020/supercam.htm

When using this dataset, please cite the SuperCam instrument team:

- Wiens, R. C., et al. (2021). The SuperCam Instrument Suite on the NASA Mars
  2020 Rover: Body Unit and Combined System Tests. *Space Science Reviews*,
  217(4), 1-87. https://doi.org/10.1007/s11214-020-00777-5
- Maurice, S., et al. (2021). The SuperCam Instrument Suite on the Mars 2020
  Rover: Science Objectives and Mast-Unit Description. *Space Science Reviews*,
  217(3), 1-108. https://doi.org/10.1007/s11214-021-00807-w

## Processing

`mars_processed.parquet` is the calibrated subset produced by
`scripts/03_ingest_supercam.py` from the PDS source bundles. Processing steps
are documented in that script and in the manuscript Methods section.
The file is retained in this repository so the screen is reproducible without
re-downloading the raw PDS archive.

## License

NASA PDS data are in the public domain (NASA Open Data policy). This
derived/calibrated subset is redistributed under the same terms.
