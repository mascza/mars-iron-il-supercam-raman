# Reference library

Lab Raman reference data for the four iron-based ionic liquids screened against SuperCam, prepared as inputs to the matching stage.

- `reference_spectra.parquet` -- long-form spectra for each (compound, laser, power, spectrum, wavenumber). Carries the lab spectrum (`intensity_normalized_lab`) and the SuperCam-resolution version (`intensity_normalized_convolved`, after Gaussian convolution at FWHM 12 cm-1 and re-normalization). The `usable_for_supercam` flag selects the 532 nm subset that matches SuperCam's Raman laser.
- `peak_catalog.parquet` -- Lorentzian-fit peak parameters at both lab and convolved resolution, with per-spectrum rows and aggregate rows. Aggregate-convolved rows are class-labeled (`class_label` plus `class_notes`) under a three-tier A/B/C anion scheme plus a separate `EMIM-cation` compound for shared cation peaks. The matcher consumes the aggregate-convolved subset.

Full lab acquisitions (raw WITec output, additional laser/power settings, and provenance) are archived on Harvard Dataverse -- see the top-level README for the DOI.
