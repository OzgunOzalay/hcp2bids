# HCP_1200 to BIDS Converter

This script converts the Human Connectome Project 1200 dataset from its original format to BIDS (Brain Imaging Data Structure) format.

## Features

- Converts T1w structural MRI data
- Converts DWI diffusion MRI data (merges multiple runs)
- Handles field maps (magnitude and phase)
- Creates proper BIDS metadata and JSON sidecars
- Supports batch processing of multiple subjects
- Includes comprehensive logging

## Requirements

- Python 3.7+
- Required packages (see `requirements.txt`):
  - nibabel
  - numpy
  - pandas

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Convert all subjects in the HCP dataset:
```bash
python convert_hcp_to_bids.py /path/to/hcp_dataset /path/to/bids_output
```

### Test Run

Test the conversion on the first subject only:
```bash
python convert_hcp_to_bids.py /path/to/hcp_dataset /path/to/bids_output --test
```

### Convert Specific Subjects

Convert only specific subjects:
```bash
python convert_hcp_to_bids.py /path/to/hcp_dataset /path/to/bids_output --subjects 103818 105923 114823
```

## Input Data Structure

The script expects the HCP_1200 dataset to be organized as follows:
```
hcp_dataset/
├── 103818/
│   └── unprocessed/
│       └── 3T/
│           ├── T1w_MPR1/
│           │   ├── 103818_3T_T1w_MPR1.nii.gz
│           │   ├── 103818_3T_FieldMap_Magnitude.nii.gz
│           │   └── 103818_3T_FieldMap_Phase.nii.gz
│           └── Diffusion/
│               ├── 103818_3T_DWI_dir95_LR.nii.gz
│               ├── 103818_3T_DWI_dir95_LR.bval
│               ├── 103818_3T_DWI_dir95_LR.bvec
│               └── ... (additional DWI runs)
├── 105923/
└── ... (additional subjects)
```

## Output BIDS Structure

The script creates a BIDS-compliant dataset:
```
bids_output/
├── dataset_description.json
├── participants.tsv
├── sub-103818/
│   ├── anat/
│   │   ├── sub-103818_T1w.nii.gz
│   │   ├── sub-103818_T1w.json
│   │   ├── sub-103818_magnitude1.nii.gz
│   │   ├── sub-103818_magnitude1.json
│   │   ├── sub-103818_phasediff.nii.gz
│   │   └── sub-103818_phasediff.json
│   └── dwi/
│       ├── sub-103818_dwi.nii.gz
│       ├── sub-103818_dwi.bval
│       ├── sub-103818_dwi.bvec
│       ├── sub-103818_dwi.json
│       └── sub-103818_dwi_sbref.nii.gz
└── ... (additional subjects)
```

## DWI Processing

The script automatically:
1. Identifies all DWI runs for each subject
2. Merges multiple DWI acquisitions into a single 4D volume
3. Concatenates corresponding bval and bvec files
4. Creates appropriate JSON metadata

## Field Map Processing

For subjects with field map data:
- Magnitude images are converted to BIDS `magnitude1` format
- Phase difference images are converted to BIDS `phasediff` format
- Appropriate JSON metadata is created

## Validation

After conversion, you can validate the BIDS dataset using the official BIDS validator:
```bash
bids-validator /path/to/bids_output
```

## Notes

- The script preserves the original data and creates copies in BIDS format
- Large datasets may require significant disk space
- Processing time depends on the number of subjects and data size
- Check the logs for any conversion issues or warnings

## Troubleshooting

1. **Memory issues**: For large datasets, consider processing subjects in batches
2. **Missing files**: The script will log warnings for missing data files
3. **Permission errors**: Ensure write permissions for the output directory

## License

This script is provided as-is for research purposes. Please cite the Human Connectome Project when using the converted data. 