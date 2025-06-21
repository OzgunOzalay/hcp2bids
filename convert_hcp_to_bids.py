#!/usr/bin/env python3
"""
HCP_1200 to BIDS Converter

This script converts the Human Connectome Project 1200 dataset from its original
format to BIDS (Brain Imaging Data Structure) format.

Author: AI Assistant
Date: 2024
"""

import os
import json
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HCPToBIDSConverter:
    """Convert HCP_1200 dataset to BIDS format."""
    
    def __init__(self, hcp_root: str, bids_root: str):
        """
        Initialize the converter.
        
        Args:
            hcp_root: Path to HCP_1200 dataset root directory
            bids_root: Path to output BIDS dataset directory
        """
        self.hcp_root = Path(hcp_root)
        self.bids_root = Path(bids_root)
        
        # Create BIDS directory structure
        self.bids_root.mkdir(parents=True, exist_ok=True)
        
        # BIDS metadata templates
        self.dataset_description = {
            "Name": "HCP_1200",
            "BIDSVersion": "1.7.0",
            "DatasetType": "raw",
            "Authors": ["Human Connectome Project"],
            "HowToAcknowledge": "Please cite the Human Connectome Project",
            "Funding": ["NIH"],
            "EthicsApprovals": ["Washington University IRB"],
            "ReferencesAndLinks": [
                "https://www.humanconnectome.org/study/hcp-young-adult"
            ],
            "DatasetDOI": "10.15387/fcp_indi.corr.hcp_1200"
        }
        
        # T1w JSON sidecar template
        self.t1w_json_template = {
            "Manufacturer": "Siemens",
            "ManufacturersModelName": "Skyra",
            "MagneticFieldStrength": 3.0,
            "ScanningSequence": "GR/IR",
            "SequenceVariant": "SP/MP",
            "ScanOptions": "IR",
            "SequenceName": "*tfl3d1_16ns",
            "RepetitionTime": 2.4,
            "EchoTime": 0.00214,
            "InversionTime": 1.0,
            "FlipAngle": 8,
            "MultibandAccelerationFactor": 8,
            "PhaseEncodingDirection": "j-",
            "EffectiveEchoSpacing": 0.00058,
            "EchoTrainLength": 1,
            "PixelBandwidth": 240,
            "ReceiveCoilActiveElements": "HEA;HEP",
            "SoftwareVersions": "syngo MR E11",
            "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"],
            "ProtocolName": "T1w_MPR1",
            "ImageComments": "HCP_1200 T1w structural scan"
        }
        
        # DWI JSON sidecar template
        self.dwi_json_template = {
            "Manufacturer": "Siemens",
            "ManufacturersModelName": "Skyra",
            "MagneticFieldStrength": 3.0,
            "ScanningSequence": "EP",
            "SequenceVariant": "SK",
            "ScanOptions": "FS",
            "SequenceName": "*ep_b0",
            "RepetitionTime": 5.52,
            "EchoTime": 0.0895,
            "FlipAngle": 78,
            "MultibandAccelerationFactor": 3,
            "PhaseEncodingDirection": "j-",
            "EffectiveEchoSpacing": 0.00078,
            "EchoTrainLength": 1,
            "PixelBandwidth": 1502,
            "ReceiveCoilActiveElements": "HEA;HEP",
            "SoftwareVersions": "syngo MR E11",
            "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND", "DIFFUSION"],
            "ProtocolName": "DWI_dir95_LR",
            "ImageComments": "HCP_1200 diffusion weighted imaging"
        }
    
    def get_subject_ids(self) -> List[str]:
        """Get list of all subject IDs from HCP dataset."""
        subject_ids = []
        for item in self.hcp_root.iterdir():
            if item.is_dir() and item.name.isdigit():
                subject_ids.append(item.name)
        return sorted(subject_ids)
    
    def create_bids_structure(self):
        """Create basic BIDS directory structure and metadata files."""
        logger.info("Creating BIDS directory structure...")
        
        # Create dataset_description.json
        dataset_desc_path = self.bids_root / "dataset_description.json"
        with open(dataset_desc_path, 'w') as f:
            json.dump(self.dataset_description, f, indent=2)
        
        # Create participants.tsv
        subject_ids = self.get_subject_ids()
        participants_df = pd.DataFrame({
            'participant_id': [f'sub-{sid}' for sid in subject_ids],
            'age': [None] * len(subject_ids),  # Will be filled if available
            'sex': [None] * len(subject_ids),  # Will be filled if available
            'group': ['HCP_1200'] * len(subject_ids)
        })
        participants_path = self.bids_root / "participants.tsv"
        participants_df.to_csv(participants_path, sep='\t', index=False)
        
        logger.info(f"Created BIDS structure with {len(subject_ids)} subjects")
    
    def convert_t1w_data(self, subject_id: str):
        """Convert T1w structural data for a single subject."""
        logger.info(f"Converting T1w data for subject {subject_id}")
        
        # Create subject directory
        subject_dir = self.bids_root / f"sub-{subject_id}" / "anat"
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Source T1w file
        t1w_source = self.hcp_root / subject_id / "unprocessed" / "3T" / "T1w_MPR1" / f"{subject_id}_3T_T1w_MPR1.nii.gz"
        
        if not t1w_source.exists():
            logger.warning(f"T1w file not found for subject {subject_id}: {t1w_source}")
            return False
        
        # Copy T1w file
        t1w_dest = subject_dir / f"sub-{subject_id}_T1w.nii.gz"
        shutil.copy2(t1w_source, t1w_dest)
        
        # Create T1w JSON sidecar
        t1w_json = subject_dir / f"sub-{subject_id}_T1w.json"
        with open(t1w_json, 'w') as f:
            json.dump(self.t1w_json_template, f, indent=2)
        
        # Handle field maps if they exist
        self._convert_fieldmaps(subject_id, subject_dir)
        
        logger.info(f"T1w conversion completed for subject {subject_id}")
        return True
    
    def _convert_fieldmaps(self, subject_id: str, anat_dir: Path):
        """Convert field map data to BIDS format."""
        t1w_dir = self.hcp_root / subject_id / "unprocessed" / "3T" / "T1w_MPR1"
        
        # Check for magnitude field map
        magnitude_source = t1w_dir / f"{subject_id}_3T_FieldMap_Magnitude.nii.gz"
        if magnitude_source.exists():
            magnitude_dest = anat_dir / f"sub-{subject_id}_magnitude1.nii.gz"
            shutil.copy2(magnitude_source, magnitude_dest)
            
            # Create magnitude JSON
            magnitude_json = anat_dir / f"sub-{subject_id}_magnitude1.json"
            magnitude_metadata = {
                "Units": "Hz",
                "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"]
            }
            with open(magnitude_json, 'w') as f:
                json.dump(magnitude_metadata, f, indent=2)
        
        # Check for phase field map
        phase_source = t1w_dir / f"{subject_id}_3T_FieldMap_Phase.nii.gz"
        if phase_source.exists():
            phase_dest = anat_dir / f"sub-{subject_id}_phasediff.nii.gz"
            shutil.copy2(phase_source, phase_dest)
            
            # Create phase JSON
            phase_json = anat_dir / f"sub-{subject_id}_phasediff.json"
            phase_metadata = {
                "Units": "rad/s",
                "EchoTime1": 0.00246,
                "EchoTime2": 0.00492,
                "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"]
            }
            with open(phase_json, 'w') as f:
                json.dump(phase_metadata, f, indent=2)
    
    def convert_dwi_data(self, subject_id: str):
        """Convert DWI diffusion data for a single subject."""
        logger.info(f"Converting DWI data for subject {subject_id}")
        
        # Create subject directory
        subject_dir = self.bids_root / f"sub-{subject_id}" / "dwi"
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all DWI files for this subject
        dwi_dir = self.hcp_root / subject_id / "unprocessed" / "3T" / "Diffusion"
        
        if not dwi_dir.exists():
            logger.warning(f"DWI directory not found for subject {subject_id}: {dwi_dir}")
            return False
        
        # Find all DWI runs
        dwi_runs = []
        for file in dwi_dir.glob(f"{subject_id}_3T_DWI_dir*_*.nii.gz"):
            if not file.name.endswith("_SBRef.nii.gz"):
                dwi_runs.append(file)
        
        if not dwi_runs:
            logger.warning(f"No DWI runs found for subject {subject_id}")
            return False
        
        # Group runs by phase encoding direction (LR vs RL)
        lr_runs = [run for run in dwi_runs if "_LR.nii.gz" in run.name]
        rl_runs = [run for run in dwi_runs if "_RL.nii.gz" in run.name]
        
        # Sort runs within each group
        lr_runs.sort()
        rl_runs.sort()
        
        success = False
        
        # Merge LR runs if they exist
        if lr_runs:
            logger.info(f"Found {len(lr_runs)} LR runs for subject {subject_id}")
            self._merge_dwi_runs(subject_id, lr_runs, subject_dir, "LR")
            success = True
        
        # Merge RL runs if they exist
        if rl_runs:
            logger.info(f"Found {len(rl_runs)} RL runs for subject {subject_id}")
            self._merge_dwi_runs(subject_id, rl_runs, subject_dir, "RL")
            success = True
        
        if not success:
            logger.warning(f"No LR or RL DWI runs found for subject {subject_id}")
            return False
        
        logger.info(f"DWI conversion completed for subject {subject_id}")
        return True
    
    def _merge_dwi_runs(self, subject_id: str, dwi_runs: List[Path], dwi_dir: Path, phase_encoding: str):
        """Merge multiple DWI runs into a single 4D volume for a specific phase encoding direction."""
        logger.info(f"Merging {len(dwi_runs)} {phase_encoding} DWI runs for subject {subject_id}")
        
        # Load and concatenate DWI data
        dwi_data_list = []
        bval_list = []
        bvec_list = []
        
        for dwi_run in dwi_runs:
            # Load DWI data
            dwi_img = nib.load(dwi_run)
            dwi_data_list.append(dwi_img.get_fdata())
            
            # Load corresponding bval and bvec files
            # Extract the base name without .nii.gz extension
            run_base = dwi_run.stem  # This removes .nii.gz
            # The stem method removes the last extension, but we need to remove .nii.gz completely
            run_base = run_base.replace('.nii', '')  # Remove .nii if present
            bval_file = dwi_run.parent / f"{run_base}.bval"
            bvec_file = dwi_run.parent / f"{run_base}.bvec"
            
            logger.info(f"Looking for bval: {bval_file}")
            logger.info(f"Looking for bvec: {bvec_file}")
            
            if bval_file.exists() and bvec_file.exists():
                bval = np.loadtxt(bval_file)
                bvec = np.loadtxt(bvec_file)
                bval_list.append(bval)
                bvec_list.append(bvec)
                logger.info(f"Successfully loaded bval/bvec for {dwi_run.name}")
            else:
                logger.warning(f"Missing bval/bvec files for {dwi_run}")
                logger.warning(f"bval exists: {bval_file.exists()}")
                logger.warning(f"bvec exists: {bvec_file.exists()}")
        
        if not dwi_data_list:
            raise ValueError("No DWI data loaded")
        
        if not bval_list or not bvec_list:
            raise ValueError("No bval/bvec data loaded")
        
        # Concatenate data along time dimension
        merged_dwi = np.concatenate(dwi_data_list, axis=3)
        merged_bval = np.concatenate(bval_list)
        merged_bvec = np.concatenate(bvec_list, axis=1)
        
        # Save merged DWI data with phase encoding direction in filename
        merged_img = nib.Nifti1Image(merged_dwi, dwi_img.affine, dwi_img.header)
        dwi_dest = dwi_dir / f"sub-{subject_id}_dir-{phase_encoding}_dwi.nii.gz"
        nib.save(merged_img, dwi_dest)
        
        # Save merged bval and bvec
        bval_dest = dwi_dir / f"sub-{subject_id}_dir-{phase_encoding}_dwi.bval"
        bvec_dest = dwi_dir / f"sub-{subject_id}_dir-{phase_encoding}_dwi.bvec"
        
        np.savetxt(bval_dest, merged_bval, fmt='%.0f')
        np.savetxt(bvec_dest, merged_bvec, fmt='%.6f')
        
        # Create DWI JSON sidecar with phase encoding direction
        dwi_json_template = self.dwi_json_template.copy()
        if phase_encoding == "LR":
            dwi_json_template["PhaseEncodingDirection"] = "j-"
        elif phase_encoding == "RL":
            dwi_json_template["PhaseEncodingDirection"] = "j"
        
        dwi_json = dwi_dir / f"sub-{subject_id}_dir-{phase_encoding}_dwi.json"
        with open(dwi_json, 'w') as f:
            json.dump(dwi_json_template, f, indent=2)
        
        # Handle SBRef images for this phase encoding direction
        self._convert_sbref_images(subject_id, dwi_dir, phase_encoding)
    
    def _convert_sbref_images(self, subject_id: str, dwi_dir: Path, phase_encoding: str):
        """Convert SBRef (single-band reference) images for a specific phase encoding direction."""
        dwi_source_dir = self.hcp_root / subject_id / "unprocessed" / "3T" / "Diffusion"
        
        # Find SBRef images for this phase encoding direction
        sbref_files = list(dwi_source_dir.glob(f"{subject_id}_3T_DWI_dir*_{phase_encoding}_SBRef.nii.gz"))
        
        if sbref_files:
            # For now, just copy the first SBRef (could be enhanced to merge multiple)
            sbref_dest = dwi_dir / f"sub-{subject_id}_dir-{phase_encoding}_dwi_sbref.nii.gz"
            shutil.copy2(sbref_files[0], sbref_dest)
            
            # Create SBRef JSON
            sbref_json = dwi_dir / f"sub-{subject_id}_dir-{phase_encoding}_dwi_sbref.json"
            sbref_metadata = {
                "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"],
                "ProtocolName": f"DWI_SBRef_{phase_encoding}"
            }
            if phase_encoding == "LR":
                sbref_metadata["PhaseEncodingDirection"] = "j-"
            elif phase_encoding == "RL":
                sbref_metadata["PhaseEncodingDirection"] = "j"
            
            with open(sbref_json, 'w') as f:
                json.dump(sbref_metadata, f, indent=2)
    
    def convert_subject(self, subject_id: str):
        """Convert all data for a single subject."""
        logger.info(f"Converting subject {subject_id}")
        
        success_t1w = self.convert_t1w_data(subject_id)
        success_dwi = self.convert_dwi_data(subject_id)
        
        if success_t1w or success_dwi:
            logger.info(f"Subject {subject_id} conversion completed")
            return True
        else:
            logger.warning(f"No data converted for subject {subject_id}")
            return False
    
    def convert_all_subjects(self, subjects: List[str] = None):
        """Convert all subjects or a specified subset."""
        if subjects is None:
            subjects = self.get_subject_ids()
        
        logger.info(f"Starting conversion of {len(subjects)} subjects")
        
        # Create BIDS structure
        self.create_bids_structure()
        
        # Convert each subject
        successful_conversions = 0
        for subject_id in subjects:
            try:
                if self.convert_subject(subject_id):
                    successful_conversions += 1
            except Exception as e:
                logger.error(f"Error converting subject {subject_id}: {e}")
        
        logger.info(f"Conversion completed. {successful_conversions}/{len(subjects)} subjects converted successfully")
        return successful_conversions

def main():
    """Main function to run the conversion."""
    parser = argparse.ArgumentParser(description="Convert HCP_1200 dataset to BIDS format")
    parser.add_argument("hcp_root", help="Path to HCP_1200 dataset root directory")
    parser.add_argument("bids_root", help="Path to output BIDS dataset directory")
    parser.add_argument("--subjects", nargs="+", help="Specific subject IDs to convert (default: all)")
    parser.add_argument("--test", action="store_true", help="Run conversion on first subject only")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = HCPToBIDSConverter(args.hcp_root, args.bids_root)
    
    if args.test:
        # Test with first subject
        subjects = converter.get_subject_ids()[:1]
        logger.info(f"Running test conversion on subject: {subjects[0]}")
        converter.convert_all_subjects(subjects)
    elif args.subjects:
        # Convert specified subjects
        converter.convert_all_subjects(args.subjects)
    else:
        # Convert all subjects
        converter.convert_all_subjects()

if __name__ == "__main__":
    main() 