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
from typing import List, Dict, Tuple, Optional
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
        
        logger.info(f"Found {len(dwi_runs)} DWI runs: {[f.name for f in dwi_runs]}")
        
        if not dwi_runs:
            logger.warning(f"No DWI runs found for subject {subject_id}")
            return False
        
        # Group runs by phase encoding direction (LR vs RL)
        lr_runs = [run for run in dwi_runs if "_LR.nii.gz" in run.name]
        rl_runs = [run for run in dwi_runs if "_RL.nii.gz" in run.name]
        
<<<<<<< HEAD
        # Group runs by phase encoding (LR vs RL)
        dwi_groups = {}
        for dwi_run in dwi_runs:
            # Extract phase encoding from filename (e.g., LR, RL)
            # Filename format: 149741_3T_DWI_dir95_LR.nii.gz
            parts = dwi_run.stem.split('_')
            phase_encoding = None
            
            # Look for LR or RL in the filename parts
            for part in parts:
                if part in ['LR', 'RL']:
                    phase_encoding = part
                    break
            
            # If not found in parts, check the full filename
            if phase_encoding is None:
                if '_LR.' in dwi_run.name:
                    phase_encoding = 'LR'
                elif '_RL.' in dwi_run.name:
                    phase_encoding = 'RL'
            
            logger.info(f"File: {dwi_run.name}, Extracted phase encoding: {phase_encoding}")
            
            if phase_encoding:
                if phase_encoding not in dwi_groups:
                    dwi_groups[phase_encoding] = []
                dwi_groups[phase_encoding].append(dwi_run)
            else:
                logger.warning(f"Could not extract phase encoding from filename: {dwi_run.name}")
        
        logger.info(f"Grouped DWI runs by phase encoding: {dwi_groups}")
        
        # Merge each phase encoding group separately
        for phase_encoding, runs in dwi_groups.items():
            logger.info(f"Processing phase encoding {phase_encoding} with {len(runs)} runs")
            self._merge_dwi_runs_by_phase_encoding(subject_id, phase_encoding, runs, subject_dir)
        
        # Handle SBRef images
        self._convert_sbref_images(subject_id, dwi_dir, subject_dir)
=======
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
>>>>>>> origin/main
        
        logger.info(f"DWI conversion completed for subject {subject_id}")
        return True
    
<<<<<<< HEAD
    def _merge_dwi_runs_by_phase_encoding(self, subject_id: str, phase_encoding: str, dwi_runs: List[Path], dwi_dir: Path):
        """Merge DWI runs for a specific phase encoding (all directions together)."""
        logger.info(f"Merging {len(dwi_runs)} DWI runs for phase encoding {phase_encoding}")
=======
    def _merge_dwi_runs(self, subject_id: str, dwi_runs: List[Path], dwi_dir: Path, phase_encoding: str):
        """Merge multiple DWI runs into a single 4D volume for a specific phase encoding direction."""
        logger.info(f"Merging {len(dwi_runs)} {phase_encoding} DWI runs for subject {subject_id}")
>>>>>>> origin/main
        
        # Load and concatenate DWI data
        dwi_data_list = []
        bval_list = []
        bvec_list = []
        
        for dwi_run in dwi_runs:
            # Load DWI data
            dwi_img = nib.load(dwi_run)
<<<<<<< HEAD
            dwi_data = dwi_img.get_fdata()
            
            # Check if data is 3D or 4D and handle accordingly
            if dwi_data.ndim == 3:
                # Add a 4th dimension if it's 3D
                dwi_data = dwi_data[..., np.newaxis]
                logger.info(f"Added 4th dimension to 3D data: {dwi_run.name}")
            
            dwi_data_list.append(dwi_data)
=======
            dwi_data_list.append(dwi_img.get_fdata())  # type: ignore
>>>>>>> origin/main
            
            # Load corresponding bval and bvec files
            run_base = dwi_run.stem.replace('.nii', '')
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
            raise ValueError(f"No DWI data loaded for phase encoding {phase_encoding}")
        
        if not bval_list or not bvec_list:
            raise ValueError(f"No bval/bvec data loaded for phase encoding {phase_encoding}")
        
        # Concatenate data along time dimension (axis 3)
        merged_dwi = np.concatenate(dwi_data_list, axis=3)
        merged_bval = np.concatenate(bval_list)
        merged_bvec = np.concatenate(bvec_list, axis=1)
        
<<<<<<< HEAD
        # Save merged DWI data
        merged_img = nib.Nifti1Image(merged_dwi, dwi_img.affine, dwi_img.header)
        dwi_dest = dwi_dir / f"sub-{subject_id}_dwi_{phase_encoding}.nii.gz"
        nib.save(merged_img, dwi_dest)
        
        # Save merged bval and bvec
        bval_dest = dwi_dir / f"sub-{subject_id}_dwi_{phase_encoding}.bval"
        bvec_dest = dwi_dir / f"sub-{subject_id}_dwi_{phase_encoding}.bvec"
=======
        # Save merged DWI data with phase encoding direction in filename
        merged_img = nib.Nifti1Image(merged_dwi, dwi_img.affine, dwi_img.header)  # type: ignore
        dwi_dest = dwi_dir / f"sub-{subject_id}_dir-{phase_encoding}_dwi.nii.gz"
        nib.save(merged_img, dwi_dest)
        
        # Save merged bval and bvec
        bval_dest = dwi_dir / f"sub-{subject_id}_dir-{phase_encoding}_dwi.bval"
        bvec_dest = dwi_dir / f"sub-{subject_id}_dir-{phase_encoding}_dwi.bvec"
>>>>>>> origin/main
        
        np.savetxt(bval_dest, merged_bval, fmt='%.0f')
        np.savetxt(bvec_dest, merged_bvec, fmt='%.6f')
        
<<<<<<< HEAD
        logger.info(f"Saved merged DWI for phase encoding {phase_encoding}")
    
    def _convert_sbref_images(self, subject_id: str, dwi_source_dir: Path, dwi_dir: Path):
        """Convert SBRef (single-band reference) images by merging LR and RL separately."""
        # Find SBRef images
        sbref_files = list(dwi_source_dir.glob(f"{subject_id}_3T_DWI_dir*_SBRef.nii.gz"))
        
        if not sbref_files:
            logger.info("No SBRef files found")
            return
        
        # Group SBRef files by phase encoding
        sbref_groups = {}
        for sbref_file in sbref_files:
            # Extract phase encoding from filename
            parts = sbref_file.stem.split('_')
            phase_encoding = None
            for part in parts:
                if part in ['LR', 'RL']:
                    phase_encoding = part
                    break
            
            if phase_encoding:
                if phase_encoding not in sbref_groups:
                    sbref_groups[phase_encoding] = []
                sbref_groups[phase_encoding].append(sbref_file)
        
        # Merge SBRef images for each phase encoding
        for phase_encoding, sbref_runs in sbref_groups.items():
            logger.info(f"Merging {len(sbref_runs)} SBRef runs for phase encoding {phase_encoding}")
            
            # Load and concatenate SBRef data
            sbref_data_list = []
            for sbref_run in sbref_runs:
                sbref_img = nib.load(sbref_run)
                sbref_data = sbref_img.get_fdata()
                
                # Check if data is 3D or 4D and handle accordingly
                if sbref_data.ndim == 3:
                    # Add a 4th dimension if it's 3D
                    sbref_data = sbref_data[..., np.newaxis]
                    logger.info(f"Added 4th dimension to 3D SBRef data: {sbref_run.name}")
                
                sbref_data_list.append(sbref_data)
            
            if sbref_data_list:
                # Concatenate along time dimension (axis 3)
                merged_sbref = np.concatenate(sbref_data_list, axis=3)
                
                # Save merged SBRef
                merged_img = nib.Nifti1Image(merged_sbref, sbref_img.affine, sbref_img.header)
                sbref_dest = dwi_dir / f"sub-{subject_id}_dwi_{phase_encoding}_sbref.nii.gz"
                nib.save(merged_img, sbref_dest)
                
                logger.info(f"Saved merged SBRef for phase encoding {phase_encoding}")
=======
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
>>>>>>> origin/main
    
    def convert_subject(self, subject_id: str):
        """Convert all data for a single subject."""
        logger.info(f"Converting subject {subject_id}")
        
        success_dwi = self.convert_dwi_data(subject_id)
        
        if success_dwi:
            logger.info(f"Subject {subject_id} conversion completed")
            return True
        else:
            logger.warning(f"No data converted for subject {subject_id}")
            return False
    
    def convert_all_subjects(self, subjects: Optional[List[str]] = None):
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