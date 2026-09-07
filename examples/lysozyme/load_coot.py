#!/usr/bin/env python
"""Coot script to load 1IEE lysozyme refined model and all synthesized maps:
  1. Refined PDB coordinates (1iee_out_model.pdb)
  2. 2mFo - DFc map (1iee_out_2fofc.ccp4) at 1.5 sigma
  3. mFo - DFc difference map (1iee_out_fofc.ccp4) at +/- 3.0 sigma
  4. Weighted intensity target gradient map (1iee_out_gradient.ccp4) at +/- 3.0 sigma

Usage:
  coot --script examples/lysozyme/load_coot.py
Or inside Coot:
  Calculate -> Run Script -> select load_coot.py
"""

from pathlib import Path
import os

HERE = Path(__file__).resolve().parent

PDB_FILE = str(HERE / "1iee_out_model.pdb")
TWOFOFC_FILE = str(HERE / "1iee_out_2fofc.ccp4")
FOFC_FILE = str(HERE / "1iee_out_fofc.ccp4")
GRAD_FILE = str(HERE / "1iee_out_gradient.ccp4")
MTZ_FILE = str(HERE / "1iee_out_maps.mtz")

print(f"Loading model into Coot: {PDB_FILE}")
imol = read_pdb(PDB_FILE)

print(f"Loading 2mFo-DFc map: {TWOFOFC_FILE}")
imol_2fofc = handle_read_ccp4_map(TWOFOFC_FILE, 0)
set_contour_level_in_sigma(imol_2fofc, 1.5)

print(f"Loading mFo-DFc difference map: {FOFC_FILE}")
imol_fofc = handle_read_ccp4_map(FOFC_FILE, 1)
set_contour_level_in_sigma(imol_fofc, 3.0)

print(f"Loading weighted intensity gradient map: {GRAD_FILE}")
imol_grad = handle_read_ccp4_map(GRAD_FILE, 1)
set_contour_level_in_sigma(imol_grad, 3.0)

print("All maps and model loaded successfully into Coot!")
