#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smi2seq_cycle
=============

Convert a cyclic peptide SMILES string back to a sequence representation.
The converter breaks a single peptide bond to linearise the molecule and
then reuses the SMILES2Sequence pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from rdkit import Chem

from smi2seq import SMILES2Sequence
from utils import clean_smiles

DEFAULT_LIB_PATH = Path("data/monomersFromHELMCoreLibrary.json")


def _init_converter(converter: SMILES2Sequence | None = None) -> SMILES2Sequence:
    if converter is not None:
        return converter
    if DEFAULT_LIB_PATH.exists():
        return SMILES2Sequence(DEFAULT_LIB_PATH)
    return SMILES2Sequence()


def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    if atom.GetAtomicNum() != 6:
        return False
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() == 8 and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            return True
    return False


def _find_peptide_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    bonds: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetAtomicNum() == 6 and b.GetAtomicNum() == 7:
            if _is_carbonyl_carbon(a):
                bonds.append((a.GetIdx(), b.GetIdx()))
        elif a.GetAtomicNum() == 7 and b.GetAtomicNum() == 6:
            if _is_carbonyl_carbon(b):
                bonds.append((b.GetIdx(), a.GetIdx()))
    return bonds


def _linearise_cycle(smiles: str) -> Tuple[str, dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    peptide_bonds = _find_peptide_bonds(mol)
    if len(peptide_bonds) < 2:
        raise ValueError("Unable to locate sufficient peptide bonds for a cyclic peptide.")

    # Choose a deterministic bond to break (highest carbon index)
    c_idx, n_idx = max(peptide_bonds, key=lambda pair: (pair[0], pair[1]))
    rw = Chem.RWMol(mol)
    if rw.GetBondBetweenAtoms(c_idx, n_idx) is None:
        raise ValueError("Peptide bond lookup failed during cyclisation break.")

    rw.RemoveBond(c_idx, n_idx)
    # Recreate terminal amine (N-H) and carboxylate (C-OH) to mimic linear peptide ends
    n_h_idx = rw.AddAtom(Chem.Atom(1))
    rw.AddBond(n_idx, n_h_idx, Chem.rdchem.BondType.SINGLE)
    o_idx = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(c_idx, o_idx, Chem.rdchem.BondType.SINGLE)
    o_h_idx = rw.AddAtom(Chem.Atom(1))
    rw.AddBond(o_idx, o_h_idx, Chem.rdchem.BondType.SINGLE)
    Chem.SanitizeMol(rw)
    linear = Chem.MolToSmiles(rw, isomericSmiles=True)
    return linear, {"break_c_idx": int(c_idx), "break_n_idx": int(n_idx)}


def smi2seq_cycle(
    smiles: str,
    converter: SMILES2Sequence | None = None,
) -> Tuple[str, dict]:
    converter = _init_converter(converter)
    cleaned = clean_smiles(smiles)
    linear_smiles, meta = _linearise_cycle(cleaned)
    sequence, details = converter.convert(linear_smiles, return_details=True)
    meta.update({"was_cyclic": True})
    if isinstance(details, dict):
        merged = dict(details)
        merged["cycle_meta"] = meta
    else:
        merged = {"details": details, "cycle_meta": meta}
    return sequence, merged


def _load_smiles(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cyclic peptide SMILES → sequence converter")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("smi2seq_cycle_input.txt"),
        help="Input file with one SMILES per line.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("smi2seq_cycle_out.txt"),
        help="File to write sequences + metadata (tab separated).",
    )
    args = parser.parse_args()

    smiles_list = _load_smiles(args.input)
    converter = _init_converter()
    rows: List[str] = []
    for smi in smiles_list:
        seq, info = smi2seq_cycle(smi, converter)
        rows.append(f"{smi}\t{seq}\t{info}")
    args.output.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
