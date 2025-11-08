#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seq2smi_cycle
=============

Convert a cyclic peptide sequence to SMILES by reusing the linear seq2smi
converter and then forming a head-to-tail peptide bond.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from rdkit import Chem

from seq2smi import MonomerLib, seq2smi, _tokenize_preserve_brackets, _strip_brackets
from utils import get_backbone_atoms

DEFAULT_LIB_PATH = Path("data/monomersFromHELMCoreLibrary.json")


def _init_lib(lib: MonomerLib | None) -> MonomerLib:
    if lib is not None:
        return lib
    if DEFAULT_LIB_PATH.exists():
        return MonomerLib(str(DEFAULT_LIB_PATH))
    return MonomerLib()


def _ensure_no_terminal_caps(seq: str) -> None:
    tokens = _tokenize_preserve_brackets(seq)
    if not tokens:
        raise ValueError("Sequence is empty.")
    if _strip_brackets(tokens[0]).lower() in {"ac", "formyl"}:
        raise ValueError("Cyclic peptides must not include N-terminal caps.")
    if _strip_brackets(tokens[-1]).lower() in {"am", "ome"}:
        raise ValueError("Cyclic peptides must not include C-terminal caps.")


def _remove_terminal_groups(rw: Chem.RWMol, n_idx: int, c_idx: int) -> None:
    """Strip the explicit [H] and HO groups that represent linear termini."""
    # Remove N-terminal hydrogens
    n_atom = rw.GetAtomWithIdx(n_idx)
    n_hydrogens = [
        nb.GetIdx()
        for nb in n_atom.GetNeighbors()
        if nb.GetAtomicNum() == 1
    ]
    for h_idx in sorted(n_hydrogens, reverse=True):
        rw.RemoveAtom(h_idx)

    # Remove C-terminal hydroxyl (oxygen + attached hydrogens)
    c_atom = rw.GetAtomWithIdx(c_idx)
    for nb in c_atom.GetNeighbors():
        if nb.GetAtomicNum() != 8:
            continue
        bond = rw.GetBondBetweenAtoms(c_idx, nb.GetIdx())
        if bond and bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            o_idx = nb.GetIdx()
            o_atom = rw.GetAtomWithIdx(o_idx)
            h_neighbors = [
                h.GetIdx() for h in o_atom.GetNeighbors() if h.GetAtomicNum() == 1
            ]
            for h_idx in sorted(h_neighbors, reverse=True):
                rw.RemoveAtom(h_idx)
            rw.RemoveAtom(o_idx)
            break


def seq2smi_cycle(sequence: str, lib: MonomerLib | None = None) -> str:
    _ensure_no_terminal_caps(sequence)
    lib = _init_lib(lib)
    linear_smiles = seq2smi(sequence, lib)
    mol = Chem.MolFromSmiles(linear_smiles)
    if mol is None:
        raise ValueError("Failed to parse intermediate linear SMILES.")

    backbone = get_backbone_atoms(mol)
    if len(backbone) < 2:
        raise ValueError("Unable to detect peptide backbone.")
    n_idx = backbone[0][0]
    c_idx = backbone[-1][2]

    rw = Chem.RWMol(mol)
    _remove_terminal_groups(rw, n_idx, c_idx)
    tmp = rw.GetMol()
    Chem.SanitizeMol(tmp)
    backbone2 = get_backbone_atoms(tmp)
    if len(backbone2) < 2:
        raise ValueError("Backbone collapsed while preparing cyclic structure.")
    n_idx2 = backbone2[0][0]
    c_idx2 = backbone2[-1][2]
    rw2 = Chem.RWMol(tmp)
    rw2.AddBond(c_idx2, n_idx2, Chem.rdchem.BondType.SINGLE)
    cyc = rw2.GetMol()
    Chem.SanitizeMol(cyc)
    return Chem.MolToSmiles(cyc, isomericSmiles=True)


def sequences_to_smiles(seqs: Iterable[str], lib: MonomerLib | None = None) -> dict:
    lib = _init_lib(lib)
    return {seq: seq2smi_cycle(seq, lib) for seq in seqs}


def _load_sequences(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cyclic peptide sequence → SMILES converter")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("seq2smi_cycle_input.txt"),
        help="Input file containing one cyclic sequence per line.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("seq2smi_cycle_out.txt"),
        help="Destination file for SMILES results.",
    )
    args = parser.parse_args()

    sequences = _load_sequences(args.input)
    lib = _init_lib(None)
    results = sequences_to_smiles(sequences, lib)
    args.output.write_text(
        "\n".join(f"{seq}\t{smiles}" for seq, smiles in results.items()),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
