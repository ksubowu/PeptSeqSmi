from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rdkit import Chem

from seq2smi import MonomerLib, seq2smi
from utils import get_backbone_atoms
from smi2seq import SMILES2Sequence


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_LIBRARY = DATA_DIR / "monomersFromHELMCoreLibrary.json"
EXTEND_LIBRARY = BASE_DIR / "extend_lib.json"
EXTEND_LIBRARY_CUSTOM = BASE_DIR / "extend_lib_custom.json"


class SequencePayload(BaseModel):
    sequence: str = Field(..., description="Peptide sequence such as ac-A-W-am.")


class SmilesPayload(BaseModel):
    smiles: str = Field(..., description="Peptide SMILES string.")


class TemplatePayload(BaseModel):
    code: str = Field(..., description="Template identifier, e.g. LyS_Custom.")
    smiles: str = Field(..., description="Isomeric SMILES for the monomer.")
    polymer_type: str = Field("PEPTIDE", description="Polymer type for the template.")
    aliases: Optional[List[str]] = Field(default=None, description="Optional alias list.")
    components: Optional[List[str]] = Field(default=None, description="Component codes.")


def _read_library_file(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read {path.name}: {exc}") from exc
    if isinstance(data, dict):
        data = list(data.values())
    if not isinstance(data, list):
        return []
    return data


def _load_extend_library() -> List[dict]:
    combined: dict[str, dict] = {}
    for entry in _read_library_file(EXTEND_LIBRARY) + _read_library_file(EXTEND_LIBRARY_CUSTOM):
        code = entry.get("code")
        if not code:
            continue
        combined[code] = entry
    return list(combined.values())


def _save_extend_library(entries: List[dict]) -> None:
    EXTEND_LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    with EXTEND_LIBRARY.open("w", encoding="utf-8") as handle:
        json.dump(sorted(entries, key=lambda item: item.get("code", "")), handle, indent=2, ensure_ascii=False)


def _append_custom_template(entry: dict) -> None:
    entries = _read_library_file(EXTEND_LIBRARY_CUSTOM)
    codes = {item.get("code") for item in entries}
    if entry.get("code") in codes:
        return
    entries.append(entry)
    EXTEND_LIBRARY_CUSTOM.parent.mkdir(parents=True, exist_ok=True)
    with EXTEND_LIBRARY_CUSTOM.open("w", encoding="utf-8") as handle:
        json.dump(sorted(entries, key=lambda item: item.get("code", "")), handle, indent=2, ensure_ascii=False)


def _to_helm_corelib_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Provided SMILES cannot be parsed.")
    Chem.SanitizeMol(mol)
    if any(atom.GetAtomMapNum() > 0 for atom in mol.GetAtoms()):
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    matches = list(get_backbone_atoms(mol))
    if not matches:
        raise HTTPException(
            status_code=400,
            detail="Failed to locate peptide backbone (N-CA-C) in the supplied SMILES.",
        )
    n_idx, ca_idx, c_idx = matches[0]
    rw = Chem.RWMol(mol)

    # ensure N has an explicit mapped hydrogen [H:1]
    n_atom = rw.GetAtomWithIdx(n_idx)
    h_idx = None
    for nb in n_atom.GetNeighbors():
        if nb.GetAtomicNum() == 1:
            h_idx = nb.GetIdx()
            break
    if h_idx is None:
        h_atom = Chem.Atom(1)
        h_atom.SetNoImplicit(True)
        h_atom.SetFormalCharge(0)
        h_atom.SetAtomMapNum(1)
        h_idx = rw.AddAtom(h_atom)
        rw.AddBond(n_idx, h_idx, Chem.BondType.SINGLE)
    else:
        h_atom = rw.GetAtomWithIdx(h_idx)
        h_atom.SetAtomMapNum(1)
        h_atom.SetNoImplicit(True)
        h_atom.SetFormalCharge(0)

    # ensure carbonyl carbon has a hydroxyl [OH:2]
    carbon = rw.GetAtomWithIdx(c_idx)
    o_idx = None
    for bond in carbon.GetBonds():
        other = bond.GetOtherAtom(carbon)
        if bond.GetBondType() == Chem.BondType.SINGLE and other.GetAtomicNum() == 8:
            o_idx = other.GetIdx()
            break
    if o_idx is None:
        o_atom = Chem.Atom(8)
        o_atom.SetFormalCharge(0)
        o_idx = rw.AddAtom(o_atom)
        rw.AddBond(c_idx, o_idx, Chem.BondType.SINGLE)
    o_atom = rw.GetAtomWithIdx(o_idx)
    o_atom.SetFormalCharge(0)
    o_atom.SetAtomMapNum(2)
    o_atom.SetNoImplicit(True)
    o_atom.SetNumExplicitHs(1)

    helm_mol = rw.GetMol()
    Chem.SanitizeMol(helm_mol)
    return Chem.MolToSmiles(helm_mol, isomericSmiles=True, canonical=False)


app = FastAPI(title="Peptide Converter", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


_library_lock = threading.Lock()
_monomer_library = MonomerLib(str(DEFAULT_LIBRARY))
_seq_converter = SMILES2Sequence(lib_path=str(DEFAULT_LIBRARY))


@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/seq2smi")
def api_seq2smi(payload: SequencePayload) -> JSONResponse:
    sequence = payload.sequence.strip()
    if not sequence:
        raise HTTPException(status_code=400, detail="Sequence cannot be empty.")
    with _library_lock:
        try:
            smiles = seq2smi(sequence, _monomer_library)
        except Exception as exc:  # user input error
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"sequence": sequence, "smiles": smiles})


@app.post("/api/smi2seq")
def api_smi2seq(payload: SmilesPayload) -> JSONResponse:
    smiles = payload.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES cannot be empty.")
    try:
        sequence, details = _seq_converter.convert(smiles, return_details=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"smiles": smiles, "sequence": sequence, "details": details})


@app.post("/api/templates")
def api_add_template(payload: TemplatePayload) -> JSONResponse:
    code = payload.code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="Template code cannot be empty.")
    smiles = payload.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="Template SMILES cannot be empty.")
    helm_smiles = _to_helm_corelib_smiles(smiles)
    molecule = Chem.MolFromSmiles(helm_smiles,sanitize=False)#othervise [H:1] droped
    if molecule is None:
        raise HTTPException(status_code=400, detail="Failed to normalize SMILES into HELM format.")
    Chem.SanitizeMol(molecule)

    entry = {
        "code": code,
        "smiles": Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=False),
        "polymer_type": payload.polymer_type or "PEPTIDE",
        "type": (payload.polymer_type or "PEPTIDE"),
    }
    if payload.aliases:
        entry["aliases"] = payload.aliases
    if payload.components:
        entry["components"] = payload.components

    with _library_lock:
        extended = _load_extend_library()
        existing_codes = {item.get("code") for item in extended}
        if code in existing_codes:
            raise HTTPException(status_code=400, detail=f"Template '{code}' already exists.")
        extended.append(entry)
        _save_extend_library(extended)
        _append_custom_template(entry)

        # update in-memory libraries immediately
        _monomer_library.by_code[code.lower()] = entry
        _monomer_library.alias[code.lower()] = code
        if payload.aliases:
            for alias in payload.aliases:
                _monomer_library.alias[alias.lower()] = code
        _seq_converter._register_template(entry, allow_overwrite=True)  # type: ignore[attr-defined]

    return JSONResponse({"message": f"Template '{code}' registered.", "entry": entry})


@app.get("/api/templates")
def api_list_templates() -> JSONResponse:
    built_in = sorted(_monomer_library.by_code.keys())
    extended_entries = _load_extend_library()
    extended_codes = sorted({item.get("code") for item in extended_entries if item.get("code")})
    custom_codes = sorted({item.get("code") for item in _read_library_file(EXTEND_LIBRARY_CUSTOM) if item.get("code")})
    return JSONResponse({"library": built_in, "extended": extended_codes, "custom": custom_codes})
