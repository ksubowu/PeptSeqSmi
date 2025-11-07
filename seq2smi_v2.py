#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert peptide sequences to SMILES representation.
Supports:
- Standard amino acids (both 1 and 3 letter codes)
- N-terminal caps (ac, formyl, etc.)
- C-terminal caps (am, etc.)
- D-amino acids (prefixed with 'd' or 'D')
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import re
from collections import namedtuple, deque
from rdkit import Chem
from rdkit.Chem import rdmolops
from utils import get_backbone_atoms

# Basic amino acid mappings
VOCAB_SEED_DEFAULT = {
    "A": "NC(C)C(=O)O",
    "R": "NC(CCCNC(N)=N)C(=O)O",
    "N": "NCC(=O)NC(=O)O",
    "D": "NCC(=O)OC(=O)O",
    "C": "NCC(S)C(=O)O",
    "E": "NCCC(=O)OC(=O)O",
    "Q": "NCCC(=O)NC(=O)O",
    "G": "NCC(=O)O",
    "H": "NC(CC1=CN=CN1)C(=O)O",
    "I": "NC(C(CC)C)C(=O)O",
    "L": "NC(CC(C)C)C(=O)O",
    "K": "NC(CCCCN)C(=O)O",
    "M": "NCC(SCC)C(=O)O",
    "F": "NC(CC1=CC=CC=C1)C(=O)O",
    "P": "N1CCCC1C(=O)O",
    "S": "NCC(O)C(=O)O",
    "T": "NC(CO)C(=O)O",
    "W": "NC(CC1=CNC2=CC=CC=C12)C(=O)O",
    "Y": "NC(C1=CC=CC=C1O)C(=O)O",
    "V": "NC(C(C)C)C(=O)O",
}

# Common terminal caps
N_CAPS = {
    "ac": "C(C)=O",      # Acetyl
    "formyl": "C=O",     # Formyl
}

C_CAPS = {
    "am": "N",           # Amide (-CONH2)
    "ome": "OC",         # Methyl ester
}
DEFAULT_N_CAP = {"symbol": "H", "smiles": "[H]"}   # “无额外封端”的占位
DEFAULT_C_CAP = {"symbol": "H", "smiles": "[H]"}   # 同上

# 1-letter to 3-letter conversion
AA1TO3 = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr"
}

# 3-letter to 1-letter conversion
AA3TO1 = {v: k for k, v in AA1TO3.items()}

ParsedSeq = namedtuple("ParsedSeq", ["residues", "n_cap", "c_cap", "staples", "cyclo"])

class MonomerLib:
    """Manages the monomer library including standard and custom amino acids."""
    def __init__(self, json_path: Optional[str] = None):
        self.by_code = {}
        self.alias = {}
        
        # Load built-in amino acids
        for code, smiles in VOCAB_SEED_DEFAULT.items():
            self.by_code[code.lower()] = {
                "code": code,
                "smiles": smiles,
                "type": "PEPTIDE"
            }
            # Add both 1-letter and 3-letter codes as aliases
            self.alias[code.lower()] = code
            if code in AA1TO3:
                three_letter = AA1TO3[code]
                self.alias[three_letter.lower()] = code
        
        # Load custom monomers from JSON if provided
        if json_path:
            self.load_json(json_path)

    def load_json(self, json_path: str):
        """Load additional monomers from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            code = item["code"]
            self.by_code[code.lower()] = item
            self.alias[code.lower()] = code
            for alias in item.get("aliases", []):
                self.alias[alias.lower()] = code

    def resolve_code(self, token: str) -> Optional[str]:
        """Resolve monomer code from token, handling both 1 and 3 letter codes."""
        return self.alias.get(token.lower())

    def get(self, code: str) -> dict:
        """Get monomer entry by code."""
        return self.by_code[code.lower()]


def _parse_pose_line(line: str) -> Dict[int, Optional[str]]:
    """Parse pose line like 'pose:1 S,6 S,10 S'."""
    if not line:
        raise ValueError("Staple pose line is empty.")
    prefix, _, payload = line.partition(":")
    if payload == "":
        raise ValueError("Staple pose line must contain ':' separating entries.")
    if prefix.strip().lower() != "pose":
        raise ValueError("Staple pose line must start with 'pose:'.")
    entries = [item.strip() for item in payload.split(",") if item.strip()]
    if not entries:
        raise ValueError("Staple pose line contains no entries.")
    result: Dict[int, Optional[str]] = {}
    for entry in entries:
        parts = entry.replace("-", " ").replace("/", " ").split()
        if not parts:
            continue
        try:
            position = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid residue index in staple pose entry '{entry}'.") from exc
        prefer_atom = None
        if len(parts) > 1:
            prefer_atom = parts[1].strip()
            if prefer_atom:
                prefer_atom = prefer_atom.upper()
        if position in result:
            raise ValueError(f"Duplicate residue index {position} in staple pose.")
        result[position] = prefer_atom
    return result


def _collect_sidechain_atoms(
    mol: Chem.Mol, ca_idx: int, backbone: Set[int]
) -> Set[int]:
    """BFS from Cα collecting atoms not belonging to the backbone."""
    side_atoms: Set[int] = set()
    queue: List[int] = [ca_idx]
    visited: Set[int] = {ca_idx}
    while queue:
        current = queue.pop()
        atom = mol.GetAtomWithIdx(current)
        for nb in atom.GetNeighbors():
            idx = nb.GetIdx()
            if idx in visited:
                continue
            visited.add(idx)
            if idx in backbone and idx != ca_idx:
                continue
            side_atoms.add(idx)
            queue.append(idx)
    return side_atoms


def _farthest_atom(
    mol: Chem.Mol, start_idx: int, candidates: List[int]
) -> int:
    """Return the candidate atom farthest (topological distance) from start."""
    target_set = set(candidates)
    best_distance = -1
    best_idx = candidates[0]
    queue: deque[Tuple[int, int]] = deque()
    queue.append((start_idx, 0))
    visited = {start_idx}
    while queue:
        idx, dist = queue.popleft()
        if idx in target_set and idx != start_idx:
            if dist > best_distance or (dist == best_distance and idx < best_idx):
                best_distance = dist
                best_idx = idx
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in visited:
                continue
            visited.add(nb_idx)
            queue.append((nb_idx, dist + 1))
    return best_idx


def _determine_side_anchor(
    fragment: Chem.Mol, prefer_element: Optional[str], residue_label: str, position: int
) -> Tuple[Optional[int], Optional[str]]:
    """Return the atom index in fragment used for linker attachment."""
    prefer_element = (prefer_element or "").upper() or None
    matches = list(get_backbone_atoms(fragment))
    if not matches:
        return None, f"残基 {position} ({residue_label}) 缺少 backbone 匹配，无法定位装订位点。"
    n_idx, ca_idx, c_idx = matches[0]
    backbone_atoms: Set[int] = {n_idx, ca_idx, c_idx}
    carbon = fragment.GetAtomWithIdx(c_idx)
    for bond in carbon.GetBonds():
        other = bond.GetOtherAtom(carbon)
        if other.GetAtomicNum() == 8:
            backbone_atoms.add(other.GetIdx())
    for atom in fragment.GetAtoms():
        if atom.GetAtomMapNum() > 0:
            backbone_atoms.add(atom.GetIdx())
    side_atoms = _collect_sidechain_atoms(fragment, ca_idx, backbone_atoms)
    heavy_atoms = [
        idx
        for idx in side_atoms
        if fragment.GetAtomWithIdx(idx).GetAtomicNum() > 1
    ]
    if not heavy_atoms:
        return None, f"残基 {position} ({residue_label}) 侧链不含重原子，无法装订。"
    candidates = heavy_atoms
    note = None
    note: Optional[str] = None
    prefer_matches = [
        idx
        for idx in heavy_atoms
        if fragment.GetAtomWithIdx(idx).GetSymbol().upper() == prefer_element
    ] if prefer_element else []
    if prefer_element and not prefer_matches:
        fallback_symbol = fragment.GetAtomWithIdx(heavy_atoms[0]).GetSymbol()
        note = (
            f"残基 {position} ({residue_label}) 侧链不含元素 {prefer_element}，"
            f"已改用默认原子 {fallback_symbol}."
        )

    candidates = prefer_matches if prefer_matches else heavy_atoms
    best_tuple = None
    anchor_idx = candidates[0]
    for idx in candidates:
        atom = fragment.GetAtomWithIdx(idx)
        degree = atom.GetDegree()
        path = rdmolops.GetShortestPath(fragment, ca_idx, idx)
        distance = len(path) - 1 if path else 0
        priority = (degree, -distance, idx)
        if best_tuple is None or priority < best_tuple:
            best_tuple = priority
            anchor_idx = idx
    return anchor_idx, note


def _attach_linker_to_peptide(
    peptide: Chem.Mol,
    residue_side_anchors: List[Optional[int]],
    staple_prefs: Dict[int, Optional[str]],
    linker_smiles: str,
) -> Tuple[Chem.Mol, List[int]]:
    linker = Chem.MolFromSmiles(linker_smiles)
    if linker is None:
        print(f'check the linker smiles@@@@@:\n{linker_smiles}')
        raise ValueError("无法解析装订肽链接体的 SMILES。")
    Chem.SanitizeMol(linker)

    map_to_idx: Dict[int, int] = {}
    for atom in linker.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap > 0:
            if amap in map_to_idx:
                raise ValueError(f"链接体中存在重复的占位符 [*: {amap}].")
            map_to_idx[amap] = atom.GetIdx()

    missing = set(staple_prefs.keys()) - set(map_to_idx.keys())
    if missing:
        missing_str = ", ".join(str(x) for x in sorted(missing))
        raise ValueError(f"链接体 SMILES 缺少占位符 [*:i] 对应残基位置: {missing_str}")

    combo = Chem.CombineMols(peptide, linker)
    rw = Chem.RWMol(combo)
    offset = peptide.GetNumAtoms()

    placeholders_to_remove: List[int] = []
    hydrogens_to_remove: List[int] = []

    for position, _ in sorted(staple_prefs.items()):
        if position < 1 or position > len(residue_side_anchors):
            raise ValueError(f"装订位置 {position} 超出残基数量范围。")
        anchor_idx = residue_side_anchors[position - 1]
        if anchor_idx is None:
            raise ValueError(f"残基 {position} 无法确定可用于装订的侧链锚点。")
        placeholder_local = map_to_idx[position]
        placeholder_global = placeholder_local + offset
        placeholder_atom = rw.GetAtomWithIdx(placeholder_global)
        neighbors = [nb.GetIdx() for nb in placeholder_atom.GetNeighbors()]
        if len(neighbors) != 1:
            raise ValueError(f"链接体占位符 [*: {position}] 连接的原子数量异常。")
        neighbor_idx = neighbors[0]

        anchor_atom = rw.GetAtomWithIdx(anchor_idx)
        anchor_atom.SetNoImplicit(True)
        anchor_atom.SetFormalCharge(0)
        anchor_atom.SetNumExplicitHs(0)
        h_neighbors = [nb.GetIdx() for nb in anchor_atom.GetNeighbors() if nb.GetAtomicNum() == 1]
        if h_neighbors:
            hydrogens_to_remove.append(h_neighbors[0])

        if anchor_idx == neighbor_idx:
            continue

        rw.AddBond(anchor_idx, neighbor_idx, Chem.rdchem.BondType.SINGLE)
        placeholders_to_remove.append(placeholder_global)

    for idx in sorted(placeholders_to_remove, reverse=True):
        rw.RemoveAtom(idx)

    final_mol = rw.GetMol()
    for atom in final_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return final_mol, hydrogens_to_remove


def anchors_and_leaving_from_helm(mol: Chem.Mol) -> Tuple[Dict[int, int], List[int]]:
    """
    从 HELM corelib 风格单体提取：
      anchors: {map_num -> anchor_heavy_atom_idx}
      leaving_atoms: [atom_idx, ...]  # 连接完成后应删除的“离去原子”（如 H 或 OH 的 O）
    规则：
      - 找到所有 AtomMapNum>0 的“带 map 原子”a（通常 H、O、Cl…）
      - 锚点重原子 = a 的唯一重原子邻居（>1）
      - 离去原子 = a 本身
    """
    anchors: Dict[int, int] = {}
    leaving: List[int] = []
    for a in mol.GetAtoms():
        amap = a.GetAtomMapNum()
        # print(amap)
        if amap in [1,2]:#NOTE this version only considering backbone not sidechain, D has 3
            heavy_nbrs = [nb for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1]
            if len(heavy_nbrs) != 1:
                raise ValueError(
                    f"AtomMapNum={amap} 的离去原子需恰有 1 个重原子邻居；"
                    f"当前 {len(heavy_nbrs)} 个（atom idx={a.GetIdx()}, sym={a.GetSymbol()})"
                )
            anchors[amap] = heavy_nbrs[0].GetIdx()
            leaving.append(a.GetIdx())
    # print(anchors,leaving)
    if not anchors:
        raise ValueError("未发现任何 AtomMapNum>0 的原子；请确认单体为 HELM corelib 风格 SMILES")
    return anchors, leaving

def monomer_to_mol_helm(entry: dict, want_D: bool) -> Chem.Mol:
    """
    entry 结构示例：
      {
        "code": "Pen",
        "polymer_type": "PEPTIDE",
        "smiles_L": "CC(C)(S[H:3])[C@H](N[H:1])C([OH:2])=O",
        "smiles_D": "CC(C)(S[H:3])[C@@H](N[H:1])C([OH:2])=O"
      }
      或帽基/化学基：
      { "code":"ac", "polymer_type":"CAP_N", "smiles":"C(C)(=O)[OH:2]" }
      { "code":"am", "polymer_type":"CAP_C", "smiles":"N[H:1]" }
    """
    # t = entry.get("polymer_type", "PEPTIDE").upper()
    # if t in ("CAP_N", "CAP_C", "CHEM"):
    #     smi = entry["smiles"]
    # else:  # PEPTIDE
    #     if want_D and entry.get("smiles_D"):
    #         smi = entry["smiles_D"]
    #     else:
    #         smi = entry.get("smiles_L") or entry["smiles"]
    smi = entry["smiles"]        
    params = Chem.SmilesParserParams()
    params.removeHs = False        # <<< 保留显式氢（[H:1]）
    # params.strictParsing = True
    m = Chem.MolFromSmiles(smi, params)
    if m is None:
        raise ValueError(f"[{entry.get('code')}] SMILES 解析失败: {smi}")
    return m

def build_fragment(entry: dict, want_D: bool):
    """
    返回片段对象: (mol, anchors_dict, leaving_atoms_list, type_str)
    anchors_dict: {1: idx, 2: idx, 3: idx ...}  # 可能只含 1/2 或部分
    """
    mol = monomer_to_mol_helm(entry, want_D)
    anchors, leaving = anchors_and_leaving_from_helm(mol)
    return mol, anchors, leaving, entry.get("polymer_type", "PEPTIDE").upper()

def fuse_by_anchor_maps_helm(molA, anchorsA, leavingA,
                             molB, anchorsB, leavingB,
                             mapA: int, mapB: int):
    """
    用 A 的 mapA 锚点与 B 的 mapB 锚点成单键。
    返回新 (mol, anchors_tail, leaving_all)
      - anchors_tail: 仅保留“右侧片段 B 的 anchors”，并加上偏移（用于下一步继续串接）
      - leaving_all: 合并后的“离去原子索引”（B 的需要加偏移）
    """
    if mapA not in anchorsA:
        raise ValueError(f"A 片段缺少 R{mapA} 锚点")
    if mapB not in anchorsB:
        raise ValueError(f"B 片段缺少 R{mapB} 锚点")

    combo = Chem.CombineMols(molA, molB)
    rw = Chem.RWMol(combo)
    offset = molA.GetNumAtoms()

    a_idx = anchorsA[mapA]
    b_idx = anchorsB[mapB] + offset
    rw.AddBond(a_idx, b_idx, Chem.rdchem.BondType.SINGLE)
    new_mol = rw.GetMol()

    # 下一步继续连接时，应当使用“右侧片段”的 R2 等锚点 => 将 B 的 anchors 统一 +offset 后返回
    anchors_tail = {k: (v + offset) for k, v in anchorsB.items()}
    leaving_all = list(leavingA) + [i + offset for i in leavingB]
    return new_mol, anchors_tail, leaving_all


def _tokenize_preserve_brackets(seq: str) -> List[str]:
    """Split sequence by hyphens while preserving bracketed content."""
    tokens = []
    buf = []
    depth = 0
    for ch in seq.strip():
        if ch == '[':
            depth += 1
            buf.append(ch)
        elif ch == ']':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == '-' and depth == 0:
            tok = ''.join(buf).strip()
            if tok:
                tokens.append(tok)
            buf = []
        else:
            buf.append(ch)
    last = ''.join(buf).strip()
    if last:
        tokens.append(last)
    return tokens

def _strip_brackets(token: str) -> str:
    """Remove surrounding brackets from token."""
    return token[1:-1].strip() if token.startswith('[') and token.endswith(']') else token.strip()

def parse_sequence(seq: str, lib: MonomerLib) -> ParsedSeq:
    """Parse peptide sequence into residues and caps."""
    raw_tokens = _tokenize_preserve_brackets(seq)
    if not raw_tokens:
        return ParsedSeq([], {"symbol": "H", "smiles": "[H]"}, {"symbol": "H", "smiles": "[H]"})

    # Process N-terminal cap
    head_sym = _strip_brackets(raw_tokens[0]).lower()
    if head_sym in N_CAPS:
        n_cap = {"symbol": head_sym, "smiles": N_CAPS[head_sym]}
        core_tokens = raw_tokens[1:]
    else:
        n_cap = {"symbol": "H", "smiles": "[H]"}
        core_tokens = raw_tokens

    # Process C-terminal cap
    if core_tokens:
        tail_sym = _strip_brackets(core_tokens[-1]).lower()
        if tail_sym in C_CAPS:
            c_cap = {"symbol": tail_sym, "smiles": C_CAPS[tail_sym]}
            core_tokens = core_tokens[:-1]
        else:
            c_cap = {"symbol": "H", "smiles": "[H]"}
    else:
        c_cap = {"symbol": "H", "smiles": "[H]"}

    # Process residues
    residues = []
    staples: List[Tuple[int, int]] = []
    cyclo = False
    d_flag_next = False

    for token in core_tokens:
        if not token:
            continue

        # Handle bracketed content
        if token.startswith('[') and token.endswith(']'):
            inner = token[1:-1].strip()
            inner_lower = inner.lower()
            if inner_lower in {"cyclo", "cycle"}:
                cyclo = True
                continue
            if inner_lower.startswith("staple"):
                start = inner.find("(")
                end = inner.rfind(")")
                if start != -1 and end != -1 and start < end:
                    payload = inner[start + 1: end].replace(" ", "")
                    payload = payload.replace("-", ",")
                    parts = [p for p in payload.split(",") if p]
                    if len(parts) == 2:
                        try:
                            i = int(parts[0])
                            j = int(parts[1])
                            if i != j and i > 0 and j > 0:
                                staples.append(tuple(sorted((i, j))))
                                continue
                        except ValueError:
                            pass
                raise ValueError(f"Invalid staple specification: '{inner}'")
            want_d = False
            
            if inner[:2].lower() == 'd-' and len(inner) > 2:
                code_str = inner[2:].strip()
                want_d = True
            elif (inner.lower().startswith('d') and 
                  lib.resolve_code(inner[1:]) and 
                  not lib.resolve_code(inner)):
                code_str = inner[1:]
                want_d = True
            else:
                code_str = inner
        else:
            code_str = token.strip()
            stripped = code_str.lower()
            if stripped in {"cyclo", "cycle"}:
                cyclo = True
                continue
            if stripped.startswith("staple"):
                start = code_str.find("(")
                end = code_str.rfind(")")
                if start != -1 and end != -1 and start < end:
                    payload = code_str[start + 1: end].replace(" ", "")
                    payload = payload.replace("-", ",")
                    parts = [p for p in payload.split(",") if p]
                    if len(parts) == 2:
                        try:
                            i = int(parts[0])
                            j = int(parts[1])
                            if i != j and i > 0 and j > 0:
                                staples.append(tuple(sorted((i, j))))
                                continue
                        except ValueError:
                            pass
                raise ValueError(f"Invalid staple specification: '{code_str}'")
            want_d = d_flag_next
            if code_str in ('d', 'd-'):
                d_flag_next = True
                continue

        code = lib.resolve_code(code_str)
        if not code:
            raise KeyError(f"Unknown residue: '{code_str}'")
        
        residues.append((code, want_d))
        d_flag_next = False

    return ParsedSeq(residues=residues, n_cap=n_cap, c_cap=c_cap, staples=staples, cyclo=cyclo)



def sequences_to_smiles(seqs: List[str], lib: MonomerLib) -> Dict[str, str]:
    """Convert multiple sequences to SMILES strings."""
    return {seq: seq2smi(seq, lib) for seq in seqs}

def remove_leaving_and_sanitize(mol: Chem.Mol, leaving_indices: List[int]) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    # 倒序删除，越界/已被 RDKit 处理掉的索引自动跳过
    for idx in sorted(set(leaving_indices), reverse=True):
        if 0 <= idx < rw.GetNumAtoms():
            try:
                rw.RemoveAtom(idx)
            except Exception:
                pass
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Chem.rdchem.AtomValenceException as exc:
        details = []
        for atom in out.GetAtoms():
            details.append(
                f"{atom.GetIdx()}:{atom.GetSymbol()} deg={atom.GetDegree()} "
                f"val={atom.GetExplicitValence()} h={atom.GetTotalNumHs()}"
            )
        summary = " | ".join(details[:20])
        if len(details) > 20:
            summary += " | ..."
        raise ValueError(
            "Sanitize failure after removing leaving atoms: " + summary
        ) from exc
    return out


def _filter_terminal_leaving(mol: Chem.Mol, leaving_indices: List[int], keep_terminal_oh: bool) -> List[int]:
    """Filter leaving atoms to preserve terminal hydroxyl when needed."""
    if not keep_terminal_oh:
        return leaving_indices

    keep = set()
    for idx in leaving_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 8:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            continue
        carbon = neighbors[0]
        if carbon.GetAtomicNum() != 6:
            continue
        is_carbonyl = any(
            bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
            and bond.GetOtherAtom(carbon).GetAtomicNum() == 8
            for bond in carbon.GetBonds()
        )
        if is_carbonyl:
            keep.add(idx)
            for nb in neighbors:
                if nb.GetAtomicNum() == 1:
                    keep.add(nb.GetIdx())

    if not keep:
        return leaving_indices

    return [idx for idx in leaving_indices if idx not in keep]


def _ensure_terminal_carboxyl(mol: Chem.Mol, c_cap_symbol: str) -> Chem.Mol:
    if c_cap_symbol.lower() != "h":
        return mol
    matches = get_backbone_atoms(mol)
    if not matches:
        return mol
    _, _, c_idx = matches[-1]
    rw = Chem.RWMol(mol)
    carbon = rw.GetAtomWithIdx(c_idx)
    single_oxygen_idx = None
    for bond in carbon.GetBonds():
        other = bond.GetOtherAtom(carbon)
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE and other.GetAtomicNum() == 8:
            single_oxygen_idx = other.GetIdx()
            break
    if single_oxygen_idx is None:
        o_idx = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(c_idx, o_idx, Chem.rdchem.BondType.SINGLE)
        h_idx = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(o_idx, h_idx, Chem.rdchem.BondType.SINGLE)
    else:
        oxygen = rw.GetAtomWithIdx(single_oxygen_idx)
        has_h = any(nb.GetAtomicNum() == 1 for nb in oxygen.GetNeighbors())
        if not has_h:
            h_idx = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(single_oxygen_idx, h_idx, Chem.rdchem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def seq2smi(
    seq: str,
    lib: MonomerLib,
    staple_prefs: Optional[Dict[int, Optional[str]]] = None,
    linker_smiles: Optional[str] = None,
):
    staple_prefs = dict(staple_prefs or {})
    parsed = parse_sequence(seq, lib)
    if not parsed.residues:
        raise ValueError("序列中未找到任何残基。")

    if parsed.staples:
        raise NotImplementedError("暂不支持在序列中直接使用 [Staple(i,j)] 标记。")

    if parsed.cyclo:
        if parsed.n_cap.get("symbol", "h").lower() != "h" or parsed.c_cap.get("symbol", "h").lower() != "h":
            raise ValueError("环化肽不应同时指定 N/C 端封端。")
        if len(parsed.residues) < 2:
            raise ValueError("环化肽至少需要包含两个残基。")

    if staple_prefs:
        if linker_smiles is None:
            raise ValueError("指定了装订位点，但缺少链接体的 SMILES。")
        if parsed.cyclo:
            raise ValueError("暂不支持同时存在环化与多点装订。")
        max_pos = len(parsed.residues)
        for pos in staple_prefs:
            if pos < 1 or pos > max_pos:
                raise ValueError(f"装订位点 {pos} 超出残基数量（{max_pos}）。")

    components: List[Tuple[dict, bool]] = []
    component_meta: List[Tuple[str, Optional[int]]] = []

    # N-cap
    if parsed.n_cap and parsed.n_cap.get("symbol") and parsed.n_cap["symbol"].lower() != "h":
        code = lib.resolve_code(parsed.n_cap["symbol"])
        if not code:
            raise KeyError(f"N-cap '{parsed.n_cap['symbol']}' 未在 monomer 库定义（需要 HELM 风格 SMILES）")
        components.append((lib.get(code), False))
        component_meta.append(("cap", None))

    # residues
    for idx, (code, is_d) in enumerate(parsed.residues, start=1):
        ent = lib.get(code)
        components.append((ent, is_d))
        component_meta.append(("residue", idx))

    # C-cap
    if parsed.c_cap and parsed.c_cap.get("symbol") and parsed.c_cap["symbol"].lower() != "h":
        code = lib.resolve_code(parsed.c_cap["symbol"])
        if not code:
            raise KeyError(f"C-cap '{parsed.c_cap['symbol']}' 未在 monomer 库定义（需要 HELM 风格 SMILES）")
        components.append((lib.get(code), False))
        component_meta.append(("cap", None))

    if not components:
        raise ValueError("没有可装配的组件（检查序列或库）")

    cur_mol, anc0, leave0, _ = build_fragment(components[0][0], components[0][1])
    leaving_all: List[int] = list(leave0)
    tail_anchors = anc0

    residue_anchor_maps: List[Optional[Dict[int, int]]] = [None] * len(parsed.residues)
    if component_meta[0][0] == "residue":
        residue_anchor_maps[component_meta[0][1] - 1] = dict(anc0)
    residue_side_anchors: List[Optional[int]] = [None] * len(parsed.residues)
    warnings: List[str] = []

    if component_meta[0][0] == "residue":
        res_pos = component_meta[0][1]
        prefer_atom = staple_prefs.get(res_pos)
        side_anchor, note = _determine_side_anchor(cur_mol, prefer_atom, components[0][0]["code"], res_pos)
        if res_pos in staple_prefs and side_anchor is None:
            raise ValueError(f"无法在残基 {res_pos} 中定位用于装订的侧链原子。")
        if note:
            warnings.append(note)
        if side_anchor is not None:
            residue_side_anchors[res_pos - 1] = side_anchor

    for comp_idx, (ent, want_d) in enumerate(components[1:], start=1):
        fragment, anchors, leaving, _ = build_fragment(ent, want_d)
        side_anchor_local: Optional[int] = None
        res_pos = None
        if component_meta[comp_idx][0] == "residue":
            res_pos = component_meta[comp_idx][1]
            prefer_atom = staple_prefs.get(res_pos)
            side_anchor_local, note = _determine_side_anchor(fragment, prefer_atom, ent["code"], res_pos)
            if res_pos in staple_prefs and side_anchor_local is None:
                raise ValueError(f"无法在残基 {res_pos} 中定位用于装订的侧链原子。")
            if note:
                warnings.append(note)
        offset = cur_mol.GetNumAtoms()
        cur_mol, tail_anchors, leaving_all = fuse_by_anchor_maps_helm(
            cur_mol,
            tail_anchors,
            leaving_all,
            fragment,
            anchors,
            leaving,
            mapA=2,
            mapB=1,
        )
        if component_meta[comp_idx][0] == "residue":
            residue_anchor_maps[component_meta[comp_idx][1] - 1] = dict(tail_anchors)
            if side_anchor_local is not None:
                residue_side_anchors[res_pos - 1] = side_anchor_local + offset

    if parsed.cyclo:
        first_map = residue_anchor_maps[0]
        last_map = residue_anchor_maps[-1]
        if first_map is None or last_map is None:
            raise ValueError("环化肽缺少必要的锚点信息。")
        cur_mol = _apply_head_tail_cyclization(cur_mol, first_map, last_map)
    if staple_prefs:
        cur_mol, extra_leaving = _attach_linker_to_peptide(
            cur_mol,
            residue_side_anchors,
            staple_prefs,
            linker_smiles,
        )
        leaving_all.extend(extra_leaving)

    cur_mol = remove_leaving_and_sanitize(cur_mol, leaving_all)
    if not parsed.cyclo:
        cur_mol = _ensure_terminal_carboxyl(cur_mol, parsed.c_cap.get("symbol", ""))

    rd_smi = Chem.MolToSmiles(cur_mol, isomericSmiles=True)
    for note in warnings:
        print(f"[seq2smi_v2] WARNING: {note}", file=sys.stderr)
    return rd_smi


def _apply_head_tail_cyclization(
    mol: Chem.Mol,
    first_anchors: Dict[int, int],
    last_anchors: Dict[int, int],
) -> Chem.Mol:
    """Create a bond between the first residue R1 and last residue R2 anchors."""
    if 1 not in first_anchors or 2 not in last_anchors:
        raise ValueError("环化肽需要首残基的 R1 与末残基的 R2 锚点。")
    n_idx = first_anchors[1]
    c_idx = last_anchors[2]
    if mol.GetBondBetweenAtoms(n_idx, c_idx):
        return mol
    rw = Chem.RWMol(mol)
    rw.AddBond(n_idx, c_idx, Chem.rdchem.BondType.SINGLE)
    return rw.GetMol()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert peptide sequences to SMILES")
    parser.add_argument(
        "--input", '-i',
        dest="input",
        default="seq2smi_v2_input.txt",
        help="Custom input smiles files input_sequences.txt)",
    )
    parser.add_argument("--output", "-o", default='seqs2smi_v2_out.txt', help="Output file for SMILES")
    parser.add_argument("--lib", "-l", default='data/monomersFromHELMCoreLibrary.json',help="Optional JSON file with additional monomers")
    args = parser.parse_args()

    # Initialize library
    lib = MonomerLib(args.lib)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return

    results: List[Tuple[str, str]] = []
    for line in lines:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if not parts:
            continue
        seq_line = parts[0]
        staple_prefs: Dict[int, Optional[str]] = {}
        linker_smiles: Optional[str] = None
        for extra in parts[1:]:
            lower = extra.lower()
            if lower.startswith("pose:"):
                if staple_prefs:
                    raise ValueError(f"发现重复的 pose 定义: '{line}'")
                staple_prefs = _parse_pose_line(extra)
            elif lower.startswith("linker:"):
                if linker_smiles is not None:
                    raise ValueError(f"发现重复的 linker 定义: '{line}'")
                _, _, payload = extra.partition(":")
                linker_smiles = payload.strip()
            else:
                raise ValueError(f"无法识别的字段: '{extra}'")

        if staple_prefs and not linker_smiles:
            raise ValueError(f"检测到 pose 配置但缺少链接体 SMILES: '{line}'")
        if linker_smiles and not staple_prefs:
            raise ValueError(f"提供了链接体但未指定 pose: '{line}'")

        smiles = seq2smi(
            seq_line,
            lib,
            staple_prefs=staple_prefs or None,
            linker_smiles=linker_smiles,
        )
        results.append((line, smiles))

    with open(args.output, "w", encoding="utf-8") as f:
        for seq, smi in results:
            f.write(f"{seq}\t{smi}\n")

if __name__ == "__main__":
    main()

"""
python seq2smi_v2.py -i seq2smi_v2_input.txt -o seqs2smi_v2_out.txt 
python -i 
"""    
