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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from collections import namedtuple
from rdkit import Chem
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

ParsedSeq = namedtuple("ParsedSeq", ["residues", "n_cap", "c_cap"])

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
    d_flag_next = False
    
    for token in core_tokens:
        if not token:
            continue

        # Handle bracketed content
        if token.startswith('[') and token.endswith(']'):
            inner = token[1:-1].strip()
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
            want_d = d_flag_next
            if code_str.lower() in ('d', 'd-'):
                d_flag_next = True
                continue

        code = lib.resolve_code(code_str)
        if not code:
            raise KeyError(f"Unknown residue: '{code_str}'")
        
        residues.append((code, want_d))
        d_flag_next = False

    return ParsedSeq(residues=residues, n_cap=n_cap, c_cap=c_cap)



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
    Chem.SanitizeMol(out)
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


def seq2smi(seq, lib):
    raw_tokens=_tokenize_preserve_brackets(seq)
    # res = sequences_to_smiles(seqs, lib)
    # parsed
    # -------- 1) 识别 N/C 端帽基 --------
    # 先只基于 symbol 判断（去括号、统一小写）
    head_sym = _strip_brackets(raw_tokens[0]).lower()
    tail_sym = _strip_brackets(raw_tokens[-1]).lower()
    if head_sym in N_CAPS:
        n_cap = {"symbol": head_sym, "smiles": N_CAPS[head_sym]}
        core_tokens = raw_tokens[1:]    # 去掉 N-cap
    else:
        n_cap = DEFAULT_N_CAP
        core_tokens = raw_tokens[:]

    if core_tokens:  # 只有在还剩 token 才判断尾帽基
        tail_sym2 = _strip_brackets(core_tokens[-1]).lower()
        if tail_sym2 in C_CAPS:
                c_cap = {"symbol": tail_sym2, "smiles": C_CAPS[tail_sym2]}
                core_tokens = core_tokens[:-1]  # 去掉 C-cap
        else:
                c_cap = DEFAULT_C_CAP
    else:
        # 没有残基（只有帽基？极端情况）
        c_cap = DEFAULT_C_CAP

    # -------- 2) 解析残基（保留你原来的 D 规则） --------
    residues: List[Tuple[str, bool]] = []
    d_flag_next = False
    for t in core_tokens:
        if not t:
            continue
        # 1) 括号 token：作为整体，不再按 '-' 拆分
        if t.startswith('[') and t.endswith(']'):
            inner = t[1:-1].strip()
            want_D = False
            # [D-Arg] / [d-Arg] / [dArg]
            if inner[:2].lower() == 'd-' and len(inner) > 2:
                code_str = inner[2:].strip()
                want_D = True
            elif (inner.lower().startswith('d')
                    and lib.resolve_code(inner[1:])
                    and not lib.resolve_code(inner)):
                code_str = inner[1:]
                want_D = True
            else:
                code_str = inner

            code = lib.resolve_code(code_str)
            if not code:
                raise KeyError(f"未知 token（括号内）: '{inner}'，请在 monomer 库添加别名/映射")
            residues.append((code, want_D))
            d_flag_next = False
            continue

        # 2) 非括号 token：处理 D 前缀与独立 D
        t_norm = t.strip()
        if t_norm in ('[D', '[d'):
            d_flag_next = True
            # continue

        want_D = False#TODO del as useless
        code_str = t_norm

        code = lib.resolve_code(code_str)
        if not code:
            raise KeyError(f"未知 token: '{t_norm}'，请在 monomer 库添加别名/映射")

        if d_flag_next:
            want_D = True
        residues.append((code, want_D))
        d_flag_next = False
    parsed=ParsedSeq(residues=residues, n_cap=n_cap, c_cap=c_cap)            
    # 1) 按顺序构造组件列表（先 N-cap，再残基，最后 C-cap）
    components = []
    # N-cap
    if parsed.n_cap and parsed.n_cap.get("symbol") and parsed.n_cap["symbol"].lower() != "h":
        code = lib.resolve_code(parsed.n_cap["symbol"])
        if not code:
            raise KeyError(f"N-cap '{parsed.n_cap['symbol']}' 未在 monomer 库定义（需要 HELM 风格 SMILES）")
        ent = lib.get(code)
        components.append( (ent, False) )  # 帽基无 D/L 概念
    # residues
    ii=0
    for code, is_D in parsed.residues:
        ent = lib.get(code)
        components.append( (ent, is_D) )
        # print(ii,(ent, is_D))
        ii+=1
    # C-cap
    if parsed.c_cap and parsed.c_cap.get("symbol") and parsed.c_cap["symbol"].lower() != "h":
        code = lib.resolve_code(parsed.c_cap["symbol"])
        if not code:
            raise KeyError(f"C-cap '{parsed.c_cap['symbol']}' 未在 monomer 库定义（需要 HELM 风格 SMILES）")
        ent = lib.get(code)
        components.append( (ent, False) )


    if not components:
        raise ValueError("没有可装配的组件（检查序列或库）")

    m0, anc0, leave0, t0 = build_fragment(components[0][0], components[0][1])    
    # tail_anchors 表示“当前最右端片段”的 anchors；leaving 累加
    cur_mol = m0
    tail_anchors = anc0
    leaving_all: List[int] = list(leave0)   
    # 3) 逐个拼接（统一按 R2 — R1）
    for ent, want_D in components[1:]:
        mB, ancB, leaveB, tB = build_fragment(ent, want_D)
        # print(mB, ancB, leaveB, tB)
        cur_mol, tail_anchors, leaving_all = fuse_by_anchor_maps_helm(
            cur_mol, tail_anchors, leaving_all,
            mB, ancB, leaveB,
            mapA=2, mapB=1
        )

    # 4) 删除所有离去原子并清理
    filtered_leaving = leaving_all
    cur_mol = remove_leaving_and_sanitize(cur_mol, filtered_leaving)
    cur_mol = _ensure_terminal_carboxyl(cur_mol, parsed.c_cap.get("symbol", ""))
    rd_smi=Chem.MolToSmiles(cur_mol)
    return rd_smi


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert peptide sequences to SMILES")
    parser.add_argument(
        "--input",
        dest="input",
        default="seq2smi_input.txt",
        help="Custom input smiles files input_sequences.txt)",
    )
    parser.add_argument("--output", "-o", default='seqs2smi_out.txt', help="Output file for SMILES")
    parser.add_argument("--lib", "-l", default='data/monomersFromHELMCoreLibrary.json',help="Optional JSON file with additional monomers")
    args = parser.parse_args()

    # Initialize library
    lib = MonomerLib(args.lib)

    # Read sequences
    with open(args.input, "r") as f:
        seqs = [line.strip() for line in f if line.strip()]

    # Convert to SMILES
    results = sequences_to_smiles(seqs, lib)

    # Write results
    with open(args.output, "w") as f:
        for seq, smi in results.items():
            f.write(f"{seq}\t{smi}\n")

if __name__ == "__main__":
    main()

"""
python seq2smi.py -i seq2smi_input.txt -o seqs2smi_out.txt 
python -i 
"""    
