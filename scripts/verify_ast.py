import sys
import json
import ast
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXPECTED_CELL_COUNTS = {
    "08": 16,
    "09": 16,
    "10": 11
}

nb_files = sorted(list(Path("notebooks/kaggle").rglob("*.ipynb")))
total_cells = 0
failed_cells = 0
structural_errors = 0

print("=" * 80)
print("AUDIT STRUKTUR NOTEBOOK & VERIFIKASI SINTAKS AST (KAGGLE PIPELINE)")
print("=" * 80)

for nb_path in nb_files:
    fname = nb_path.name
    rel_path = nb_path.relative_to(Path("."))
    
    # Determine expected cell count
    prefix = fname[:2]
    expected_count = EXPECTED_CELL_COUNTS.get(prefix, None)
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    actual_count = len(code_cells)
    
    print(f"\n📂 Notebook: {rel_path}")
    print(f"   Jumlah Sel Kode: {actual_count} (Ekspektasi: {expected_count if expected_count else 'Bebas'})")
    
    if expected_count is not None and actual_count != expected_count:
        print(f"   ❌ STRUCTURAL MISMATCH: Ekspektasi {expected_count} sel kode, ditemukan {actual_count}!")
        structural_errors += 1
    else:
        print(f"   ✅ Structural Check OK: Jumlah sel kode sesuai ({actual_count} sel).")
        
    code_cell_idx = 0
    for cell in code_cells:
        code_cell_idx += 1
        total_cells += 1
        src = "".join(cell.get("source", []))
        
        # Extract stage header
        stage_hdr = "TANPA HEADER TAHAP"
        for line in src.splitlines():
            line_str = line.strip()
            if "STAGE" in line_str.upper() or "RQ" in line_str.upper() or "EXPERIMENT" in line_str.upper():
                stage_hdr = line_str.strip("# -=").strip()
                break
            elif line_str.startswith("#") and len(line_str) > 5 and stage_hdr == "TANPA HEADER TAHAP":
                stage_hdr = line_str.strip("# -=").strip()
                
        # Clean ipython magic lines for ast.parse
        clean_lines = []
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("%") or stripped.startswith("!"):
                clean_lines.append(f"# {line}")
            else:
                clean_lines.append(line)
        clean_code = "\n".join(clean_lines)
        
        try:
            ast.parse(clean_code)
            print(f"   Sel {code_cell_idx:02d}: ✅ AST Valid | [{stage_hdr[:60]}]")
        except SyntaxError as e:
            failed_cells += 1
            print(f"   Sel {code_cell_idx:02d}: ❌ SYNTAX ERROR: {e} | [{stage_hdr[:60]}]")

print("\n" + "=" * 80)
print("HASIL AUDIT STRUKTUR DAN AST")
print("=" * 80)
print(f"Total Sel Kode Diuji   : {total_cells}")
print(f"Kegagalan Sintaks AST  : {failed_cells}")
print(f"Kesalahan Struktur Sel : {structural_errors}")

if failed_cells == 0 and structural_errors == 0:
    print("🎉 100% AUDIT PASS: Semua sel kode lolos ast.parse dan memenuhi kuota struktur sel!")
    sys.exit(0)
else:
    print("⚠️ AUDIT GAGAL: Periksa kesalahan sintaks atau ketidaksesuaian jumlah sel di atas!")
    sys.exit(1)
