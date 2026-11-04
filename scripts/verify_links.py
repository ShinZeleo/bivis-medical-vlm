import os
import re
import sys
from urllib.parse import unquote

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

broken = []
total_links = 0
link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

for root, dirs, files in os.walk('.'):
    if any(skip in root for skip in ['.git', '.venv', '__pycache__', '.pytest_cache']):
        continue
    for f in files:
        if f.endswith('.md'):
            doc_path = os.path.join(root, f)
            with open(doc_path, 'r', encoding='utf-8', errors='ignore') as fp:
                content = fp.read()
            for match in link_pattern.finditer(content):
                target = match.group(2).strip()
                if any(target.startswith(p) for p in ['http://', 'https://', 'mailto:', '#', 'conversation://', 'sidecar://', 'file:///']):
                    continue
                clean_target = target.split('#')[0].split('?')[0].strip()
                if not clean_target:
                    continue
                clean_target = clean_target.strip('<>"\'')
                clean_target = unquote(clean_target)
                total_links += 1
                doc_dir = os.path.dirname(doc_path)
                resolved = os.path.normpath(os.path.join(doc_dir, clean_target))
                if not os.path.exists(resolved):
                    broken.append((doc_path, match.group(0), clean_target, resolved))

print(f"Total internal links scanned: {total_links}")
print(f"Total broken links: {len(broken)}")
for doc, raw, target, resolved in broken:
    print(f"FAIL in {doc} -> {target} (resolved: {resolved})")
