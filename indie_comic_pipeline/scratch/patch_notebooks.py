import json, os, glob

PIPELINE_DIR = r'c:\Users\Dell\Downloads\drid\indie_comic_pipeline'

SETUP_CELL = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'colab_setup_cell',
    'metadata': {},
    'outputs': [],
    'source': [
        '# ============================================================\n',
        '# Universal Colab/Local Setup — run this first in every notebook\n',
        '# ============================================================\n',
        'import os, sys\n',
        '\n',
        'try:\n',
        '    from google.colab import files  # type: ignore\n',
        '    _IN_COLAB = True\n',
        'except ImportError:\n',
        '    _IN_COLAB = False\n',
        '\n',
        'if _IN_COLAB:\n',
        '    import subprocess\n',
        '    _repo = "/content/Indie-Comic"\n',
        '    if not os.path.exists(_repo):\n',
        '        subprocess.run(["git", "clone", "--depth", "1",\n',
        '            "https://github.com/Cyberpunk-San/Indie-Comic.git", _repo], check=True)\n',
        '    exec(open(f"{_repo}/indie_comic_pipeline/colab_setup.py").read())\n',
        'else:\n',
        '    _candidates = [\n',
        '        os.path.join(os.getcwd(), "colab_setup.py"),\n',
        '        os.path.join(os.getcwd(), "indie_comic_pipeline", "colab_setup.py"),\n',
        '    ]\n',
        '    _found = next((p for p in _candidates if os.path.exists(p)), None)\n',
        '    if _found:\n',
        '        exec(open(_found).read())\n',
        '    else:\n',
        '        print("colab_setup.py not found — run from repo root")\n',
    ]
}

notebooks = sorted(glob.glob(os.path.join(PIPELINE_DIR, '0*.ipynb')))
patched = 0
for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    
    if cells and cells[0].get('id') == 'colab_setup_cell':
        print(f'  (skip, already patched) {os.path.basename(nb_path)}')
        continue
    
    # Insert after leading markdown section headers
    insert_pos = 0
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'markdown':
            insert_pos = i + 1
        else:
            break
    
    nb['cells'].insert(insert_pos, SETUP_CELL)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f'  patched: {os.path.basename(nb_path)}')
    patched += 1

print(f'\nDone. {patched} notebooks patched.')
