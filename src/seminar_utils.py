"""Small local-data helpers; no API calls or model downloads on import."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def select_device(requested='cpu'):
    import torch
    if requested not in {'cpu', 'mps', 'cuda', 'auto'}:
        raise ValueError('DEVICE должен быть cpu, mps, cuda или auto')
    if requested == 'auto':
        requested = ('cuda' if torch.cuda.is_available() else
                     'mps' if torch.backends.mps.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна; выберите cpu или mps.')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS недоступен; выберите cpu.')
    return torch.device(requested), (torch.float32 if requested == 'cpu' else torch.float16)


def synchronize_device(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def load_pilot(data_dir):
    """Return model inputs separately from historical reference labels."""
    data_dir = Path(data_dir)
    records = {r['sample_id']: r for r in map(json.loads,
               (data_dir / 'records.jsonl').read_text().splitlines())}
    inputs, references = [], []
    with (data_dir / 'visual_audit.csv').open() as stream:
        for row in csv.DictReader(stream):
            if not row['decision'].startswith('pilot'):
                continue
            if row['answer_leakage'] != 'false':
                raise ValueError('Учебная картинка попала в выборку')
            sid = row['sample_id']
            path = (data_dir / 'images' / f'{sid}.png').resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            inputs.append({'sample_id': sid, 'image_path': str(path),
                           'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            refs = records[sid]['known_solutions']
            if len(refs) != 1:
                raise ValueError(f'Неоднозначный контроль: {sid}')
            gold = refs[0]['output_values']
            references.append({'sample_id': sid, 'store': row['store'],
                               'reference_price': gold.get('correct_price', {}).get('price'),
                               'reference_no_price': gold['no_price'],
                               'needs_policy_review': row['needs_policy_review'] == 'true',
                               'gold_verified_for_seminar': False})
    if not inputs or len({x['image_sha256'] for x in inputs}) != len(inputs):
        raise ValueError('Пустая выборка или дубликаты изображений')
    return inputs, references


def parse_hint(raw):
    """Strict validation, not constrained decoding. Invalid output stays invalid."""
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)[:-3].strip()
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {'price', 'status', 'hint'}:
        raise ValueError('Ожидаются ровно поля price, status, hint')
    if result['status'] not in ('ok', 'no_price', 'uncertain'):
        raise ValueError('Неизвестный status')
    if not isinstance(result['hint'], str) or not result['hint'].strip():
        raise ValueError('Пустая подсказка')
    price = result['price']
    if result['status'] == 'ok':
        if (type(price) not in (int, float) or not math.isfinite(price) or price <= 0):
            raise ValueError('Для ok нужна положительная конечная числовая цена')
    elif price is not None:
        raise ValueError('Для no_price/uncertain ожидается price=null')
    return result


def make_messages(prompt):
    # Deliberate allowlist: no record, reference answer, store or audit text.
    return [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': prompt}]}]


def evaluate_predictions(predictions, references):
    """Diagnostic comparison against provisional references, including failures."""
    import pandas as pd
    rows = []
    refs = {x['sample_id']: x for x in references}
    for pred in predictions:
        ref = refs[pred['sample_id']]
        parsed = pred.get('parsed')
        price = parsed['price'] if parsed else None
        gold = float(ref['reference_price']) if ref['reference_price'] is not None else None
        correct = bool(parsed and (
            (ref['reference_no_price'] and parsed['status'] == 'no_price') or
            (not ref['reference_no_price'] and parsed['status'] == 'ok'
             and abs(price - gold) <= 0.005)))
        rows.append({'sample_id': pred['sample_id'], 'prompt_version': pred['prompt_version'],
                     'store': ref['store'], 'model_price': price, 'reference_price': gold,
                     'valid_output': parsed is not None, 'matches_reference': correct,
                     'relative_error': (price - gold) / gold
                     if price is not None and gold is not None and gold > 0 else None,
                     'hint': parsed['hint'] if parsed else None,
                     'error': pred.get('error'),
                     'needs_policy_review': ref['needs_policy_review']})
    return pd.DataFrame(rows)
