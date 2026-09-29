"""Infrastructure for the two seminar workbooks. No network calls on import."""
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse
import requests

ORIGIN = 'https://tasks.yandex.ru'


def connect_tasks(journal):
    from getpass import getpass
    token = getpass('OAuth-токен Яндекс Заданий: ').strip()
    if not token:
        raise ValueError('Токен не введён.')
    return TasksAPI(token, journal, allow_write=True)


def open_paid_pool(api, pool_id, journal_key):
    from getpass import getpass
    phrase = f'OPEN {pool_id}'
    if getpass(f'Платный запуск. После проверки модерации, бюджета и интерфейса введите {phrase}: ') != phrase:
        raise RuntimeError('Запуск отменён.')
    return api.write_once(journal_key, 'POST', f'/api/v1/pools/{pool_id}/open')


class TasksAPI:
    def __init__(self, token, journal, allow_write=False):
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'OAuth {token}',
                                     'Content-Type': 'application/json'})
        self.journal = Path(journal)
        self.allow_write = allow_write

    def request(self, method, path, **kwargs):
        if not path.startswith(('/api/v1/', '/api/new/requester/')) or '?' in path:
            raise ValueError('Используйте относительный путь API; параметры — через params')
        if method != 'GET' and not self.allow_write:
            raise RuntimeError('Запись выключена: ALLOW_API_WRITE=False')
        r = self.session.request(method, ORIGIN + path, timeout=(10, 90),
                                 allow_redirects=False, **kwargs)
        if not 200 <= r.status_code < 300:
            # Do not print headers, tokens or the full submitted payload.
            raise RuntimeError(f'API HTTP {r.status_code}: {r.text[:500]}')
        return r.json() if r.content else None

    def write_once(self, key, method, path, payload=None, params=None):
        if not self.allow_write:
            raise RuntimeError('Сначала разрешите запись явно.')
        digest = hashlib.sha256(json.dumps([method, path, payload, params],
                               sort_keys=True).encode()).hexdigest()
        state = json.loads(self.journal.read_text()) if self.journal.exists() else {}
        if key in state:
            entry = state[key]
            if entry['digest'] != digest:
                raise RuntimeError('Конфиг изменился: используйте новый ключ/запуск.')
            if entry['status'] != 'done':
                raise RuntimeError('Предыдущий запрос мог выполниться. Сверьте кабинет/API; не повторяйте POST вслепую.')
            return entry['response']
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        state[key] = {'digest': digest, 'status': 'pending'}
        self.journal.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        result = self.request(method, path, json=payload, params=params)
        state[key].update(status='done', response=result)
        self.journal.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        return result

    def all_items(self, path, params=None):
        query = dict(params or {}, limit=100, sort='id')
        rows = []
        while True:
            page = self.request('GET', path, params=query)
            items = page.get('items', [])
            rows.extend(items)
            if not page.get('has_more'):
                return rows
            if not items or query.get('id_gt') == items[-1]['id']:
                raise RuntimeError('Некорректная пагинация API')
            query['id_gt'] = items[-1]['id']


def require_sources(frame):
    if frame.empty or frame.sample_id.duplicated().any():
        raise ValueError('Нужны непустые уникальные sample_id')
    for url in frame.source_url:
        parts = urlparse(str(url))
        if (parts.scheme != 'https' or not parts.hostname or parts.username
                or parts.hostname in {'localhost', '127.0.0.1', 'example.com'}):
            raise ValueError('Заполните реальные HTTPS source_url; локальные пути не подходят.')


def view_spec(with_hint=False):
    # Same UI in both arms except one hint block. Never prefill the answer.
    source = {'type': 'data.input', 'path': 'source_url'}
    status = {'type': 'data.output', 'path': 'status'}
    items = [{'type': 'view.link', 'url': source, 'content': 'Открыть источник'}]
    if with_hint:
        items.append({'type': 'view.alert', 'theme': 'info', 'label': 'Найдено моделью',
                      'content': {'type': 'view.text',
                                  'content': {'type': 'data.input', 'path': 'hint'}}})
    items += [
        {'type': 'field.radio-group', 'data': status,
         'validation': {'type': 'condition.required'},
         'options': [{'value': 'ok', 'label': 'Цена определена'},
                     {'value': 'no_price', 'label': 'На странице цены нет'},
                     {'value': 'uncertain', 'label': 'Не удалось определить'}]},
        {'type': 'helper.if',
         'condition': {'type': 'condition.equals', 'data': status, 'to': 'ok'},
         'then': {'type': 'field.number', 'label': 'Цена в рублях',
                  'data': {'type': 'data.output', 'path': 'price'},
                  'minimum': 0.01, 'validation': {'type': 'condition.required'}}},
    ]
    config = {'view': {'type': 'view.list', 'items': items},
              'plugins': [{'type': 'plugin.toloka', 'layout': {'kind': 'scroll', 'taskWidth': 650}}]}
    lock = {'core': '1.7.0', 'view.list': '1.0.5', 'field.radio-group': '1.1.10',
            'condition.required': '1.1.5', 'plugin.toloka': '1.1.8',
            'view.link': '1.0.0', 'view.text': '1.0.0', 'field.number': '1.0.0',
            'helper.if': '1.0.0', 'condition.equals': '1.0.0'}
    if with_hint:
        lock['view.alert'] = '1.0.0'
    return {'type': 'tb', 'config': json.dumps(config, ensure_ascii=False), 'lock': lock,
            'settings': {'showSkip': True, 'showTimer': True, 'showSubmit': True,
                         'showInstructions': True, 'showFinish': True}}


def flatten_assignments(assignments, condition):
    import pandas as pd
    rows = []
    for a in assignments:
        if a['status'] not in {'SUBMITTED', 'ACCEPTED', 'REJECTED'}:
            continue
        tasks, solutions = a.get('tasks', []), a.get('solutions', [])
        if len(tasks) != len(solutions):
            raise ValueError('Количество заданий и ответов различается')
        for task, solution in zip(tasks, solutions):
            out = solution.get('output_values', {})
            raw_price = out.get('price') if out.get('status') == 'ok' else None
            valid = (out.get('status') in {'no_price', 'uncertain'} or
                     (out.get('status') == 'ok' and type(raw_price) in (int, float)
                      and math.isfinite(raw_price) and raw_price > 0))
            seconds = ((pd.Timestamp(a['submitted']) - pd.Timestamp(a['created'])).total_seconds()
                       if a.get('submitted') and a.get('created') else None)
            rows.append({'sample_id': task['input_values']['sample_id'], 'condition': condition,
                         'assignment_id': a['id'], 'assignment_status': a['status'],
                         'performer': a['user_id'], 'status': out.get('status'),
                         'price': raw_price if valid else None, 'valid_answer': valid,
                         'duration_sec': seconds})
    return pd.DataFrame(rows, columns=['sample_id', 'condition', 'assignment_id',
        'assignment_status', 'performer', 'status', 'price', 'valid_answer', 'duration_sec'])


def load_saved_hints(run_dir, inputs, version):
    from seminar_utils import parse_hint
    from two_stage_hints import visible_hint
    p = Path(run_dir)
    manifest = json.loads((p / 'manifest.json').read_text())
    raw = [json.loads(s) for s in (p / 'predictions.jsonl').read_text().splitlines()]
    rows = [r for r in raw if r['prompt_version'] == version]
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Дубликаты ответов модели')
    by_id = {r['sample_id']: r for r in rows}
    selected = []
    for item in inputs:
        r = by_id[item['sample_id']]
        if r['image_sha256'] != item['image_sha256']:
            raise ValueError('Изображение не совпадает с модельным прогоном')
        if not r.get('parsed') or r.get('error'):
            raise ValueError('Есть ошибочный ответ модели; не пропускайте объект молча.')
        parsed = parse_hint(json.dumps(r['parsed']))
        selected.append({'sample_id': item['sample_id'], 'hint': visible_hint(parsed),
                         'model_price': parsed['price'], 'model_status': parsed['status']})
    return selected, manifest
