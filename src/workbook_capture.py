"""URL -> fresh browser captures. No credentials, stealth or captcha bypass."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd


def validate_url(url):
    p = urlsplit(url)
    if p.scheme != 'https' or not p.hostname or p.username or p.password:
        raise ValueError('Нужен публичный HTTPS URL без логина/пароля.')
    if p.hostname in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('Используйте публичную страницу магазина.')
    return url


async def capture_urls(urls, root, viewport=None):
    from playwright.async_api import async_playwright
    urls = [validate_url(str(u).strip()) for u in urls]
    if not urls or len(set(urls)) != len(urls):
        raise ValueError('Нужен непустой список уникальных URL.')
    viewport = viewport or {'width': 1440, 'height': 1200}
    run = Path(root) / ('capture_' + uuid.uuid4().hex[:12])
    run.mkdir(parents=True, exist_ok=False)
    records = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            for url in urls:
                sample_id = hashlib.sha256(url.encode()).hexdigest()[:16]
                row = dict(sample_id=sample_id, source_url=url,
                           captured_at=datetime.now(timezone.utc).isoformat(),
                           viewport=viewport, error=None)
                # Isolated session per URL; never reuse a personal browser profile.
                context = await browser.new_context(viewport=viewport, locale='ru-RU',
                    timezone_id='Europe/Moscow', device_scale_factor=1)
                try:
                    page = await context.new_page()
                    response = await page.goto(url, wait_until='domcontentloaded', timeout=45000)
                    row.update(final_url=page.url, http_status=response.status if response else None)
                    if response is None or response.status >= 400:
                        raise RuntimeError(f'Страница не загружена: HTTP {row["http_status"]}')
                    await page.locator('body').wait_for(state='visible', timeout=15000)
                    # Bounded settle time, not a guarantee that all dynamic widgets loaded.
                    await page.wait_for_timeout(2000)
                    html, text = await page.content(), await page.locator('body').inner_text()
                    html_path, text_path = run / f'{sample_id}.html', run / f'{sample_id}.txt'
                    html_path.write_text(html, encoding='utf-8')
                    text_path.write_text(text, encoding='utf-8')
                    viewport_path = run / f'{sample_id}_viewport.png'
                    full_path = run / f'{sample_id}_full.png'
                    await page.screenshot(path=str(viewport_path), full_page=False)
                    await page.screenshot(path=str(full_path), full_page=True)
                    row.update(title=await page.title(), html_path=str(html_path.resolve()),
                        text_path=str(text_path.resolve()), html_chars=len(html), text_chars=len(text),
                        image_path=str(viewport_path.resolve()), full_image_path=str(full_path.resolve()),
                        image_sha256=hashlib.sha256(viewport_path.read_bytes()).hexdigest(),
                        screenshot_bytes=viewport_path.stat().st_size)
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                finally:
                    await context.close()
                records.append(row)
                (run / 'captures.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
                print(sample_id, 'OK' if row['error'] is None else row['error'])
        finally:
            await browser.close()
    frame = pd.DataFrame(records)
    gold = frame[['sample_id']].copy()
    gold['image_sha256'] = frame.get('image_sha256', '')
    gold['reference_status'] = ''
    gold['reference_price'] = ''
    gold['reviewed_by'] = ''
    gold.to_csv(run / 'gold.csv', index=False)
    return frame, run


def capture_inputs(frame):
    if frame.empty or frame.error.notna().any():
        raise ValueError('Есть ошибки захвата: проверьте таблицу. Не исключайте их молча.')
    for row in frame.itertuples():
        if hashlib.sha256(Path(row.image_path).read_bytes()).hexdigest() != row.image_sha256:
            raise ValueError('Снимок изменился после захвата; обновите набор и gold.')
    return frame[['sample_id','image_path','image_sha256']].to_dict('records')


def read_gold(path, inputs):
    gold = pd.read_csv(path, dtype=str).fillna('')
    expected = pd.DataFrame(inputs)[['sample_id','image_sha256']]
    if gold.sample_id.duplicated().any() or set(gold.sample_id) != set(expected.sample_id):
        raise ValueError('Gold должен содержать ровно по одной записи для каждого снимка.')
    aligned = gold.merge(expected, on='sample_id', suffixes=('', '_expected'), validate='one_to_one')
    if not aligned.image_sha256.eq(aligned.image_sha256_expected).all():
        raise ValueError('Gold относится к другим снимкам.')
    if not gold.reviewed_by.str.strip().ne('').all():
        raise ValueError('Заполните gold.csv вручную по инструкции, независимо от модели.')
    if not gold.reference_status.isin(['ok','no_price']).all():
        raise ValueError('Gold: ok/no_price. Неоднозначную карточку сначала разберите отдельно.')
    price = pd.to_numeric(gold.reference_price, errors='coerce')
    import numpy as np
    is_price = gold.reference_status.eq('ok')
    if not (price[is_price].gt(0) & np.isfinite(price[is_price])).all():
        raise ValueError('Для gold ok требуется положительная конечная цена.')
    if gold.loc[~is_price,'reference_price'].ne('').any():
        raise ValueError('Для no_price оставьте цену пустой, не ноль.')
    return gold.assign(reference_price=price,reference_no_price=~is_price).to_dict('records')


def show_capture(row):
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax, key, title in zip(axes,['image_path','full_image_path'],
                             ['Видимая область → вход VLM','Вся страница → аудит полноты']):
        with Image.open(row[key]) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(row['sample_id']+' — '+row.get('title',''),fontsize=11)
    return fig


def hint_gallery(inputs, predictions, render_hint):
    """Always show the exact input image; escape model/page text, never execute it."""
    import base64
    import html
    from IPython.display import HTML
    by_id = {r['sample_id']: r for r in predictions}
    cards = []
    for item in inputs:
        result = by_id[item['sample_id']]
        if result['image_sha256'] != item['image_sha256']:
            raise ValueError('Картинка и подсказка относятся к разным прогонам.')
        raw = Path(item['image_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['image_sha256']:
            raise ValueError('Изображение на диске изменилось.')
        encoded = base64.b64encode(raw).decode()
        parsed = result.get('parsed')
        hint = render_hint(parsed) if parsed and not result.get('error') else 'Ошибка формата — подсказка не готова'
        cards.append(f'''<section style="display:flex;gap:20px;border:1px solid #ddd;padding:16px;margin:12px 0;align-items:flex-start">
        <a href="data:image/png;base64,{encoded}" target="_blank"><img src="data:image/png;base64,{encoded}" style="max-width:520px;width:100%;max-height:650px;object-fit:contain"></a>
        <div style="min-width:220px;flex:1"><h3>{html.escape(item['sample_id'])}</h3>
        <p><b>{html.escape(hint)}</b></p><p>Время генерации: {result.get('latency_sec',0):.1f} сек.</p>
        <details><summary>Ответ модели и ошибка формата</summary><pre style="white-space:pre-wrap">{html.escape(result.get('raw',''))}\n{html.escape(str(result.get('error')))}</pre></details></div></section>''')
    return HTML(''.join(cards))
