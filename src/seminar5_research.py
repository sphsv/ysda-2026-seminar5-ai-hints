"""Three connected measures for the seminar. No network or synthetic fallback."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARMS=['baseline','assisted']
LABELS=['Baseline','AI hint']
COLORS=['#6479AF','#D98559']


def load_experiment(folder):
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text())
    if not meta['complete']:
        raise RuntimeError('Новый эксперимент ещё не завершён. Обновите выгрузку после 300 ответов в каждом пуле; подмена старыми или синтетическими ответами не выполняется.')
    cards=pd.read_csv(folder/'cards.csv',dtype={'sample_id':str})
    answers=pd.read_csv(folder/'answers.csv',dtype={'sample_id':str,'worker':str})
    return score(answers,cards),cards,meta


def load_snapshot(folder):
    """Read the actual dated snapshot, retaining an explicit completion status."""
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text())
    cards=pd.read_csv(folder/'cards.csv',dtype={'sample_id':str})
    answers=pd.read_csv(folder/'answers.csv',dtype={'sample_id':str,'worker':str})
    if answers.empty:raise RuntimeError('В выгрузке ещё нет ответов людей.')
    return score(answers,cards),cards,meta


def score(answers,cards):
    required=['sample_id','reference_price','hint_price','hint_origin','trap_type']
    if cards.sample_id.duplicated().any():raise ValueError('Повтор эталона')
    if answers.response_id.duplicated().any():raise ValueError('Повтор ответа')
    if set(answers.condition)!=set(ARMS):raise ValueError('Нужны обе группы')
    if answers.groupby('worker').condition.nunique().gt(1).any():raise ValueError('Исполнитель попал в обе группы')
    d=answers.merge(cards[required],on='sample_id',validate='many_to_one')
    if len(d)!=len(answers):raise ValueError('Ответ без эталона')
    for col in ['reference_price','hint_price']:
        if not (np.isfinite(d[col]) & d[col].gt(0)).all():raise ValueError('Некорректная цена в эталоне/подсказке')
    price=pd.to_numeric(d.price,errors='coerce')
    d['numeric']=d.status.eq('ok') & np.isfinite(price) & price.gt(0)
    d['price']=price.where(d.numeric)
    d['correct']=d.numeric & d.price.sub(d.reference_price).abs().lt(.005)
    d['hint_wrong']=d.hint_price.sub(d.reference_price).abs().ge(.005)
    d['matches_hint']=d.numeric & d.price.sub(d.hint_price).abs().lt(.005)
    d['duration_sec']=pd.to_numeric(d.duration_sec,errors='coerce')
    d['duration_sec']=d.duration_sec.where(np.isfinite(d.duration_sec) & d.duration_sec.gt(0))
    return d


def summary(d):
    rows=[]
    for arm in ARMS:
        g=d[d.condition.eq(arm)];bad=g[g.hint_wrong]
        rows.append(dict(condition=arm,answers=len(g),workers=g.worker.nunique(),
            cards=g.sample_id.nunique(),timed=g.duration_sec.notna().sum(),
            median_seconds=g.duration_sec.median(),accuracy=g.correct.mean(),
            wrong_hint_answers=len(bad),wrong_hint_match_rate=bad.matches_hint.mean()))
    return pd.DataFrame(rows).set_index('condition')


def cluster_interval(d,column,subset='all',repeats=2000,seed=42):
    """Worker-cluster percentile interval, conditional on selected cards.

    Correct/adoption: difference of answer-weighted rates; time: ratio of medians.
    Whole workers resampled within arm, never answers independently.
    """
    source=d[d.hint_wrong] if subset=='wrong' else d
    rng=np.random.default_rng(seed);draws=[];groups=[];observed=[]
    for arm in ARMS:
        g=source[source.condition.eq(arm)].dropna(subset=[column])
        values=[x[column].to_numpy(dtype=float) for _,x in g.groupby('worker')]
        if len(values)<2:return dict(effect=np.nan,low=np.nan,high=np.nan)
        groups.append(values)
        observed.append(float(g[column].median() if column=='duration_sec' else g[column].mean()))
    def effect(a,b):return b/a-1 if column=='duration_sec' else b-a
    for _ in range(repeats):
        stat=[]
        for values in groups:
            sample=np.concatenate([values[i] for i in rng.integers(0,len(values),len(values))])
            stat.append(float(np.median(sample) if column=='duration_sec' else sample.mean()))
        draws.append(effect(*stat))
    low,high=np.quantile(draws,[.025,.975])
    return dict(effect=effect(*observed),low=float(low),high=float(high))


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)


def plot_time(d):
    from matplotlib.ticker import PercentFormatter
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    for arm,label,color in zip(ARMS,LABELS,COLORS):
        v=np.sort(d.loc[d.condition.eq(arm),'duration_sec'].dropna())
        if not len(v):continue
        ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=color,lw=2.5,
                label=f'{label} · медиана {np.median(v):.1f} с · n={len(v)}')
    ax.axhline(.5,color='#999999',ls=':',lw=1)
    ax.set(xscale='log',xlabel='Время от выдачи до отправки, секунды (логарифмическая шкала)',
           ylabel='Доля ответов, завершённых к этому времени',ylim=(0,1.02),title='Изменилось ли время выполнения?')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False);clean(ax);return fig


def effect_panel(d):
    """Three conditional effects; overall result is secondary, all values computed."""
    panels=[]
    for title,mask,column,color in [
        ('Верная подсказка: правильность',~d.hint_wrong,'correct','#277d6d'),
        ('Неверная подсказка: правильность',d.hint_wrong,'correct','#b04843'),
        ('Неверная подсказка: совпадение с её ценой',d.hint_wrong,'matches_hint','#b04843'),
    ]:
        g=d.loc[mask];rates=g.groupby('condition')[column].mean()
        a,b=rates['baseline'],rates['assisted']
        panels.append(f'<div style="flex:1;min-width:230px;border:1px solid {color};border-top:5px solid {color};padding:18px;border-radius:10px">'
                      f'<div>{title}</div><div style="font-size:26px;margin:12px 0">{a:.1%} → {b:.1%}</div>'
                      f'<b style="color:{color}">{100*(b-a):+.1f} п.п.</b><div>{g.sample_id.nunique()} карточек · Baseline → AI hint</div></div>')
    rates=d.groupby('condition').correct.mean()
    card_rows=d.drop_duplicates('sample_id')
    good=int((~card_rows.hint_wrong).sum());bad=int(card_rows.hint_wrong.sum())
    return ('<div style="display:flex;gap:12px;flex-wrap:wrap">'+''.join(panels)+'</div>'
            f'<div style="padding:16px;background:#f1f3f7;margin-top:12px;border-radius:10px">'
            f'На всём пуле: <b>{rates["baseline"]:.1%} → {rates["assisted"]:.1%}</b> '
            f'({100*(rates["assisted"]-rates["baseline"]):+.1f} п.п.). '
            f'Итог относится к смеси {good} верных / {bad} неверных подсказок, не к любому проекту.</div>')


def plot_quality(d):
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    masks=[np.ones(len(d),dtype=bool),~d.hint_wrong,d.hint_wrong]
    for j,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
        rates=[d.loc[mask & d.condition.eq(arm),'correct'].mean()*100 for mask in masks]
        xs=np.arange(3)+(j-.5)*.32
        ax.bar(xs,rates,width=.30,color=color,label=label)
        for x,y in zip(xs,rates):ax.text(x,y+1,f'{y:.1f}%',ha='center',fontsize=10)
    ax.set(xticks=range(3),xticklabels=['Все карточки','Верная подсказка','Неверная подсказка'],
           ylim=(0,112),ylabel='Правильных ответов, %',title='Общий результат и две разные ситуации')
    ax.legend(frameon=False,loc='upper center',ncols=2);clean(ax);return fig


def quality_breakdown(d):
    rows=[]
    for title,mask in [('Весь пул',pd.Series(True,index=d.index)),
                       ('Карточки с верной подсказкой',~d.hint_wrong),
                       ('Карточки с неверной подсказкой',d.hint_wrong)]:
        g=d[mask];a=g[g.condition.eq('baseline')];b=g[g.condition.eq('assisted')]
        rows.append({'Срез':title,'Карточек':g.sample_id.nunique(),
                     'Ответов без AI':len(a),'Ответов с AI':len(b),
                     'Верных без AI':int(a.correct.sum()),'Верных с AI':int(b.correct.sum()),
                     'Без подсказки, %':100*a.correct.mean(),'С подсказкой, %':100*b.correct.mean(),
                     'Разница, п.п.':100*(b.correct.mean()-a.correct.mean())})
    return pd.DataFrame(rows).set_index('Срез')


def plot_quality_balance(d):
    """Exact additive decomposition only with equal exposure on the same cards."""
    counts=d.groupby(['sample_id','condition']).size().unstack().reindex(columns=ARMS)
    if counts.isna().any().any() or not counts.baseline.eq(counts.assisted).all():
        raise ValueError('Для разложения нужны одинаковые карточки и число ответов в группах')
    t=quality_breakdown(d);n=int(t.iloc[0]['Ответов без AI'])
    delta=t['Верных с AI']-t['Верных без AI']
    good,bad,total=100*delta.iloc[1]/n,100*delta.iloc[2]/n,100*delta.iloc[0]/n
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    ax.bar([0,1,2],[good,bad,total],bottom=[0,good,0],color=['#3A9C89','#D76D67','#6479AF'],width=.58)
    for i,start,value,count in zip(range(3),[0,good,0],[good,bad,total],[delta.iloc[1],delta.iloc[2],delta.iloc[0]]):
        ax.text(i,start+value/2,f'{value:+.1f} п.п.\n{count:+d} ответов',ha='center',va='center',color='white',weight='bold')
    ax.axhline(0,color='#aaa',lw=1)
    ax.set(xticks=range(3),xticklabels=['Вклад верных подсказок','Вклад неверных подсказок','Изменение на всём пуле'],
           ylabel='Вклад в общую правильность, п.п.',title='Почему в сумме качество выросло?')
    clean(ax);return fig


def plot_adoption(d):
    bad=d[d.hint_wrong];fig,ax=plt.subplots(figsize=(9,3.2),layout='constrained')
    for y,arm in enumerate(ARMS):
        g=bad[bad.condition.eq(arm)]
        rates=[g.correct.mean(),g.matches_hint.mean(),(~g.correct & ~g.matches_hint).mean()]
        left=0
        for rate,color,label in zip(rates,['#3A9C89','#D76D67','#CBD1DD'],
                                   ['Верный ответ','Та же неверная цена','Другая ошибка / отказ']):
            ax.barh(y,rate*100,left=left,color=color,label=label if y==0 else None,height=.48)
            if rate>.04:ax.text(left+rate*50,y,f'{rate:.0%}',ha='center',va='center')
            left+=rate*100
    ax.set(yticks=[0,1],yticklabels=LABELS,xlim=(0,100),xlabel='Доля всех ответов на 33 карточках, %',
           title='Выбирают ли люди конкретную ошибочную цену чаще?')
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.22),ncols=3);clean(ax);return fig


def case_ranking(d):
    counts=d.groupby(['sample_id','condition']).size().unstack().reindex(columns=ARMS).fillna(0)
    complete=counts.index[counts.eq(3).all(axis=1)]
    b=d[d.hint_wrong & d.sample_id.isin(complete)]
    table=b.groupby(['sample_id','condition']).matches_hint.mean().unstack().reindex(columns=ARMS)
    table['difference']=table.assisted-table.baseline
    return table.sort_values(['difference','sample_id'],ascending=[False,True])


def click_report(d):
    """Persisted output fields, not a claim about navigation success or attention."""
    values=d.get('source_clicked',pd.Series(pd.NA,index=d.index,dtype='string')).astype('string').str.lower()
    rows=[]
    for arm,label in zip(ARMS,LABELS):
        v=values[d.condition.eq(arm)];n=len(v)
        yes=int(v.eq('true').sum());no=int(v.eq('false').sum());unknown=n-yes-no
        rows.append({'Группа':label,'Ответов':n,'Клик записан':yes,'Явное false':no,'Неизвестно':unknown,
                     'Доля с записанным кликом':yes/n if n else np.nan,
                     'Верхняя граница при пропусках':(yes+unknown)/n if n else np.nan})
    return pd.DataFrame(rows).set_index('Группа')


def plot_case(d,cards,sid,root):
    from PIL import Image
    r=cards.set_index('sample_id').loc[sid];g=d[d.sample_id.eq(sid)]
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained',gridspec_kw={'width_ratios':[1.15,1]})
    with Image.open(Path(root)/r.image_path) as im:axes[0].imshow(im)
    axes[0].axis('off');axes[0].set_title('Снимок, использованный при проверке эталона')
    for i,(arm,color) in enumerate(zip(ARMS,COLORS)):
        v=g[g.condition.eq(arm)]
        offsets=np.linspace(-.12,.12,len(v))
        axes[1].scatter(v.price/float(r.reference_price),i+offsets,s=65,color=color,alpha=.8)
        missing=v.price.isna().sum()
        if missing:axes[1].text(.02,.92-i*.08,f'{LABELS[i]}: {missing} ответов без цены',transform=axes[1].transAxes,fontsize=9)
    axes[1].axvline(1,color='#3A9C89',ls='--',label=f'Эталон {r.reference_price:g} ₽')
    axes[1].axvline(r.hint_price/r.reference_price,color='#D76D67',ls=':',label=f'Подсказка {r.hint_price:g} ₽')
    axes[1].set(yticks=[0,1],yticklabels=LABELS,ylim=(-.4,1.4),xlabel='Ответ / эталонная цена',
                title='Одна точка — один числовой ответ')
    axes[1].legend(frameon=False);clean(axes[1]);return fig


def conclusion(d,intervals):
    s=summary(d);lines=['### Что получилось в этом эксперименте','']
    table=quality_breakdown(d)
    lines += [f"**Главный результат на всём пуле:** правильность {s.loc['baseline','accuracy']:.1%} → {s.loc['assisted','accuracy']:.1%}, то есть {100*(s.loc['assisted','accuracy']-s.loc['baseline','accuracy']):+.1f} п.п. Это не общее ухудшение качества.", '']
    for title in table.index[1:]:
        r=table.loc[title]
        lines += [f"- **{title} ({int(r['Карточек'])}):** {r['Без подсказки, %']:.1f}% → {r['С подсказкой, %']:.1f}% ({r['Разница, п.п.']:+.1f} п.п.)."]
    counts=d.groupby(['sample_id','condition']).size().unstack().reindex(columns=ARMS)
    if not counts.isna().any().any() and counts.baseline.eq(counts.assisted).all():
        delta=table['Верных с AI']-table['Верных без AI'];n=int(table.iloc[0]['Ответов без AI'])
        lines += ['',f"**Баланс:** {int(delta.iloc[1]):+d} верных ответов на первой подвыборке и {int(delta.iloc[2]):+d} на второй = {int(delta.iloc[0]):+d} из {n} ответов в группе. Помощь верных подсказок перевесила вред неверных в этой выборке. Это разложение групповых итогов, не число людей, изменивших ответ.",'']
    for key,title,units in [('speed','Изменение медианного времени','%'),
                            ('quality','Изменение доли правильных ответов','п.п.'),
                            ('adoption','Изменение совпадения с неверной подсказкой','п.п.')]:
        x=intervals[key]
        lines.append(f"- {title}: **{100*x['effect']:+.1f} {units}**; приближённый 95% интервал [{100*x['low']:+.1f}; {100*x['high']:+.1f}].")
    a=intervals['adoption'];q=intervals['quality']
    if a['low']>0:
        lines+=['','На выбранных карточках показ подсказки увеличил совпадение с конкретной неверной ценой. Это согласуется с переносом ошибок; мы не наблюдаем внутреннюю причину решения каждого человека.']
    else:
        lines+=['','Для переноса ошибок интервал не исключает нулевой эффект: убедительного направленного результата здесь нет. Отсутствие подтверждения не доказывает отсутствие влияния.']
    if q['low']<=0<=q['high']:
        lines+=['Разницу общего качества нельзя уверенно отличить от нуля по этому расчёту. Не называем её доказанным ухудшением или доказанной эквивалентностью.']
    elif q['low']>0:
        lines += [f"При этом общая правильность выросла: {s.loc['baseline','accuracy']:.1%} → {s.loc['assisted','accuracy']:.1%}. Интервал разности выше нуля по выбранному приближённому расчёту. Рост качества в среднем совместим с переносом конкретных ошибок."]
    else:
        lines += ['Общая правильность снизилась; интервал разности ниже нуля по выбранному приближённому расчёту.']
    t=intervals['speed']
    if t['low']<=0<=t['high']:
        lines += [f"Ускорение не установлено: медианы {s.loc['baseline','median_seconds']:.1f} с без подсказки и {s.loc['assisted','median_seconds']:.1f} с с подсказкой; интервал изменения включает ноль."]
    rates=d.groupby(['condition','hint_wrong']).correct.mean()
    if all((arm,True) in rates.index and (arm,False) in rates.index for arm in ARMS):
        lines += [f"На карточках с верной подсказкой правильность: {rates['baseline',False]:.1%} → {rates['assisted',False]:.1%}; с неверной: {rates['baseline',True]:.1%} → {rates['assisted',True]:.1%}. Это описательный разрез, не отдельное доказательство значимости различий."]
    clicks=click_report(d)
    lines += [f"Записанное нажатие источника: {clicks.iloc[0]['Доля с записанным кликом']:.1%} → {clicks.iloc[1]['Доля с записанным кликом']:.1%}. По этим данным нельзя утверждать, что люди с AI реже открывали сайты; пропуски событий и чтение страницы остаются ненаблюдаемыми."]
    majority=majority_report(d)
    lines += [f"После голосования большинства верная цена получена на {majority.iloc[0]['Верное большинство']:.0%} → {majority.iloc[1]['Верное большинство']:.0%} карточек; неверная общая цена — на {majority.iloc[0]['Ошибочное большинство']:.0%} → {majority.iloc[1]['Ошибочное большинство']:.0%}. Согласие само по себе не гарантирует правильность; эти доли описательные."]
    lines+=['','**Практический вывод:** проверка формата JSON не заменяет проверку подсказки. Решение о внедрении принимаем по качеству и времени работы человека, а не по убедительности ответа AI.',
        '', 'Это специально отобранные карточки и контролируемые ошибки подсказок. Интервалы условны на этой выборке; они не описывают все магазины и естественное качество VLM. Эталоны проверены ассистенткой, а не независимыми экспертами.']
    return '\n'.join(lines)


def progress_panel(d, metadata):
    from IPython.display import HTML
    blocks=[]
    for arm,label,color in zip(ARMS,LABELS,COLORS):
        g=d[d.condition.eq(arm)];counts=g.groupby('sample_id').size()
        blocks.append(f'''<section style="flex:1;padding:20px;background:#f5f7fb;border-top:5px solid {color};border-radius:12px">
        <h3>{label}</h3><div style="font-size:32px">{len(g)} / 300 ответов</div>
        <progress value="{len(g)}" max="300" style="width:100%"></progress>
        <p>{g.sample_id.nunique()} карточек затронуто · {(counts==3).sum()} полностью размечено из 100</p>
        <p>{g.worker.nunique()} исполнителей · {g.duration_sec.notna().sum()} ответов с измеренным временем</p></section>''')
    return HTML('<div style="display:flex;gap:16px;flex-wrap:wrap">'+''.join(blocks)+'</div>')


def interval_alpha(units):
    """Krippendorff interval alpha; pairable units only, variable number of ratings."""
    units=[np.asarray(v,dtype=float) for v in units]
    units=[v[np.isfinite(v)] for v in units]
    units=[v for v in units if len(v)>=2]
    if not units:return np.nan
    pooled=np.concatenate(units);n=len(pooled)
    do=sum(np.square(v[:,None]-v[None,:]).sum()/(len(v)-1) for v in units)/n
    de=2*np.var(pooled,ddof=1)
    return float(1-do/de) if de>0 else np.nan


def agreement_report(d):
    counts=d[d.numeric].groupby(['sample_id','condition']).size().unstack().reindex(columns=ARMS).fillna(0)
    eligible=counts.index[counts.ge(2).all(axis=1)]
    rows=[]
    for arm,label in zip(ARMS,LABELS):
        g=d[d.condition.eq(arm) & d.sample_id.isin(eligible) & d.numeric]
        units=[np.log(x.price/x.reference_price).to_numpy() for _,x in g.groupby('sample_id')]
        groups=list(g.groupby('sample_id'))
        pairs=sum(len(x)*(len(x)-1)//2 for _,x in groups)
        same=sum(sum(abs(a-b)<.005 for i,a in enumerate(x.price) for b in list(x.price)[i+1:]) for _,x in groups)
        rows.append({'Группа':label,'Карточек ≥2 числовых ответа в обеих группах':len(eligible),
                     'Числовых ответов':len(g),'α (log-отклонение)':interval_alpha(units),
                     'Совпадение пар':same/pairs if pairs else np.nan})
    return pd.DataFrame(rows).set_index('Группа')


def majority_report(d):
    counts=d.groupby(['sample_id','condition']).size().unstack().reindex(columns=ARMS).fillna(0)
    ids=counts.index[counts.eq(3).all(axis=1)]
    rows=[]
    for arm,label in zip(ARMS,LABELS):
        groups=list(d[d.condition.eq(arm)&d.sample_id.isin(ids)].groupby('sample_id'))
        right=wrong=absent=0
        for _,g in groups:
            # Kopeck-valued exact majority; missing/non-price responses cannot vote for a price.
            prices=g.loc[g.numeric,'price'].map(lambda x:round(float(x)*100)).value_counts()
            if prices.empty or prices.iloc[0]<2:absent+=1
            elif abs(prices.index[0]/100-g.reference_price.iloc[0])<.005:right+=1
            else:wrong+=1
        n=len(groups)
        rows.append({'Группа':label,'Общих полностью размеченных карточек':n,
                     'Верное большинство':right/n if n else np.nan,
                     'Ошибочное большинство':wrong/n if n else np.nan,
                     'Нет большинства по цене':absent/n if n else np.nan})
    return pd.DataFrame(rows).set_index('Группа')


def tolerance_rates(d,tolerance):
    if tolerance<0:raise ValueError('Допуск не может быть отрицательным')
    valid=d.numeric & (d.price.sub(d.reference_price).abs().lt(.005) |
                       d.price.div(d.reference_price).sub(1).abs().le(tolerance))
    return [float(valid[d.condition.eq(arm)].mean()) for arm in ARMS]


def tolerance_explorer(d,output):
    """Self-contained Plotly slider, also usable offline in a browser without a kernel."""
    import plotly.graph_objects as go
    from IPython.display import HTML
    grid=[0,.001,.005,.01,.02,.05,.10,.20]
    rates=np.array([tolerance_rates(d,t) for t in grid])*100
    fig=go.Figure()
    for i,(label,color) in enumerate(zip(LABELS,COLORS)):
        fig.add_trace(go.Scatter(x=np.array(grid)*100,y=rates[:,i],mode='lines+markers',name=label,line_color=color,
            hovertemplate='Допуск %{x:.1f}%<br>Доля ответов %{y:.1f}%<extra>%{fullData.name}</extra>'))
    def annotations(i):
        return [dict(x=.5,y=1.15,xref='paper',yref='paper',showarrow=False,
                     text=f'Допуск {grid[i]:.1%}: без AI {rates[i,0]:.1f}% · с AI {rates[i,1]:.1f}% · Δ {rates[i,1]-rates[i,0]:+.1f} п.п.')]
    def shape(t):return [dict(type='line',x0=t*100,x1=t*100,y0=0,y1=100,line=dict(color='#555',dash='dot'))]
    steps=[dict(method='relayout',label=f'{t*100:g}%',args=[{'annotations':annotations(i),'shapes':shape(t)}]) for i,t in enumerate(grid)]
    fig.update_layout(template='plotly_white',height=520,margin=dict(t=100,b=100),
        title='Насколько результат зависит от допустимой ошибки?',annotations=annotations(0),shapes=shape(0),
        sliders=[dict(steps=steps,currentvalue=dict(prefix='Допуск: '),pad=dict(t=45))],
        xaxis_title='Допустимое относительное отклонение, %',yaxis_title='Доля всех ответов в пределах допуска, %',
        yaxis_range=[0,101],legend=dict(orientation='h',y=1.02))
    body=fig.to_html(full_html=True,include_plotlyjs=True)
    Path(output).write_text(body)
    return HTML(fig.to_html(full_html=False,include_plotlyjs=True))


def synthesis(d,intervals,metadata):
    """Dashboard with explicit denominators; no manufactured final conclusion."""
    from IPython.display import HTML
    from html import escape
    s=summary(d);a=agreement_report(d);m=majority_report(d)
    def pct(x):return '—' if pd.isna(x) else f'{100*x:.1f}%'
    rows=[]
    definitions=[('Скорость','Медиана времени, с','median_seconds','speed'),
                 ('Качество','Точная цена, % всех ответов','accuracy','quality'),
                 ('Механизм','Совпадение с неверной подсказкой, % ответов на этих карточках','wrong_hint_match_rate','adoption')]
    for question,label,col,key in definitions:
        values=[f'{s.loc[arm,col]:.1f}' if col=='median_seconds' else pct(s.loc[arm,col]) for arm in ARMS]
        z=intervals[key];unit='%' if key=='speed' else 'п.п.'
        interval=f"{100*z['effect']:+.1f} {unit} [{100*z['low']:+.1f}; {100*z['high']:+.1f}]"
        rows.append([question,label,*values,interval])
    rows.append(['Согласие','α log-отклонений (только парные числовые ответы)',
                 *['—' if pd.isna(v) else f'{v:.3f}' for v in a['α (log-отклонение)']],'Описательный показатель, не качество'])
    rows.append(['Агрегация','Ошибочное большинство (общие полные тройки)',
                 *[pct(v) for v in m['Ошибочное большинство']],'Даже согласие 2 из 3 не гарантирует правильность'])
    header=''.join(f'<th style="padding:12px;text-align:left">{x}</th>' for x in ['Вопрос','Что измеряем',*LABELS,'Изменение / интерпретация'])
    body=''.join('<tr>'+''.join(f'<td style="padding:12px;border-bottom:1px solid #ddd">{escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)
    title='Итоги эксперимента' if metadata['complete'] else 'Промежуточная сводка — не окончательный вывод'
    note=('Сначала проверяем интервал переноса ошибки, затем смотрим цену в качестве и времени. ' if metadata['complete'] else
          f"Собрано {len(d)} из 600 ответов. Группы ещё различаются по покрытию карточек; знак и величина эффектов могут измениться. ")
    tiles=[]
    for key,label,unit,description in [
        ('speed','Время','%','Изменение медианного времени'),
        ('quality','Правильность','п.п.','Изменение доли точных цен'),
        ('adoption','Повторение ошибки','п.п.','Изменение совпадений с неверной подсказкой')]:
        z=intervals[key]
        status='Промежуточная оценка' if not metadata['complete'] else ('Интервал включает ноль' if z['low']<=0<=z['high'] else 'Интервал не включает ноль')
        tiles.append(f'<section style="flex:1;min-width:190px;padding:20px;background:#f0f4fa;border-radius:14px"><div>{label}</div><div style="font-size:32px;font-weight:700;margin:12px 0">{100*z["effect"]:+.1f} {unit}</div><div>{description}</div><p style="color:#52627a">95%: [{100*z["low"]:+.1f}; {100*z["high"]:+.1f}]<br>{status}</p></section>')
    message=('Повторение конкретной неверной цены увеличилось: это согласуется с переносом ошибки подсказки.'
             if intervals['adoption']['low']>0 else 'Направленный перенос ошибки пока не подтверждён выбранным интервальным расчётом.')
    if metadata['complete']:
        message=f"Общая правильность: {s.loc['baseline','accuracy']:.1%} → {s.loc['assisted','accuracy']:.1%}. "+message
        if intervals['speed']['low']<=0<=intervals['speed']['high']:
            message+=' Убедительного ускорения не установлено.'
    if not metadata['complete']:message='Сбор ещё идёт. Сейчас можно описать ответы, но нельзя фиксировать окончательный эффект.'
    return HTML(f'<h3>{title}</h3><div style="display:flex;gap:12px;flex-wrap:wrap">'+''.join(tiles)+
        f'</div><p style="padding:18px;border-left:5px solid #6479AF;background:#f5f7fb;font-size:18px">{message}</p>'+
        f'<details><summary>Полная сводка: значения групп, согласие и большинство</summary><table style="width:100%;border-collapse:collapse"><thead style="background:#eaf0f8"><tr>{header}</tr></thead><tbody>{body}</tbody></table></details><p>{note}Для времени показана экономия, для долей — AI минус baseline. Высокое согласие не заменяет проверку правильности.</p>')
