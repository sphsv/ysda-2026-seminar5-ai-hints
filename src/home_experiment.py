"""Optional, explicit paid reproduction. Never called by the classroom notebook."""
import copy

def stage_pair(api,baseline_spec,assisted_spec,pool_settings,baseline_rows,assisted_rows,
               cohort_filters,overlap=3,budget_rub=0,commission=.30):
    """Caller must supply genuinely disjoint audience filters; no universal cohort IDs."""
    if set(cohort_filters)!={'baseline','assisted'} or cohort_filters['baseline']==cohort_filters['assisted']:
        raise ValueError('Нужны явно заданные фильтры непересекающихся групп; одинаковые фильтры недопустимы')
    if not baseline_rows or len(baseline_rows)!=len(assisted_rows):raise ValueError('Несовпадающие входы')
    a={x['sample_id']:x['source_url'] for x in baseline_rows}
    b={x['sample_id']:x['source_url'] for x in assisted_rows}
    if a!=b or len(a)!=len(baseline_rows):raise ValueError('Нужны одинаковые уникальные URL и ID')
    if overlap<1:raise ValueError('Некорректное перекрытие')
    cost=2*len(a)*overlap*pool_settings['reward_per_assignment']*(1+commission)
    if cost>budget_rub or budget_rub<=0:raise ValueError(f'Нужно подтвердить бюджет не менее {cost:.2f} ₽ с указанной комиссией')
    staged={}
    for arm,spec,rows in [('baseline',baseline_spec,baseline_rows),('assisted',assisted_spec,assisted_rows)]:
        project=api.write_once(arm+'_project','POST','/api/v1/projects',copy.deepcopy(spec))
        config=copy.deepcopy(pool_settings);config.update(project_id=project['id'],filter=cohort_filters[arm])
        config['private_name']+=' / '+arm
        config['defaults']={'default_overlap_for_new_task_suites':overlap}
        pool=api.write_once(arm+'_pool','POST','/api/v1/pools',config)
        fields=['sample_id','source_url']+(['hint'] if arm=='assisted' else [])
        tasks=[dict(pool_id=pool['id'],input_values={k:x[k] for k in fields},overlap=overlap) for x in rows]
        api.write_once(arm+'_tasks','POST','/api/v1/tasks',tasks,
                       params={'async_mode':'false','skip_invalid_items':'false','open_pool':'false'})
        staged[arm]=pool['id']
    return {'pools':staged,'planned_cost_rub':cost}

def request_launch(api,pool_id,budget_rub):
    """Moderation may automatically start paid labeling. Explicit separate confirmation."""
    checks=api.request('GET',f'/api/new/requester/pools/{pool_id}/validate')
    blockers=[c for c in checks if c['active'] and c['blocker']]
    if any(c['type']!='MUST_PASS_MODERATION' for c in blockers):
        raise RuntimeError('Устраните блокирующие проверки в UI перед запуском')
    if input(f'Проверьте UI, аудиторию и смету {budget_rub:g} ₽. Для платного запуска введите START: ').strip()!='START':
        raise RuntimeError('Запуск не подтверждён')
    if blockers:return api.write_once(str(pool_id)+'_moderation','PUT',f'/api/new/requester/poolModeration/{pool_id}',{'status':'READY'})
    return api.write_once(str(pool_id)+'_open','POST',f'/api/v1/pools/{pool_id}/open')
