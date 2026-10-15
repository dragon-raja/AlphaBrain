from AlphaBrain.research.dsol.data.artifacts import write
from benchmarks.Robocasa365.cluster_balance import observe


def test_timing_balance_ignores_success_labels(tmp_path):
    models={}
    for model in ('official','xr1'):
        root=tmp_path/model;root.mkdir()
        choice=root/'choice.json';write(choice,dict(task='task',evaluation_views=['v','w']))
        plan=root/'plan.json';write(plan,dict(selections={model:{'i':dict(path=str(choice))}},execution_repeats=[0,1],task_horizons=dict(task=100)))
        trial=root/'results/cells/i/v/execution_0.json'
        write(trial,dict(model=model,initial='i',steps=100,seconds=10,success=False))
        models[model]=dict(plan=str(plan),results=[str(root/'results')],workers_per_gpu=2,gpus=2)
    config=tmp_path/'config.json'
    write(config,dict(models=models,total_gpus=4,minimum_gain=.15,minimum_task_trials=1,
        minimum_model_trials=1,minimum_observed_tasks=1,startup_seconds_per_trial=0,output=str(tmp_path/'result.json')))
    before=observe(config)
    write(tmp_path/'official/results/cells/i/v/execution_0.json',dict(model='official',initial='i',steps=100,seconds=10,success=True))
    after=observe(config)
    assert before['recommendation']==after['recommendation']
    assert before['models']==after['models']
    assert not after['recommendation']['change_recommended']
