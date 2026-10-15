"""Finish-time planning across models; never dispatch from a stale estimate."""
import math


def plan_allocation(work, current, total_gpus, *, qualified, minimum_gain=.15):
    """work is remaining GPU-seconds at the admitted workers/GPU setting.

    Exhaustive integer allocation is small for a few models and GPUs. A change
    is a proposal only; queue locks, host admission and disjoint ownership must
    still be checked by the handoff adapter. Qualification excludes cold pilots.
    """
    if type(total_gpus) is not int or total_gpus<1 or set(work)!=set(current) or set(work)!=set(qualified):
        raise ValueError('Matching models and positive GPU budget required')
    if any(not math.isfinite(v) or v<0 for v in work.values()) or any(type(v) is not int or v<0 for v in current.values()):
        raise ValueError('Finite nonnegative work and integer GPU allocation required')
    if sum(current.values())!=total_gpus or not 0<=minimum_gain<1:
        raise ValueError('Allocation must cover the whole GPU budget')
    models=sorted(m for m in work if work[m]>0)
    if not models:return dict(status='COMPLETE',allocation=current,change_recommended=False)
    if len(models)>total_gpus:raise ValueError('Insufficient physical GPUs')
    options=[]
    def enumerate_alloc(index,left,partial):
        model=models[index]
        if index==len(models)-1:
            allocation={m:0 for m in work};allocation.update(partial);allocation[model]=left
            end=max(work[m]/allocation[m] for m in models)
            movement=sum(abs(allocation[m]-current[m]) for m in work)
            options.append((end,movement,allocation));return
        for n in range(1,left-(len(models)-index-1)+1):
            enumerate_alloc(index+1,left-n,{**partial,model:n})
    enumerate_alloc(0,total_gpus,{})
    end,_,allocation=min(options,key=lambda x:(x[0],x[1],tuple(x[2].values())))
    current_end=max((work[m]/current[m] if current[m] else float('inf')) for m in models)
    gain=1-end/current_end
    measured=all(qualified[m] for m in models)
    return dict(status='QUALIFIED' if measured else 'WAITING_FOR_REPRESENTATIVE_TIMING',
                allocation=allocation,current_allocation=current,
                current_remaining_hours=current_end/3600 if math.isfinite(current_end) else None,
                proposed_remaining_hours=end/3600,relative_gain=gain,
                per_model_current_hours={m:work[m]/current[m]/3600 if current[m] else None for m in models},
                change_recommended=measured and gain>=minimum_gain,
                scope='Minimize final makespan, not equal trial counts; proposal is not authorization to duplicate or interrupt work')
