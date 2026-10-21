"""Frozen-budget attention PPO using admitted native batch1 policy tables."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ..acquisition.search_budget import mixed_budget_env, attention_observation
from ..acquisition.search_env import rollout_baseline
from ..data.artifacts import read, sha, write
from ..data.search_native import load_native_search
from .search_training import make_core, assert_observation, compare_nested
from .statistics import paired_interval


SEARCH = dict(neighbors=6, translation_scale_m=.1, rotation_scale_deg=15., query_cost=.01, motion_cost=0.)


def environment(data, initial, starts, budgets=(4,8,16), training=False):
    core = make_core(data, initial, 8, SEARCH)
    field = data.accel_members[0,initial].mean(-1) if training else None
    return mixed_budget_env(core, starts, budgets, full_accel=field)


def checkpoint_gate(model, path, data):
    from ..selectors.search_attention_ppo import AttentionPPO
    restored = AttentionPPO.load(path,device="cpu")
    if (not compare_nested(model.policy.state_dict(),restored.policy.state_dict())
            or not compare_nested(model.policy.optimizer.state_dict(),restored.policy.optimizer.state_dict())):
        raise ValueError("Checkpoint state or optimizer mismatch")
    paths = []
    for agent in (model,restored):
        episodes = []
        for budget in (4,8,16):
            env = environment(data,data.development[0],[1])
            obs,_ = env.reset(seed=91,options={"start":1,"budget":budget})
            actions = []
            while True:
                assert_observation(obs)
                action,_ = agent.predict(obs,deterministic=True,action_masks=env.action_masks())
                actions.append(int(action))
                obs,_,done,truncated,info = env.step(int(action))
                if done or truncated:
                    break
            episodes.append(dict(budget=budget,actions=actions,visited=info["visited"],selected=info["selected"]))
            env.close()
        paths.append(episodes)
    if paths[0] != paths[1] or restored.policy.field_provider is not None:
        raise ValueError("Deployment actor restoration/isolation failed")
    return dict(parameters_exact=True,optimizer_exact=True,deployment_paths_exact=True,
                no_deployment_privileged_provider=True,paths=paths[0])


def train_seed(data, plan, output, seed, steps):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ..selectors.search_attention_ppo import build_attention_model

    class Audit(BaseCallback):
        def __init__(self):
            super().__init__(); self.checked = 0; self.budgets = set()

        def _on_step(self):
            action = np.asarray(self.locals["actions"]).reshape(-1)
            masks = np.asarray(self.locals["action_masks"],bool)
            if (not masks[np.arange(len(action)),action].all()
                    or not np.isfinite(self.locals["rewards"]).all()
                    or not torch.isfinite(self.locals["values"]).all()):
                raise ValueError("Invalid sampled action, reward or critic")
            self.checked += len(action)
            with (output / "episodes.jsonl").open("a") as stream:
                for slot,(done,info) in enumerate(zip(self.locals["dones"],self.locals["infos"])):
                    if done:
                        row = dict(info,step=int(self.model.num_timesteps),initial=int(data.development[slot]))
                        row.pop("terminal_observation",None)
                        self.budgets.add(row["budget"])
                        self.model.audit_episodes.append(row)
                        stream.write(json.dumps(row,allow_nan=False) + "\n")
            return True

    output.mkdir(exist_ok=False)
    env = DummyVecEnv([lambda i=i:environment(data,i,range(1,65),training=True) for i in data.development])
    ppo = {k:plan["training"][k] for k in ("n_steps","batch_size","n_epochs","learning_rate",
             "gamma","gae_lambda","clip_range","ent_coef","vf_coef","max_grad_norm")}
    model = build_attention_model(env,seed,ppo)
    model.audit_output,model.audit_episodes = str(output),[]
    actor = {k:v.clone() for k,v in model.policy.actor.state_dict().items()}
    critic = {k:v.clone() for k,v in model.policy.critic.state_dict().items()}
    callback = Audit(); start=time.monotonic()
    try:
        model.learn(total_timesteps=steps,callback=callback,progress_bar=False)
        if model.num_timesteps != steps or callback.checked != steps or callback.budgets != {4,8,16}:
            raise ValueError("Training budget or mixture coverage failed")
        if compare_nested(actor,model.policy.actor.state_dict()) or compare_nested(critic,model.policy.critic.state_dict()):
            raise ValueError("Actor/critic did not update")
        path=output/"final.zip"; model.save(path)
        gate=checkpoint_gate(model,path,data)
        torch.save(dict(state_dict=model.policy.actor.state_dict(),visual_dim=data.visual.shape[-1],
                        score_scale=data.score_scales[0],noise_count=2,search=SEARCH),output/"actor.pt")
        if seed==0:
            controls=[]
            model.policy.actor.eval()
            for initial in range(0,32,4):
                core=make_core(data,initial,8,SEARCH);core.reset(start=65);actions=[]
                while not core.done:
                    observation=attention_observation(core,core.observation())
                    tensor={k:torch.as_tensor(v[None],dtype=torch.float32) for k,v in observation.items()}
                    with torch.inference_mode():
                        action=int(model.policy.actor(tensor).argmax(-1).item())
                    actions.append(action);core.step(action)
                controls.append(dict(initial=initial,actions=actions,visited=core.info()['visited'],selected=core.best))
            write(output/'deployment-controls.json',dict(controls=controls,scope='CPU actor runtime parity, no VLA calls'))
        receipt=dict(seed=seed,steps=steps,checked_actions=callback.checked,budgets=sorted(callback.budgets),
            actor_updated=True,critic_updated=True,elapsed_seconds=time.monotonic()-start,
            checkpoint_sha256=sha(path),actor_sha256=sha(output/"actor.pt"),checkpoint_admission=gate)
        write(output/"receipt.json",receipt)
        return receipt
    finally:
        env.close()


def evaluate(data,outcomes,output):
    from ..selectors.search_attention_ppo import AttentionPPO
    models={s:AttentionPPO.load(output/f"seed-{s}"/"final.zip",device="cpu") for s in (0,1,2)}
    rows=[]
    with (output/"evaluation.jsonl").open("x") as stream:
        for budget in (4,8,16):
            for initial in data.evaluation:
                for start in range(65,97):
                    methods=[("stay",0),("local_accel",0)]+[(m,s) for m in ("random","ppo") for s in (0,1,2)]
                    for method,seed in methods:
                        if method=="ppo":
                            env=environment(data,initial,[start])
                            obs,_=env.reset(seed=10000+initial*100+start,options={"start":start,"budget":budget})
                            while True:
                                assert_observation(obs)
                                action,_=models[seed].predict(obs,deterministic=True,action_masks=env.action_masks())
                                obs,_,done,truncated,info=env.step(int(action))
                                if done or truncated:
                                    break
                            env.close()
                        else:
                            core=make_core(data,initial,budget,SEARCH)
                            info=rollout_baseline(core,start,method,seed=seed+initial*100000+start*100)
                        label=outcomes.paired(0,initial,start,info["selected"])
                        if label["paired_repeats"] != 32:
                            raise ValueError("Missing evaluation label; never impute failure")
                        row=dict(info,method=method,seed=seed,initial=initial,start=start,budget=budget,
                                 task=data.states[initial]["task_id"],selected_candidate=info["selected"],
                                 accel_gap=(info["best_accel"]-data.accel_members[0,initial].mean(-1).min())/data.score_scales[0])
                        row.update(label);row["success"]=label["selected"]
                        rows.append(row);stream.write(json.dumps(row,allow_nan=False)+"\n")
                stream.flush()
                print(f"evaluation budget={budget} initial={initial} complete",flush=True)
    summary=[]
    metrics=("success","stay","gain","rescued_fraction","harmed_fraction","queries","total_reward","best_accel","accel_gap")
    for budget in (4,8,16):
        for method in ("stay","local_accel","random","ppo"):
            group=[r for r in rows if r["budget"]==budget and r["method"]==method]
            measured={}
            for metric in metrics:
                per_initial=np.asarray([np.mean([r[metric] for r in group if r["initial"]==i]) for i in data.evaluation])
                measured[metric]={**paired_interval(per_initial,np.arange(16)),"per_initial":per_initial.tolist()}
            summary.append(dict(method=method,budget=budget,metrics=measured,
                per_seed={str(s):{k:float(np.mean([r[k] for r in group if r["seed"]==s])) for k in metrics}
                          for s in sorted({r["seed"] for r in group})}))
    comparisons=[]
    for budget in (4,8,16):
        ppo=next(r for r in summary if r["budget"]==budget and r["method"]=="ppo")
        for method in ("stay","local_accel","random"):
            baseline=next(r for r in summary if r["budget"]==budget and r["method"]==method)
            comparisons.append(dict(budget=budget,baseline=method,ppo_minus_baseline={k:paired_interval(
                np.asarray(ppo["metrics"][k]["per_initial"])-baseline["metrics"][k]["per_initial"],np.arange(16))
                for k in ("success","queries","total_reward","accel_gap")}))
    result=dict(rows=summary,comparisons=comparisons,evaluation_episodes=len(rows),
        ppo_episodes=sum(r["method"]=="ppo" for r in rows),
        scope="New batch1 scores and native SigLIP; historical complete-task labels, previously inspected known-task initials")
    write(output/"summary.json",result)
    return result


def run(source,native,inputs,plan_file,output,*,execute=False):
    if not execute:
        raise ValueError("Explicit --execute required")
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=read(plan_file)
    if (plan["training"]["smoke_steps"] != 2048 or plan["training"]["seeds"] != [0,1,2]
            or plan["training"]["steps_per_seed"] != 32768 or plan["search"]["training_budgets"] != [4,8,16]):
        raise ValueError("Only the frozen one-round training budget is authorized")
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        data,outcomes=load_native_search(source,native,inputs)
        write(output/"frozen-plan.json",dict(plan=plan,source_hashes=data.source_hashes,
            native_path=str(native),score_scale=data.score_scales[0],development=data.development,
            evaluation=data.evaluation,source=str(source),inputs=str(inputs),
            runtime_device="cpu",cpu_threads=4,success_labels_training=False,
            resource_contract="Single CPU process, 16 logical environments, sequential seeds; no old queue"))
        smoke=train_seed(data,plan,output/"smoke",17,2048)
        write(output/"smoke-admission.json",dict(status="PASS",**smoke))
        trained=[train_seed(data,plan,output/f"seed-{seed}",seed,32768) for seed in (0,1,2)]
        result=evaluate(data,outcomes,output)
        for path,digest in data.source_hashes.items():
            if sha(path)!=digest:
                raise ValueError("Input changed during training: "+path)
        write(output/"receipt.json",dict(status="COMPLETE",total_decisions=100352,
             trained_seeds=trained,ppo_evaluation_episodes=result["ppo_episodes"],new_vla_calls=0,
             new_simulator_rollouts=0,checkpoint_selection="final_only"))
    except BaseException as error:
        write(output/"failure.json",dict(status="FAILED_STOPPED",error=repr(error)))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","native","inputs","plan","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    run(args.source,args.native,args.inputs,args.plan,args.output,execute=args.execute)
