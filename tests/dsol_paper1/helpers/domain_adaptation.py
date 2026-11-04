"""Synthetic domain inputs; no execution permission or feature admission."""
from pathlib import Path
import json
from AlphaBrain.research.dsol.data import domain_adaptation as domain
from AlphaBrain.research.dsol.execution.manifest import file_digest


def contract(root, campaign, schedule, initial):
    root=Path(root)
    def save(name, value):
        path=root/name;path.write_text(json.dumps(value));return str(path)
    visual=save('shared-visual.json',dict(synthetic=True,dimension=4608))
    identity=save('domain.json',dict(schema='SEARCH_ADAPTATION_DOMAIN_V1',visual_dim=4608,
        visual_contract=visual,**{k:campaign[k] for k in
            ('sim_config','training_inputs','training_noise','operation_model','model_sha256')}))
    paths=[identity,visual,schedule,str(initial),campaign['sim_config'],
           campaign['training_inputs'],campaign['training_noise']]
    return save('domain-protocol.json',dict(schema=domain.SCHEMA,arms=list(domain.ARMS),
        seeds=[0,1,2],updates=8,episodes_per_update=8,query_cap=8,query_cost=.01,visual_dim=4608,
        domain=identity,schedules={str(s):schedule for s in (0,1,2)},
        source_checkpoints={str(s):dict(path=str(initial),visual_contract_sha256=file_digest(visual)) for s in (0,1,2)},
        input_hashes={p:file_digest(p) for p in paths}))
