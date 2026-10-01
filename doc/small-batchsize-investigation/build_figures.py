#!/usr/bin/env python3
"""Build vector figures from portable extracts; --refresh rereads campaign evidence.

No training, scheduler mutation, checkpoint access, or campaign writes occur.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, shutil, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'; ASSETS=HERE/'assets'
BATCHES=[16,32,64,128]
COLORS={'sync':'#0072B2','clipped':'#8B61B3','unclipped':'#D55E00'}
LABELS={'sync':'Synchronous','clipped':'Decentralized, clipped','unclipped':'Decentralized, no clipping'}
PREFIX='runs/packed4-awc-exponential-20m-c512-small-batch'
ROOTS={'sync':'runs/sync-20m-c512-small-batch','clipped':PREFIX+'-unified', 'clipped-base':PREFIX,'clipped-extension':PREFIX+'-beta2-extension','ablation':PREFIX+'-clipping','unclipped':PREFIX+'-no-clipping'}
SOURCE_IDS={'sync':'S1','clipped':'S2','clipped-base':'S2','clipped-extension':'S2','audit':'S3','ablation':'S4','unclipped':'S5'}
FIGSIZE=(28/2.54,8/2.54)

def load(p): return json.loads(Path(p).read_text())
def save(p,o): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(o): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def csvsave(p,rows):
    if not rows:return
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)

def refresh(repo):
    """Pin published authenticated records plus original manifests/import evidence."""
    sources={};extracts=[]
    def register(path,key):
        path=Path(path)
        if not path.is_absolute():path=repo/path
        rel=str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path)
        sources[rel]={'id':SOURCE_IDS[key],'label':key,'path':rel,'sha256':sha(path),'bytes':path.stat().st_size}
        return path
    def extract(key,rel,name):
        p=register(Path(ROOTS[key])/rel,key);q=DATA/name;shutil.copyfile(p,q)
        extracts.append({'file':'data/'+name,'sha256':sha(q),'sources':[sources[str(p.relative_to(repo))]['path']],'selection':'complete source file; bytes unchanged'})
        return load(q)
    for key in ['sync','clipped','ablation','unclipped']:
        extract(key,'reports/runs.json',key+'-runs.json')
        extract(key,'reports/validation.json',key+'-validation.json')
    for key in ['sync','clipped-base','clipped-extension','ablation','unclipped']:
        extract(key,'manifest.json',key+'-manifest.json')
    for key in ['clipped','unclipped']:
        for name in ['replicated','screening','selected-recipes','performance-summary']:
            extract(key,'reports/'+name+'.json',key+'-'+name+'.json')
    for name in ['matched-screening','matched-replicated','supplemental-imports','synchronous-selected']:
        extract('unclipped','reports/'+name+'.json',('unclipped-supplemental' if name=='supplemental-imports' else name)+'.json')
    for name in ['beta1-replicated','confirmation','throughput-summary']:
        extract('sync','reports/'+name+'.json','sync-'+name+'.json')
    extract('clipped','reports/unified-campaign.json','clipped-protocol.json')
    extract('clipped','reports/source-verification.json','clipped-source-verification.json')
    extract('clipped','reports/input-evidence.json','clipped-input-evidence.json')
    extract('ablation','reports/clipping-comparison.json','ablation-comparison.json')
    for name in ['selected-clipping','synchronous-screen-clipping','clipping-runs','summary','input-evidence']:
        p=Path(ROOTS['ablation'])/'clipping-audit'/f'{name}.json'
        if (repo/p).exists():
            q=register(p,'audit');t=DATA/('audit-'+name+'.json');shutil.copyfile(q,t)
            extracts.append({'file':str(t.relative_to(HERE)),'sha256':sha(t),'sources':[str(p)],'selection':'complete source file; bytes unchanged'})
    imports=extract('unclipped','provenance/imports.json','unclipped-imports.json')
    extract('unclipped','provenance/screening-selection.json','unclipped-selection.json')
    for key in ['clipped-base','clipped-extension']:
        for rel in ['provenance/screening-selection.json','provenance/protocol.json']:
            if (repo/ROOTS[key]/rel).exists():extract(key,rel,key+'-'+Path(rel).name)
    # Check published table hashes against each campaign's completed collection record.
    for key in ['sync','clipped','ablation','unclipped']:
        validation=load(DATA/(key+'-validation.json'))
        assert validation.get('complete',validation.get('all_successful',False)),key
        for entry in extracts:
            original=entry['sources'][0]
            if str(Path(original).parent)==str(Path(ROOTS[key])/'reports'):
                expected=validation.get('output_hashes',{}).get(Path(original).name)
                if expected is not None:assert entry['sha256']==expected,original
    # Authentication remains rooted in original receipts, configuration and artifact hashes.
    for entry in imports:
        assert entry['expected_config']['optimizer']['grad_clip'] is None
        assert entry['row']['scheduler']=={'exit_code':'0:0','state':'COMPLETED'}
        for path,expected in entry['files'].items():
            q=register(path,'ablation');assert sources[str(q.relative_to(repo))]['sha256']==expected,path
    # Copy source/configuration/dependency identities; frozen manifests pin these bytes.
    for key in ['sync','clipped-base','clipped-extension','unclipped','ablation']:
        m=load(DATA/(key+'-manifest.json'))
        for rel,expected in m['source_files'].items():
            source_path=Path(ROOTS[key])/'source'/rel
            if not (repo/source_path).exists(): source_path=Path(ROOTS[key])/rel
            q=register(source_path,key)
            assert sha(q)==expected, str(q)
    save(HERE/'provenance.json',{'schema_version':1,'as_of':'2026-10-01','sources':list(sources.values()),'extracts':extracts,
        'authentication':'Previously authenticated campaign reports; original frozen source identities and all twelve no-clipping import evidence hashes rechecked on refresh.',
        'no_experiments_executed':True})

def normalized(r,method):
    r=copy.deepcopy(r); r['method']=method;r['workers']=1 if method=='sync' else 4
    r['local_batch']=r['batch']//r['workers'];r['grad_clip']=None if method=='unclipped' else 1.0
    r.setdefault('weight_decay',0.1)
    for beta in ['beta1','beta2']:
        u=math.log(.5)/math.log(r[beta]) if r[beta]>0 else 0.0;r[beta+'_half_life_updates']=u
        r[beta+'_global_half_life_targets']=u*r['batch']*512
        r[beta+'_per_worker_half_life_targets']=u*r['batch']*512/r['workers']
        r.setdefault(beta+'_boundary',False)
    r.setdefault('lr_boundary',False)
    if 'losses' in r:
        assert len(r['losses'])==3
        r['seeds']=[42,43,44];r['seed42'],r['seed43'],r['seed44']=r['losses']
        r['mean_loss']=statistics.mean(r['losses']);r['std_loss']=statistics.stdev(r['losses'])
        r['replication_mean_loss']=statistics.mean(r['losses'][1:])
    return r

def analyze():
    """Derived records contain no references required at build time outside this directory."""
    prov=load(HERE/'provenance.json');m={k:load(DATA/(k+'-manifest.json')) for k in ['sync','clipped-base','clipped-extension','ablation','unclipped']}
    replicated={k:[normalized(r,k) for r in load(DATA/(k+'-replicated.json'))] for k in ['clipped','unclipped']}
    sync=load(DATA/'synchronous-selected.json')
    replicated['sync']=[normalized(r,'sync') for r in sync]
    for r in replicated['sync']:
        chosen=m['sync']['beta1_replication']['recipes'][f"b{r['batch']}-wd0.1"]
        for key in ['lr_boundary','beta1_boundary']:r[key]=chosen.get(key,False)
    winners={k:[min((r for r in rows if r['batch']==b),key=lambda r:(r['mean_loss'],r['lr'],r['beta1'],r['beta2'])) for b in BATCHES] for k,rows in replicated.items()}
    screening={k:[normalized(r,k) for r in load(DATA/(k+'-screening.json'))] for k in ['clipped','unclipped']}
    raw={k:[normalized(r,k) for r in load(DATA/(k+'-runs.json'))] for k in ['sync','clipped','unclipped']}
    supplemental=[normalized(r,'unclipped') for r in load(DATA/'unclipped-supplemental.json')]
    raw['unclipped_all']=raw['unclipped']+supplemental
    # Import deduplication uses the frozen full configuration identity (includes clipping/seed).
    assert len(raw['unclipped_all'])==524
    assert len({(r['recipe_identity'],r['seed'],r['grad_clip']) for r in raw['unclipped_all']})==524
    imports=load(DATA/'unclipped-imports.json')
    config_evidence=[]
    specmap={}
    for key in ['sync','clipped-base','clipped-extension','ablation','unclipped']:
        for spec in m[key]['runs']+m[key].get('historical',[]):
            if 'config' in spec:specmap[(spec['recipe_identity'],spec['seed'])]=(key,spec)
    for method in ['sync','clipped','unclipped_all']:
        for r in raw[method]:
            pair=specmap.get((r.get('recipe_identity'),r['seed']))
            if pair:
                key,spec=pair;cfg=copy.deepcopy(spec['config']);cfg['runtime'].pop('output_dir',None)
                config_evidence.append({'method':'unclipped' if method=='unclipped_all' else method,'recipe':r['recipe'],'seed':r['seed'],'recipe_identity':r['recipe_identity'],'normalized_config_sha256':digest(cfg),'config_sha256':spec.get('config_sha256'),'config':cfg,'source_campaign':key,'primary':r in raw.get('unclipped',[]) if method=='unclipped_all' else True})
    save(DATA/'configuration-identities.json',config_evidence)
    selectedaudit=load(DATA/'audit-selected-clipping.json')
    clipping={}
    for method in ['sync','clipped']:
        rows=[]
        for win in winners[method]:
            rr=[r for r in raw[method] if r['recipe']==win['recipe'] and r['seed'] in [42,43,44] and abs(r['loss']-win['losses'][r['seed']-42])<1e-12]
            ff=[r['clipping_frequency'] for r in rr if r.get('clipping_frequency') is not None]
            audit=next((r for r in selectedaudit if r['recipe']==win['recipe'] and r['method']==('Synchronous' if method=='sync' else 'Decentralized')), {})
            row={**audit,**{k:win[k] for k in ['batch','local_batch','recipe','lr','beta1','beta2','workers']},'method':method,'clipping_frequency':statistics.mean(ff) if ff else None,'seed_clipping_frequencies':ff,'available_seeds':[r['seed'] for r in rr if r.get('clipping_frequency') is not None],'steps_per_seed':408092672//(win['batch']*512)}
            row['denominator_per_seed']=row['steps_per_seed']*row['workers']
            if ff and not row.get('clip_counts'):row['clip_counts']=[round(f*row['denominator_per_seed']) for f in ff]
            row['local_clipping_frequencies']=[r.get('local_clipping_frequencies') for r in rr]
            rows.append(row)
        clipping[method]=rows
    # Existing performance tables match the current environment; retain their original definitions.
    performance=[]
    for row in load(DATA/'clipped-performance-summary.json'):
        q=copy.deepcopy(row);q['method']='sync' if row['method']=='Synchronous' else 'clipped';q['source_id']='S2';performance.append(q)
    for row in load(DATA/'unclipped-performance-summary.json'):
        if row['method']=='No clipping':q=copy.deepcopy(row);q['method']='unclipped';q['source_id']='S5';performance.append(q)
    matched=load(DATA/'matched-screening.json');matchedstats=[]
    for b in BATCHES:
        rr=[r for r in matched if r['batch']==b];dd=[r['loss_difference'] for r in rr]
        matchedstats.append({'batch':b,'n':len(rr),'median_delta':statistics.median(dd),'mean_delta':statistics.mean(dd),'clipped_wins':sum(d>0 for d in dd),'unclipped_wins':sum(d<0 for d in dd),'ties':sum(d==0 for d in dd),'minimum_delta':min(dd),'maximum_delta':max(dd)})
    gaps=[]
    for s,c,u in zip(winners['sync'],winners['clipped'],winners['unclipped']):
        d=c['mean_loss']-s['mean_loss'];gaps.append({'batch':s['batch'],'clipped_minus_sync':d,'perplexity_increase_percent':100*math.expm1(d),'unclipped_minus_clipped':u['mean_loss']-c['mean_loss'],'unclipped_minus_sync':u['mean_loss']-s['mean_loss']})
    confirmation=[]
    for r in load(DATA/'sync-confirmation.json'):
        if r['weight_decay']==.1:confirmation.append({**r,**m['sync']['confirmation']['0.1'],'seeds':[45,46,47]})
    syncgrid=[]
    groups=defaultdict(list)
    for r in raw['sync']:
        if r['weight_decay']==.1 and r['seed']==42:groups[(r['batch'],r['stage'])].append(r)
    for (b,stage),rr in sorted(groups.items()):syncgrid.append({'batch':b,'stage':stage,'n':len(rr),'lrs':sorted({r['lr'] for r in rr}),'beta1s':sorted({r['beta1'] for r in rr}),'beta2s':sorted({r['beta2'] for r in rr}),'joint_configs':[{k:r[k] for k in ['lr','beta1','beta2']} for r in rr]})
    protocol={'shared':{'parameters':20403520,'context':512,'training_targets':408092672,'warmup_targets':20447232,'validation_targets':197411295,'epochs':40,'warmup_updates':{'16':2496,'32':1248,'64':624,'128':312},'updates':{'16':49816,'32':24908,'64':12454,'128':6227},'epsilon':1e-8,'weight_decay':.1,'checkpoint_policy':'none','training_tokens_per_parameters':20.001092,'epoch_tokens_per_parameters':.5000273,'validation_batch':128,'amp':'BF16','parameters_and_moments':'FP32','schedule':'cosine to zero','cache_identity':m['unclipped']['cache_identity'],'dataset':m['unclipped']['base_config']['data'],'model':m['unclipped']['base_config']['model']},'clipped':load(DATA/'clipped-protocol.json'),'unclipped':m['unclipped']['protocol'],'sync':{'observed_seed42_grid':syncgrid,'beta1_followup':m['sync']['beta1_replication'],'confirmation':m['sync']['confirmation']},'ablation':m['ablation']['protocol']}
    for key in ['sync','clipped-base','clipped-extension','ablation','unclipped']:
        protocol.setdefault('identities',{})[key]={k:m[key].get(k) for k in ['source_hash','cache_identity','versions','version','selection_sha256','protocol_sha256']}
    beta2profiles=[]
    for b in BATCHES:
        rr=[r for r in screening['clipped'] if r['batch']==b]
        for beta2 in sorted({r['beta2'] for r in rr}):
            r=min((r for r in rr if r['beta2']==beta2),key=lambda r:r['loss']);beta2profiles.append({k:r[k] for k in ['batch','beta2','beta1','lr','loss','recipe']})
    importledger=[]
    primary_ids={(r['recipe_identity'],r['seed']) for r in raw['unclipped']}
    for entry in imports:
        r=entry['row'];importledger.append({'recipe':r['recipe'],'seed':r['seed'],'recipe_identity':r['recipe_identity'],'grad_clip':None,'loss':r['loss'],'primary':(r['recipe_identity'],r['seed']) in primary_ids,'account':entry['account'],'job_key':r['job_key'],'artifact_hashes':entry['artifact_hashes']})
    data={'schema_version':1,'as_of':'2026-10-01','winners':winners,'replicated':replicated,'screening':screening,'matched_screening':matched,'matched_replicated':load(DATA/'matched-replicated.json'),'matched_summary':matchedstats,'ablation':load(DATA/'ablation-comparison.json'),'ablation_seeds':load(DATA/'ablation-runs.json'),'clipping_selected':clipping,'performance':performance,'confirmation':confirmation,'protocols':protocol,'beta2_profiles':beta2profiles,'imports':importledger,'supplemental':supplemental,'gaps':gaps,'sources':prov['sources'],'callouts':{'clipped_sync_gap_min':min(r['clipped_minus_sync'] for r in gaps),'clipped_sync_gap_max':max(r['clipped_minus_sync'] for r in gaps),'clipped_sync_ppl_percent_min':min(r['perplexity_increase_percent'] for r in gaps),'clipped_sync_ppl_percent_max':max(r['perplexity_increase_percent'] for r in gaps)},'counts':{'clipped_observations':920,'clipped_screening':840,'clipped_replicated':40,'unclipped_primary':520,'unclipped_screening':480,'unclipped_replicated':20,'unclipped_supplemental':4,'unclipped_unique_observations':524,'imported_observations':12,'imported_primary':sum(r['primary'] for r in importledger)},'figure_metadata':{name:{'batch_ticks':BATCHES if name in ['quality','gap','beta1','clipping','matched-grid','retuned','throughput'] else None,'vector':True,'symlog_linear_threshold':.025 if name=='matched-grid' else None} for name in ['quality','gap','beta1','beta2','clipping','ablation','matched-grid','retuned','throughput','replication']}}
    add_heatmap_data(data)
    save(DATA/'analysis.json',data)
    for key in ['matched_summary','gaps','ablation','performance']:csvsave(DATA/(key.replace('_','-')+'.csv'),data[key])
    for method in ['sync','clipped','unclipped']:csvsave(DATA/('replicated-'+method+'.csv'),replicated[method])
    csvsave(DATA/'matched-screening.csv',matched)
    prov['derived']=[{'file':str(p.relative_to(HERE)),'sha256':sha(p)} for p in sorted(DATA.glob('*')) if p.is_file() and str(p.relative_to(HERE)) not in {e['file'] for e in prov['extracts']}]
    save(HERE/'provenance.json',prov)
    return data

def add_heatmap_data(data):
    """Retain every measured seed-42 cell; do not combine later conditional stages."""
    cells=[]
    figures=[]
    for method in ['clipped','unclipped']:
        for batch in BATCHES:
            rows=[r for r in data['screening'][method] if r['batch']==batch]
            minimum=min(r['loss'] for r in rows)
            name=f'heatmap-{method}-b{batch}'
            panels=[]
            for beta1 in sorted({r['beta1'] for r in rows}):
                subset=[r for r in rows if r['beta1']==beta1]
                panels.append({'batch':batch,'beta1':beta1,'lrs':sorted({r['lr'] for r in subset}),
                               'beta2s':sorted({r['beta2'] for r in subset}),
                               'minimum_loss':minimum,'cell_count':len(subset)})
            figures.append({'name':name,'method':method,'batch':batch,'scope':'complete seed-42 Cartesian screening grid',
                            'cell_count':len(rows),'panels':panels})
            for r in rows:
                cells.append({**{k:r[k] for k in ['batch','lr','beta1','beta2','seed','recipe','recipe_identity','loss','stage','kind']},
                              'figure':name,'method':method,'minimum_loss':minimum,'delta':r['loss']-minimum,'historical':False})
    # Available original beta1=.9 candidates include the broader historical B128 grid.
    # The current-source reference repeat is excluded rather than replacing its historical twin.
    sync=[r for r in load(DATA/'sync-runs.json') if r['seed']==42 and r['weight_decay']==.1
          and r['beta1']==.9 and r['role']=='candidate' and r['stage'] in ['screen','historical']]
    panels=[]
    for batch in BATCHES:
        rows=[r for r in sync if r['batch']==batch];minimum=min(r['loss'] for r in rows)
        panels.append({'batch':batch,'beta1':.9,'lrs':sorted({r['lr'] for r in rows}),
                       'beta2s':sorted({r['beta2'] for r in rows}),
                       'minimum_loss':minimum,'cell_count':len(rows)})
        for r in rows:
            cells.append({**{k:r[k] for k in ['batch','lr','beta1','beta2','seed','recipe','recipe_identity','loss','stage','kind']},
                          'figure':'heatmap-sync','method':'sync','minimum_loss':minimum,
                          'delta':r['loss']-minimum,'historical':r['kind']=='historical'})
    figures.append({'name':'heatmap-sync','method':'sync','batch':None,'cell_count':len(sync),'panels':panels,
                    'scope':'Available initial beta1=.9 candidates only: current screening plus broad historical B128; later beta1 sensitivity and fresh reference repeat excluded'})
    maximum=max(r['delta'] for r in cells)
    scale={'normalization':'symlog','linthresh':.01,'linscale':1.0,'base':10,'vmin':0.0,'vmax':maximum,
           'cmap':'cividis','saturation':False,'shared_across':'all 1418 cells in nine figures'}
    bundle={'schema_version':1,'color_scale':scale,'figures':figures,'cells':cells,
            'annotation':'Delta rounded to 3 decimals, leading zero omitted below 1; exact zero shown as 0; H suffix marks historical synchronous measurements.',
            'missing_cells':'Unmeasured synchronous combinations are light gray with an em dash; no missing decentralized grid cells.',
            'half_life_label':'HL10M = 2^(-global_batch * 512 / 10000000)',
            'method_colors':COLORS}
    assert len(cells)==1418 and len(sync)==98
    save(DATA/'heatmap-cells.json',bundle)
    for f in figures:
        data['figure_metadata'][f['name']]={'batch_ticks':None,'vector':True,'cell_count':f['cell_count'],
                                          'scope':f['scope'],'color_scale':scale,
                                          'data_file':'data/heatmap-cells.json'}


def heatmap_annotation(cell):
    value=cell['delta']
    text='0' if value==0 else f'{value:.3f}'
    if text.startswith('0.'):text=text[1:]
    return text+('H' if cell['historical'] else '')


def draw_heatmaps(data):
    from matplotlib.colors import SymLogNorm
    from matplotlib.patches import Rectangle
    bundle=load(DATA/'heatmap-cells.json');scale=bundle['color_scale']
    norm=SymLogNorm(linthresh=scale['linthresh'],linscale=scale['linscale'],base=scale['base'],
                   vmin=scale['vmin'],vmax=scale['vmax'])
    cmap=plt.get_cmap(scale['cmap']).copy();cmap.set_bad('#E5E7EB')
    for metadata in bundle['figures']:
        method=metadata['method'];panels=metadata['panels']
        columns=3 if len(panels)==5 else 2
        f,axes=plt.subplots(2,columns,figsize=(27/2.54,13.5/2.54),layout='constrained')
        axes=np.asarray(axes).ravel()
        selected=[r for r in bundle['cells'] if r['figure']==metadata['name']]
        for ax,panel in zip(axes,panels):
            beta2s=panel['beta2s'];lrs=panel['lrs'];batch=panel['batch'];beta1=panel['beta1']
            rows=[r for r in selected if r['batch']==batch and r['beta1']==beta1]
            lookup={(r['lr'],r['beta2']):r for r in rows}
            z=np.full((len(lrs),len(beta2s)),np.nan)
            for j,lr in enumerate(lrs):
                for i,beta2 in enumerate(beta2s):
                    if (lr,beta2) in lookup:z[j,i]=lookup[lr,beta2]['delta']
            im=ax.imshow(np.ma.masked_invalid(z),norm=norm,cmap=cmap,aspect='auto',interpolation='none')
            ax.grid(False)
            labels=['HL10M' if abs(beta-2**(-batch*512/1e7))<1e-12 else f'{beta:.9f}'.rstrip('0') for beta in beta2s]
            ax.set_xticks(range(len(beta2s)),labels,rotation=35,ha='right',fontsize=9.5)
            ax.set_yticks(range(len(lrs)),[f'{lr:g}' for lr in lrs],fontsize=9.5)
            ax.set_xlabel('β₂',fontsize=11,labelpad=1);ax.set_ylabel('Learning rate',fontsize=11,labelpad=2)
            title=f'β₁ = {beta1:g}' if method!='sync' else f'Global batch {batch} · β₁ = .9'
            ax.set_title(title,fontsize=11.5,color=COLORS[method],pad=4)
            for j,lr in enumerate(lrs):
                for i,beta2 in enumerate(beta2s):
                    cell=lookup.get((lr,beta2))
                    if cell is None:ax.text(i,j,'—',ha='center',va='center',fontsize=10,color='#6B7280');continue
                    rgba=cmap(norm(cell['delta']));luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                    ax.text(i,j,heatmap_annotation(cell),ha='center',va='center',fontsize=9.5 if columns==3 else 10.5,
                            color='white' if luminance<.48 else '#111827')
                    if cell['delta']==0:ax.add_patch(Rectangle((i-.48,j-.48),.96,.96,fill=False,edgecolor='#E15759',lw=1.7))
            ax.set_xticks(np.arange(-.5,len(beta2s),1),minor=True);ax.set_yticks(np.arange(-.5,len(lrs),1),minor=True)
            ax.grid(which='minor',color='white',linewidth=.6);ax.tick_params(which='minor',length=0)
            for spine in ax.spines.values():spine.set_visible(False)
        for ax in axes[len(panels):]:
            ax.axis('off')
            minimum=min(r['loss'] for r in selected)
            ax.text(.04,.89,f"Global batch {metadata['batch']}\n{metadata['cell_count']} seed-42 configurations\nMinimum loss: {minimum:.6f}\n\nCell: loss − batch minimum\nOutline: minimum observed cell\nHL10M: 10M-global-target half-life",transform=ax.transAxes,
                    fontsize=10.5,va='top',linespacing=1.45,color='#17324D')
        ticks=[0,.01,.1,1,scale['vmax']]
        cbar=f.colorbar(im,ax=list(axes),orientation='horizontal',fraction=.055,pad=.04,shrink=.88,ticks=ticks,aspect=65)
        cbar.ax.set_xticklabels(['0','.01','.1','1',f"{scale['vmax']:.3f}"],fontsize=10)
        cbar.set_label('Δ validation loss from method/batch minimum · shared scale: linear to .01, logarithmic above',fontsize=10)
        finish(f,metadata['name'])
    # Refresh derived/file hashes after adding the atlas to the original ten figures.
    prov=load(HERE/'provenance.json')
    prov['figures']=[{'file':str(p.relative_to(HERE)),'sha256':sha(p)} for p in sorted(ASSETS.glob('*')) if p.suffix in ['.svg','.pdf']]
    save(HERE/'provenance.json',prov)


# Figures use exact raw values; no smoothing or hidden outcome exclusions.
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.labelsize':16,'axes.titlesize':16,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':13,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#17324D','text.color':'#17324D','axes.edgecolor':'#64748B','axes.grid':True,'grid.color':'#E5E7EB','grid.alpha':.8,'grid.linewidth':.6,'axes.axisbelow':True,'svg.fonttype':'none','svg.hashsalt':'small-batchsize-investigation','pdf.fonttype':42,'savefig.facecolor':'white'})
def fig(n=1):return plt.subplots(1,n,figsize=FIGSIZE,layout='constrained')
def finish(f,name):
    for ext in ['svg','pdf']:f.savefig(ASSETS/f'{name}.{ext}',bbox_inches='tight',pad_inches=.05,metadata={'Creator':'small-batchsize-investigation','CreationDate':None,'ModDate':None} if ext=='pdf' else {'Creator':'small-batchsize-investigation','Date':None})
    plt.close(f)
def batchaxis(ax):
    ax.set_xscale('log',base=2);ax.xaxis.set_major_locator(FixedLocator(BATCHES));ax.set_xticklabels([str(b) for b in BATCHES]);ax.xaxis.set_minor_locator(NullLocator());ax.set_xlim(13,158);ax.set_xlabel('Global batch size')
def lossplot(ax,data,methods):
    for method in methods:
        rr=data['winners'][method];ax.errorbar(BATCHES,[r['mean_loss'] for r in rr],yerr=[r['std_loss'] for r in rr],marker='o',capsize=4,lw=2.1,color=COLORS[method],label=LABELS[method])
    batchaxis(ax);ax.set_ylabel('Validation loss');ax.legend(frameon=False,loc='best',fontsize=12)
def draw(data):
    style();ASSETS.mkdir(exist_ok=True)
    f,a=fig();lossplot(a,data,['sync','clipped']);finish(f,'quality')
    f,a=fig();rr=data['gaps'];a.bar(np.arange(4),[r['clipped_minus_sync'] for r in rr],color=COLORS['clipped'],width=.55)
    a.set_xticks(range(4),BATCHES);a.set_xlabel('Global batch size');a.set_ylabel('Mean loss difference\n(decentralized − sync.)');a.set_ylim(0,.011)
    for i,r in enumerate(rr):a.text(i,r['clipped_minus_sync']+.00035,f"{r['clipped_minus_sync']:.4f}\n+{r['perplexity_increase_percent']:.2f}% PPL",ha='center',fontsize=16)
    finish(f,'gap')
    f,aa=fig(2)
    for method in ['sync','clipped','unclipped']:
        rr=data['winners'][method];opt=dict(marker='o',linestyle='--' if method=='unclipped' else '-',markerfacecolor='none' if method=='unclipped' else COLORS[method],markersize=10 if method=='unclipped' else 6,markeredgewidth=2 if method=='unclipped' else 1);aa[0].plot(BATCHES,[r['beta1'] for r in rr],color=COLORS[method],label=LABELS[method],**opt);aa[1].plot(BATCHES,[r['beta1_global_half_life_targets']/1e6 for r in rr],color=COLORS[method],**opt)
    for a in aa:batchaxis(a)
    aa[0].set_ylabel('Selected β₁');aa[0].set_ylim(.885,1.001);aa[0].legend(frameon=False,fontsize=10,loc='lower left');aa[1].set_ylabel('β₁ half-life\n(million global targets)');finish(f,'beta1')
    f,aa=fig(2)
    bcols=['#0072B2','#8B61B3','#D55E00','#009E73']
    for b,col in zip(BATCHES,bcols):
        rr=[r for r in data['beta2_profiles'] if r['batch']==b];aa[0].plot([1-r['beta2'] for r in rr],[r['loss'] for r in rr],'o-',color=col,label=f'B={b}')
    aa[0].set_xscale('log');aa[0].invert_xaxis();aa[0].set_xlabel('1 − β₂ (closer to 1 →)');aa[0].set_ylabel('Best seed-42 loss');aa[0].legend(frameon=False,fontsize=11,ncol=2);aa[0].set_title('Clipped: minimum over LR and β₁',fontsize=13)
    rr=sorted([r for r in data['replicated']['clipped'] if r['batch']==32 and r['lr']==.004 and r['beta1']==.98],key=lambda r:r['beta2'])
    aa[1].errorbar(range(len(rr)),[r['mean_loss'] for r in rr],yerr=[r['std_loss'] for r in rr],marker='o',capsize=4,color=COLORS['clipped']);aa[1].set_xticks(range(len(rr)),[f"{r['beta2']:.6f}".rstrip('0') for r in rr]);aa[1].set_xlabel('β₂');aa[1].set_ylabel('Three-seed mean ± SD');aa[1].set_title('B=32, LR=.004, β₁=.98',fontsize=13);finish(f,'beta2')
    f,a=fig()
    for method,off in [('sync',-.13),('clipped',.13)]:
        rows=data['clipping_selected'][method];xx=np.arange(4)+off; yy=[r['clipping_frequency']*100 if r['clipping_frequency'] is not None else 0 for r in rows]
        a.bar(xx,yy,width=.25,color=COLORS[method],label=LABELS[method]);
        for x,y,r in zip(xx,yy,rows):a.text(x,y+(6 if method=='clipped' and y<10 else 2),f'{y:.2f}%' if r['clipping_frequency'] is not None else 'missing',ha='center',fontsize=13,rotation=0)
    a.set_xticks(range(4),BATCHES);a.set_xlabel('Global batch size');a.set_ylabel('Worker updates\nclipped (%)');a.set_ylim(0,100);a.legend(frameon=False,loc='upper right');finish(f,'clipping')
    f,a=fig();rr=data['ablation'];xx=np.arange(len(rr));
    a.axhline(0,color='#64748B',lw=1);a.errorbar(xx,[r['mean_loss_difference'] for r in rr],yerr=[r['std_loss_difference'] for r in rr],fmt='D',capsize=6,color=COLORS['unclipped'],markersize=8,label='Mean ± sample SD')
    for i,r in enumerate(rr):a.scatter(np.array([i-.08,i,i+.08]),r['seed_loss_differences'],s=38,color='#17324D',zorder=4)
    a.set_xticks(xx,[f"B={r['batch']}\nβ₂={r['beta2']}" for r in rr]);a.set_ylabel('Δ validation loss\n(no clipping − clipped)');a.legend(frameon=False,loc='upper left',fontsize=13);finish(f,'ablation')
    f,a=fig();dd=[[r['loss_difference'] for r in data['matched_screening'] if r['batch']==b] for b in BATCHES]
    bp=a.boxplot(dd,positions=range(4),widths=.44,patch_artist=True,showfliers=False,medianprops={'color':'#17324D','lw':2});
    for patch in bp['boxes']:patch.set_facecolor('#F4D7C6')
    for i,vals in enumerate(dd):a.scatter(i+np.linspace(-.17,.17,len(vals)),sorted(vals),s=7,alpha=.45,color=COLORS['unclipped'])
    a.set_yscale('symlog',linthresh=.025);a.set_ylim(-.075,1.5);a.axhline(0,color='#17324D',lw=1);a.set_xticks(range(4),BATCHES);a.set_xlabel('Global batch size');a.set_ylabel('Δ validation loss\n(no clipping − clipped)');a.set_title('')
    for i,s in enumerate(data['matched_summary']):a.text(i,1.035,f"median {s['median_delta']:+.4f}\nclip wins {s['clipped_wins']}/120",transform=a.get_xaxis_transform(),ha='center',va='bottom',fontsize=13,clip_on=False)
    finish(f,'matched-grid')
    f,aa=fig(2);lossplot(aa[0],data,['sync','clipped','unclipped']);aa[0].legend(frameon=False,fontsize=10,loc='upper left')
    for method in ['clipped','unclipped']:
        rr=data['winners'][method];aa[1].plot(BATCHES,[r['lr']*1000 for r in rr],'o-',color=COLORS[method],label=LABELS[method])
    batchaxis(aa[1]);aa[1].set_ylabel('Selected LR × 1,000');aa[1].legend(frameon=False,fontsize=11);finish(f,'retuned')
    f,aa=fig(2)
    for method in ['sync','clipped','unclipped']:
        rr=sorted([r for r in data['performance'] if r['method']==method],key=lambda r:r['batch']);xx=[r['batch'] for r in rr];yy=[r['steady_targets_per_second']/1e6 for r in rr];aa[0].plot(xx,yy,'o-',color=COLORS[method],label=LABELS[method]);aa[0].fill_between(xx,[r['steady_targets_per_second_q25']/1e6 for r in rr],[r['steady_targets_per_second_q75']/1e6 for r in rr],color=COLORS[method],alpha=.12);aa[1].plot(xx,[r['session_seconds']/60 for r in rr],'o-',color=COLORS[method])
    for a in aa:batchaxis(a)
    aa[0].set_ylabel('Steady throughput\n(million targets/s)');aa[0].legend(frameon=False,fontsize=10);aa[1].set_ylabel('Session duration (min)');finish(f,'throughput')
    f,aa=fig(2)
    rr=sorted([r for r in data['replicated']['unclipped'] if r['batch']==16 and r['lr']==.002 and r['beta1']==.99],key=lambda r:r['seed42']);rank={r['recipe']:i+1 for i,r in enumerate(sorted(rr,key=lambda r:r['mean_loss']))}
    for i,r in enumerate(rr):
        aa[0].plot([0,1],[i+1,rank[r['recipe']]],'o-',color=COLORS['unclipped'],alpha=.8);aa[0].text(-.06,i+1,('HL 10M' if abs(r['beta2']-2**(-16*512/1e7))<1e-12 else f"β₂={r['beta2']}"),ha='right',va='center',fontsize=14)
    aa[0].set_xticks([0,1],['Seed 42','Three-seed mean']);aa[0].set_yticks(range(1,5));aa[0].invert_yaxis();aa[0].set_xlim(-.8,1.2);aa[0].set_ylabel('Rank among\nmatched four');aa[0].set_title('No clipping, B=16, LR=.002, β₁=.99',fontsize=13)
    rr=sorted([r for r in data['replicated']['clipped'] if r['batch']==32 and r['lr']==.004 and r['beta1']==.98],key=lambda r:r['beta2']);xx=np.arange(len(rr));aa[1].plot(xx,[r['seed42'] for r in rr],'o--',color='#64748B',label='Seed 42');aa[1].errorbar(xx,[r['mean_loss'] for r in rr],yerr=[r['std_loss'] for r in rr],fmt='o-',capsize=4,color=COLORS['clipped'],label='Three-seed mean ± SD');aa[1].set_xticks(xx,[str(r['beta2']) for r in rr]);aa[1].set_xlabel('β₂');aa[1].set_ylabel('Validation loss');aa[1].set_title('Clipped, B=32, LR=.004, β₁=.98',fontsize=12);aa[1].legend(frameon=False,fontsize=11);finish(f,'replication')
    prov=load(HERE/'provenance.json');prov['figures']=[{'file':str(p.relative_to(HERE)),'sha256':sha(p)} for p in sorted(ASSETS.glob('*')) if p.suffix in ['.svg','.pdf']];save(HERE/'provenance.json',prov)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--refresh',action='store_true');p.add_argument('--repo',type=Path,default=HERE.parent.parent);args=p.parse_args()
    if args.refresh:refresh(args.repo.resolve())
    data=analyze();draw(data);draw_heatmaps(data);print('Rebuilt nineteen SVG/PDF figures and portable analysis from bundled evidence.')
if __name__=='__main__':main()
