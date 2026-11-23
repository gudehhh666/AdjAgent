"""Host-reviewed functional estimand; raw model proposals remain immutable."""
from copy import deepcopy
import hashlib
import json
import math

PROTOCOL_VERSION = 'matched-antigen-functional-output-v2'
HOST_NOTE = ('功能卡的比较分组、终点计数和背景扣除经过宿主统一校正；原始模型提案及逐字段修订记录另存。'
             '独立脾细胞制备必须追溯至独立动物；同一动物分孔或重复制备是技术单位，不能增加生物重复分母。'
             '该校正是实验设计审查，未产生湿实验结果。')
OUTPUT = ('在P0锁定同一目标CD4或CD8群与门控；以计数珠或体积流式测量终点活目标T细胞数N_end。'
          '在同一终点活目标群内测IFNγ+比例p；阳性细胞绝对数N_IFN=p×N_end。'
          '每孔记录其起始目标T细胞数N_start；Y=N_IFN/N_start。起始数仅用于归一，不能替代终点活细胞数。')
ENDPOINT = ('每孔先计算Y=p×N_end/N_start。处理组的抗原净响应F_C=Y(C+)−Y(C−)，'
            'vehicle组F_V=Y(V+)−Y(V−)；主要比较Δ=F_C−F_V。'
            'C+、C−、V+、V−各自使用本孔N_start；保留负值，不截断为零。'
            '按独立动物配对汇总；Δ的效应、区间和决策阈值须在揭盲前锁定。'
            'IFNγ读出是拟议的抗原功能先导终点，未证明保护效力。')
POSITIVE = ('C+=本分子+抗原，V+=匹配vehicle+同一抗原；在同一动物来源、同一目标T细胞群、'
            '相同培养/洗脱/再刺激条件下比较。如包含基础佐剂B，四组均保持相同B条件。')
NEGATIVE = ('C−=本分子+无抗原，V−=匹配vehicle+无抗原；除抗原外分别与C+、V+匹配。'
            '处理组扣除C−，vehicle组扣除V−；不能用V−代替C−。')
GROUPS = ['同一动物来源C+：本分子+抗原', '同一动物来源C−：本分子+无抗原',
          '同一动物来源V+：匹配vehicle+同一抗原', '同一动物来源V−：匹配vehicle+无抗原']


def _hash(value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()


def functional_contrast(wells):
    """Example calculation only; callers must supply measured, comparable wells."""
    if set(wells) != {'C+','C-','V+','V-'}: raise ValueError('Need four matched antigen wells')
    y={}
    for key,well in wells.items():
        p,n_end,n_start=(well[k] for k in ('positive_fraction','endpoint_live_count','starting_count'))
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in (p,n_end,n_start)):
            raise ValueError('Finite counts and fractions required')
        if not 0<=p<=1 or n_end<0 or n_start<=0: raise ValueError('Invalid functional measurement')
        y[key]=p*n_end/n_start
    f_c=y['C+']-y['C-'];f_v=y['V+']-y['V-']
    return {'Y':y,'F_C':f_c,'F_V':f_v,'delta':f_c-f_v}


def normalize_function_design(raw_plan):
    plan=deepcopy(raw_plan);changes=[]
    def replace(container,key,value,path):
        if container[key] != value:
            changes.append({'path':path,'before':deepcopy(container[key]),'after':deepcopy(value)})
            container[key]=value
    # Correct a prospective-design statement, never rewrite measured evidence.
    def prospective(value,path=''):
        if isinstance(value,dict):
            for k,v in value.items():
                if isinstance(v,str) and '经验证的C0倍数先导' in v:
                    replace(value,k,v.replace('经验证的C0倍数先导','拟议的C0倍数先导'),path+'/'+k)
                else: prospective(v,path+'/'+k)
        elif isinstance(value,list):
            for i,v in enumerate(value):
                if isinstance(v,str) and '经验证的C0倍数先导' in v:
                    replace(value,i,v.replace('经验证的C0倍数先导','拟议的C0倍数先导'),path+'/'+str(i))
                else: prospective(v,path+'/'+str(i))
    prospective(plan)
    # Specific ambiguities found in the reviewed batch. These changes are
    # explicitly logged; they add no measured value or pharmacology claim.
    if plan.get('compound')=='Thioridazine':
        key='mechanism_hypothesis'
        if 'Traj16缺失' in plan.get(key,''):
            replace(plan,key,plan[key].replace('Traj16缺失','Traj16转录信号未检出'),'/'+key)
        corrections={
            'Hmox1、Slc7a11、Cth、Nlrp3、Il1b、Mmp9、Gzmb、Ifng、Il2和Traj16的RNA与相应蛋白/细胞内染色；至少覆盖上升与下降方向。':
            'Hmox1、Slc7a11、Cth、Nlrp3、Il1b、Mmp9、Gzmb、Ifng、Il2的RNA与适用蛋白/细胞内染色；Traj16仅作RNA/TCR组成分析，不假定存在可用的独立蛋白抗体；至少覆盖上升与下降方向。',
            '分选后检测Hmox1、Slc7a11、Nlrp3、Il1b、Mmp9、Ifng和Traj16的RNA/蛋白。':
            '分选后检测Hmox1、Slc7a11、Nlrp3、Il1b、Mmp9、Ifng的RNA与适用蛋白；Traj16仅作RNA/TCR组成分析。'}
        for i,card in enumerate(plan['experiments']):
            for j,readout in enumerate(card.get('readouts',[])):
                if readout in corrections:
                    replace(card['readouts'],j,corrections[readout],f'/experiments/{i}/readouts/{j}')
    if plan.get('compound')=='Rapamycin':
        for i,card in enumerate(plan['experiments']):
            original='Rapamycin导致的T细胞功能下降是否依赖可逆mTOR通路抑制，还是由髓系炎症、组成变化或非特异性毒性驱动？'
            if card.get('question')==original:
                replace(card,'question','若E3确认抗原功能下降，再检验该变化是否与可逆mTOR通路抑制有关，或由髓系炎症、组成变化、非特异性毒性解释；依赖性仍需独立干预证据。',f'/experiments/{i}/question')
    for i,card in enumerate(plan['experiments']):
        if card['stage'] != 'function': continue
        path='/experiments/'+str(i)
        fd=card['function_design']
        replace(card,'groups',GROUPS.copy(),path+'/groups')
        # Remove old count/background expressions so competing formulas cannot
        # survive as secondary readouts. Preserve molecule-specific other assays.
        secondary=[s for s in card['readouts'] if not any(token in s.lower() for token in
            ('ifn','背景','归一','起始','终点','绝对','产出','目标t细胞'))]
        replace(card,'readouts',[OUTPUT,ENDPOINT,*secondary][:10],path+'/readouts')
        for key,value in [('antigen_positive_group',POSITIVE),('antigen_negative_group',NEGATIVE),
                          ('live_cell_output',OUTPUT),('primary_endpoint',ENDPOINT)]:
            replace(fd,key,value,path+'/function_design/'+key)
    limitations=list(plan['limitations'])
    if HOST_NOTE not in limitations:
        if len(limitations)>=15: limitations[-1]+=' '+HOST_NOTE
        else: limitations.append(HOST_NOTE)
    replace(plan,'limitations',limitations,'/limitations')
    return plan,{'protocol_version':PROTOCOL_VERSION,'kind':'host_design_review_not_model_output',
                 'raw_plan_sha256':_hash(raw_plan),'published_plan_sha256':_hash(plan),'changes':changes}
