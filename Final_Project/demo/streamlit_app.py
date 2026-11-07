"""Interactive exploration of precomputed experiment results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import streamlit as st
from calculations import ranked, choose_examples

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
st.set_page_config(page_title='Beyond Ensemble Agreement',page_icon='◈',layout='wide')
st.markdown('''<style>
.block-container {max-width:1180px;padding-top:2rem;padding-bottom:2rem;}
h1 {letter-spacing:-.045em;font-weight:750!important;}
[data-testid="stMetric"] {background:#f0f4f8;border-radius:12px;padding:18px;}
[data-testid="stMetricLabel"] {color:#25364a;}
[data-testid="stCaptionContainer"] {font-size:.88rem;}
</style>''',unsafe_allow_html=True)

@st.cache_data
def manifest():return json.loads((ASSETS/'manifest.json').read_text())
@st.cache_data
def load_scores(ensemble,condition):
    with np.load(ASSETS/f'ensemble_{ensemble:02d}_{condition}.npz',allow_pickle=False) as d:return {k:d[k] for k in d.files}
@st.cache_data
def effects():return pd.read_csv(ASSETS/'replication_effects.csv')
@st.cache_data
def individual_effects():return pd.read_csv(ASSETS/'replication_individual.csv')
@st.cache_data
def thumbnail(condition,image_id):
    with Image.open(ASSETS/f'{condition}.webp') as atlas:
        index=manifest()['thumbnail_positions'][condition][str(int(image_id))]
        x=index%100*32;y=index//100*32
        return atlas.crop((x,y,x+32,y+32)).resize((192,192),Image.Resampling.NEAREST)

m=manifest()
st.title('Beyond Ensemble Agreement')
st.write('Can internal representations reveal confident errors that an agreeing neural-network ensemble misses?')
st.caption('Interactive exploration of precomputed experiment results.')
with st.sidebar:
    st.markdown('### Explore one ensemble')
    corruption=st.selectbox('Corruption',['Clean','Gaussian noise','Motion blur','JPEG compression'])
    severity=st.select_slider('Severity',options=[1,3,5],value=3,disabled=corruption=='Clean')
    regime_label=st.radio('Detector training',['Clean-trained','Noise-adapted'])
    coverage=st.slider('Coverage · predictions kept (%)',50,100,90,1)
    with st.expander('Individual ensemble',expanded=False):
        ensemble=st.selectbox('Ensemble',list(range(1,11)),format_func=lambda i:f'Ensemble {i:02d}',index=0)
        st.caption('Default: ensemble 01, fixed by ID—not chosen for its results. Each ensemble has three independently trained CNNs.')
    st.divider()
    st.caption('Presentation path: clean → Gaussian noise 3 → noise-adapted → motion blur 3. Keep coverage at 90%.')
condition='clean' if corruption=='Clean' else {'Gaussian noise':'gaussian_noise','Motion blur':'motion_blur','JPEG compression':'jpeg_compression'}[corruption]+f'_{severity}'
regime='clean_trained' if regime_label=='Clean-trained' else 'adapted'
explore,evidence=st.tabs(['Explore cached examples','Replication evidence'])
with explore:
    d=load_scores(ensemble,condition);cov=coverage/100
    a=ranked(d['error'],d[f'{regime}_A'],d['id'],cov);b=ranked(d['error'],d[f'{regime}_B'],d['id'],cov)
    st.markdown(f'**Ensemble {ensemble:02d}** · {corruption}'+('' if condition=='clean' else f' · severity {severity}')+f' · {regime_label.lower()} MLPs')
    st.caption('A: output uncertainty only. B: the same outputs + mean predicted-class representation novelty. Predictions are fixed; only the risk ranking changes.')
    fig=go.Figure()
    for key,label,color,result in [('A','A · Output only','#2563eb',a),('B','B · + Mean novelty','#ea580c',b)]:
        # Include exact selected point in the pre-exported curve.
        x=np.r_[d['curve_coverage'],cov];y=np.r_[d[f'{regime}_{key}_curve'],result['risk']];order=np.argsort(x,kind='stable')
        fig.add_trace(go.Scatter(x=x[order]*100,y=y[order]*100,name=label,line=dict(color=color,width=3),hovertemplate='%{x:.1f}% kept<br>%{y:.3f}% error<extra>'+label+'</extra>'))
        fig.add_trace(go.Scatter(x=[coverage],y=[result['risk']*100],mode='markers',marker=dict(size=11,color=color,line=dict(width=2,color='white')),showlegend=False,hovertemplate=f'{result["errors"]:,} accepted errors<extra>{label}</extra>'))
    fig.add_vline(x=coverage,line_dash='dot',line_color='#64748b')
    fig.update_layout(height=340,margin=dict(l=10,r=10,t=12,b=10),template='plotly_white',legend=dict(orientation='h',y=1.13),xaxis=dict(title='Coverage · predictions kept (%)',range=[0,100]),yaxis=dict(title='Error rate among accepted predictions (%)',rangemode='tozero'))
    st.plotly_chart(fig,width='stretch')
    c1,c2=st.columns(2)
    c1.metric('A · Accepted errors',f'{a["errors"]:,}',help=f'{a["kept"]:,} predictions kept; {a["risk"]:.2%} accepted error rate.')
    difference=a['errors']-b['errors']
    c2.metric('B · Accepted errors',f'{b["errors"]:,}',delta=f'{difference:+,} additional errors rejected vs A',help='Positive means B leaves fewer errors at exactly the same coverage.',delta_color='normal')
    st.caption(f'Both keep exactly {a["kept"]:,} / {len(d["id"]):,} predictions. AURC: A {a["aurc"]:.6f} · B {b["aurc"]:.6f} (lower is better). Coverage is an exploration control, not a newly fitted deployment threshold.')
    st.markdown('#### Where novelty helps—and where it hurts')
    columns=st.columns(4)
    for column,(category,j) in zip(columns,choose_examples(d['error'],d['id'],a['accepted'],b['accepted'])):
        with column:
            st.markdown(f'**{category}**')
            if j is None:
                st.info('No example of this type at the selected coverage.')
                continue
            image_id=int(d['id'][j]);st.image(thumbnail(condition,image_id),width='stretch')
            st.caption(f'Source image #{image_id}')
            st.write(f'True: **{m["labels"][int(d["label"][j])]}**  \nPredicted: **{m["labels"][int(d["prediction"][j])]}**')
            st.write(f'Ensemble confidence: **{d["confidence"][j]:.1%}**')
            st.write('A: '+('Accept' if a['accepted'][j] else 'Reject')+' · B: '+('Accept' if b['accepted'][j] else 'Reject'))
    st.caption('Examples are illustrative, selected by smallest source ID within each available category at the current coverage. They are not aggregate replication evidence. CIFAR images are 32 × 32 pixels; thumbnails are enlarged.')
with evidence:
    st.markdown('### What holds across ten independent ensembles?')
    st.caption('Aggregate evidence below always includes all ten additional ensembles. It is independent of the individual-ensemble selector. The original discovery ensemble is excluded.')
    e=effects()
    def get(study,comparison,family='mlp_selected',cond=condition,subset='all'):
        rows=e[e.study.eq(study)&e.comparison.eq(comparison)&e.family.eq(family)&e.condition.eq(cond)&e.subset.eq(subset)]
        if len(rows)!=1:raise ValueError('Missing or duplicate offline effect')
        return rows.iloc[0]
    st.markdown('**Primary clean hypothesis: novelty variation adds value beyond mean novelty (C−B).**')
    items=[];individual=individual_effects();dots=[]
    for label,study,comparison in [('Mean novelty · clean-trained','replication','mlp_selected:B_minus_A'),('Variation · clean-trained','replication','mlp_selected:C_minus_B'),('Mean novelty · adapted','adaptation','adapted_B_minus_A'),('Adaptation change · Gamma','adaptation','Gamma')]:
        row=get(study,comparison);items.append(dict(Comparison=label,mean=row['mean'],low=row.ci_low,high=row.ci_high,Improving=f'{int(row.negative)}/10',t_low=row.t_low,t_high=row.t_high))
        g=individual[individual.study.eq(study)&individual.comparison.eq(comparison)&individual.family.eq('mlp_selected')&individual.condition.eq(condition)&individual.subset.eq('all')]
        dots.append((label,g))
    table=pd.DataFrame(items)
    st.markdown(f'#### Selected condition: {corruption}'+('' if condition=='clean' else f' {severity}'))
    fig=go.Figure(go.Scatter(x=table['mean']*1000,y=table.Comparison,mode='markers',marker=dict(size=10,color=['#2563eb','#64748b','#ea580c','#0d9488']),error_x=dict(type='data',symmetric=False,array=(table.high-table['mean'])*1000,arrayminus=(table['mean']-table.low)*1000),customdata=table.Improving,hovertemplate='ΔAURC × 1,000: %{x:.4f}<br>Negative in %{customdata}<extra></extra>'))
    fig.data[0].showlegend=False
    for label,g in dots:
        fig.add_trace(go.Scatter(x=g.delta_aurc*1000,y=[label]*len(g),mode='markers',marker=dict(size=5,color='#64748b',opacity=.45),customdata=g.ensemble,hovertemplate='Ensemble %{customdata}<br>ΔAURC × 1,000: %{x:.4f}<extra></extra>',showlegend=False))
    fig.add_vline(x=0,line_dash='dot',line_color='gray');fig.update_layout(template='plotly_white',height=300,margin=dict(l=5,r=15,t=10,b=10),xaxis_title='ΔAURC × 1,000 · negative favors added feature',yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig,width='stretch')
    st.caption('Small gray dots: each of the ten ensemble effects. Colored markers and bars: mean and offline 95% ensemble-bootstrap interval.')
    display=table.rename(columns={'mean':'Mean ΔAURC','low':'95% CI low','high':'95% CI high','t_low':'t interval low','t_high':'t interval high'})
    st.dataframe(display,hide_index=True,width='stretch',column_config={c:st.column_config.NumberColumn(format='%.7f') for c in ['Mean ΔAURC','95% CI low','95% CI high','t interval low','t interval high']})
    st.markdown('**Reading the evidence:** Mean novelty gives a modest clean MLP gain and helps blur/JPEG consistently. Clean-trained novelty transfers poorly to Gaussian noise. Noise adaptation recovers an incremental MLP benefit at severities 3 and 5 in all ten ensembles, while eroding novelty’s advantage elsewhere. Recovery requires adapted B beating adapted A; negative Gamma alone is insufficient.')
    st.caption('Intervals are precomputed 20,000-draw ensemble-bootstrap 95% percentile intervals, conditional on the shared dataset/splits and one architecture. They do not include new-image uncertainty. Student-t intervals provide sensitivity checks. Follow-up/extension comparisons are exploratory; intervals are pointwise and unadjusted. Detector seeds and corruption conditions are not independent ensemble replications.')
    with st.expander('Full offline evidence · LR, MLP, geometry, depth and subsets'):
        st.dataframe(e,hide_index=True,width='stretch')
        st.download_button('Download offline effect table',e.to_csv(index=False),'replication_effects.csv','text/csv')
with st.expander('Methods and provenance'):
    st.write('CNN predictions use the uncalibrated ensemble average. Novelty uses the predicted class, CNN-training centroids and original clean detector-training normalization. Adapted A/B saw clean plus Gaussian-noise development data only. Validation selected each MLP and seed. Neither viewing examples nor moving the coverage slider changes trained models.')
    st.write('Ranking sorts by increasing cached risk, then ascending source image ID; exactly ceil(coverage × N) predictions are kept. AURC uses every prefix, not the displayed curve subsampling. All 80%/90% counts and AURC/AUROC checks were verified offline against saved evaluation tables.')
    st.caption(f'Export verification: {m["verification_checks"]:,} operating-point checks. Model weights, datasets and training environments are excluded from this app.')
