"""
Tech Challenge — Previsão de Obesidade
Sistema Preditivo + Dashboard Analítico
Autor: Cientista de Dados (FIAP Tech Challenge)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ──────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────
# LOAD ARTIFACTS
# ──────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "model.pkl"))

@st.cache_data
def load_json(name):
    with open(os.path.join(BASE, name)) as f:
        return json.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "Obesity.csv"))
    for col in ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
        df[col] = df[col].round().astype(int)
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['age_group'] = pd.cut(df['Age'], bins=[0,18,30,45,100],
                              labels=['teen','young_adult','adult','senior'])
    df['inactive'] = (df['FAF'] == 0).astype(int)
    df['risk_score'] = ((df['FAVC'] == 'yes').astype(int) +
                        df['inactive'] +
                        (df['family_history'] == 'yes').astype(int))
    return df

model     = load_model()
classes   = load_json("classes.json")
model_info= load_json("model_info.json")
feat_imp  = load_json("feature_importance.json")
eda       = load_json("eda_stats.json")
df        = load_data()

# ──────────────────────────────────────────────────
# LABEL MAPS
# ──────────────────────────────────────────────────
CLASS_PT = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight":       "Peso Normal",
    "Overweight_Level_I":  "Sobrepeso Grau I",
    "Overweight_Level_II": "Sobrepeso Grau II",
    "Obesity_Type_I":      "Obesidade Grau I",
    "Obesity_Type_II":     "Obesidade Grau II",
    "Obesity_Type_III":    "Obesidade Grau III",
}
CLASS_COLOR = {
    "Insufficient_Weight": "#3b82f6",
    "Normal_Weight":       "#22c55e",
    "Overweight_Level_I":  "#facc15",
    "Overweight_Level_II": "#f97316",
    "Obesity_Type_I":      "#ef4444",
    "Obesity_Type_II":     "#dc2626",
    "Obesity_Type_III":    "#991b1b",
}
CLASS_RISK = {
    "Insufficient_Weight": "⚠️ Baixo Peso",
    "Normal_Weight":       "✅ Saudável",
    "Overweight_Level_I":  "🟡 Atenção",
    "Overweight_Level_II": "🟠 Alerta",
    "Obesity_Type_I":      "🔴 Risco Moderado",
    "Obesity_Type_II":     "🔴 Risco Alto",
    "Obesity_Type_III":    "🚨 Risco Muito Alto",
}
RECOMENDACOES = {
    "Insufficient_Weight": [
        "Aumentar aporte calórico com alimentos nutritivos",
        "Consultar nutricionista para plano alimentar individualizado",
        "Avaliar causas subjacentes (exames laboratoriais)",
        "Monitorar peso semanalmente",
    ],
    "Normal_Weight": [
        "Manter hábitos alimentares atuais",
        "Continuar praticando atividade física regularmente",
        "Revisão médica anual preventiva",
        "Monitorar histórico familiar de doenças crônicas",
    ],
    "Overweight_Level_I": [
        "Reduzir consumo de alimentos ultra-processados e calóricos",
        "Iniciar ou intensificar atividade física (≥150 min/semana)",
        "Aumentar ingestão de vegetais e fibras",
        "Monitorar pressão arterial e glicemia",
    ],
    "Overweight_Level_II": [
        "Acompanhamento multidisciplinar (médico + nutricionista)",
        "Estabelecer metas realistas de perda de peso (0,5–1 kg/semana)",
        "Reduzir consumo de álcool e refrigerantes",
        "Solicitar exames: perfil lipídico, glicemia e função hepática",
    ],
    "Obesity_Type_I": [
        "Encaminhar para acompanhamento médico especializado",
        "Programa estruturado de atividade física com educador físico",
        "Avaliação de comorbidades (DM2, hipertensão, apneia)",
        "Terapia cognitivo-comportamental para mudança de hábitos",
    ],
    "Obesity_Type_II": [
        "Avaliação clínica urgente para comorbidades graves",
        "Considerar tratamento farmacológico (decisão médica)",
        "Programa intensivo de mudança de estilo de vida",
        "Monitoramento cardiológico e metabólico frequente",
    ],
    "Obesity_Type_III": [
        "🚨 Encaminhamento imediato para especialista em obesidade",
        "Avaliar elegibilidade para cirurgia bariátrica",
        "Suporte psicológico e nutricional intensivo",
        "Monitoramento contínuo de risco cardiovascular",
    ],
}

# ──────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8fafc; }
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    text-align: center;
}
.metric-card .value { font-size: 2rem; font-weight: 700; }
.metric-card .label { font-size: .85rem; color: #64748b; margin-top: 4px; }
.result-box {
    border-radius: 14px;
    padding: 24px 28px;
    margin: 16px 0;
    border-left: 6px solid;
}
.prob-bar-wrap { background: #e2e8f0; border-radius: 8px; height: 18px; margin: 4px 0; }
.prob-bar      { border-radius: 8px; height: 18px; transition: width .4s; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.title("ObesityAI")
    st.caption("Sistema de Apoio à Decisão Médica")
    st.divider()
    page = st.radio(
        "Navegação",
        ["🔮 Sistema Preditivo", "📊 Dashboard Analítico", "📋 Sobre o Modelo"],
        label_visibility="collapsed"
    )
    st.divider()
    st.metric("Acurácia do Modelo", f"{model_info['test_accuracy']*100:.1f}%")
    st.metric("Validação Cruzada",  f"{model_info['cv_mean']*100:.1f}%")
    st.caption("Random Forest — 300 estimadores")

# ══════════════════════════════════════════════════
# PAGE 1 — SISTEMA PREDITIVO
# ══════════════════════════════════════════════════
if page == "🔮 Sistema Preditivo":
    st.title("🔮 Sistema Preditivo de Obesidade")
    st.markdown("Preencha os dados do paciente para obter o diagnóstico preditivo.")
    st.divider()

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Dados Pessoais")
            gender  = st.selectbox("Gênero", ["Male", "Female"])
            age     = st.slider("Idade (anos)", 14, 80, 25)
            height  = st.slider("Altura (m)", 1.40, 2.10, 1.70, step=0.01)
            weight  = st.slider("Peso (kg)", 30.0, 200.0, 70.0, step=0.5)
            family  = st.selectbox("Histórico familiar de excesso de peso?", ["yes", "no"],
                                   format_func=lambda x: "Sim" if x=="yes" else "Não")

        with col2:
            st.subheader("🍎 Hábitos Alimentares")
            favc = st.selectbox("Come alimentos calóricos com frequência?", ["yes","no"],
                                format_func=lambda x: "Sim" if x=="yes" else "Não")
            fcvc = st.select_slider("Frequência de vegetais nas refeições",
                                    options=[1,2,3],
                                    format_func={1:"Raramente",2:"Às vezes",3:"Sempre"}.get)
            ncp  = st.select_slider("Nº de refeições principais/dia",
                                    options=[1,2,3,4],
                                    format_func={1:"1 refeição",2:"2 refeições",
                                                 3:"3 refeições",4:"4+ refeições"}.get,
                                    value=3)
            caec = st.selectbox("Come entre as refeições?",
                                ["no","Sometimes","Frequently","Always"],
                                format_func={"no":"Não","Sometimes":"Às vezes",
                                             "Frequently":"Frequentemente","Always":"Sempre"}.get,
                                index=1)
            ch2o = st.select_slider("Consumo diário de água",
                                    options=[1,2,3],
                                    format_func={1:"< 1 L/dia",2:"1–2 L/dia",3:"> 2 L/dia"}.get,
                                    value=2)
            calc = st.selectbox("Frequência de consumo de álcool?",
                                ["no","Sometimes","Frequently","Always"],
                                format_func={"no":"Não","Sometimes":"Às vezes",
                                             "Frequently":"Frequentemente","Always":"Sempre"}.get,
                                index=1)

        with col3:
            st.subheader("🏃 Estilo de Vida")
            smoke = st.selectbox("Fuma?", ["no","yes"],
                                 format_func=lambda x: "Não" if x=="no" else "Sim")
            scc   = st.selectbox("Monitora calorias ingeridas?", ["no","yes"],
                                 format_func=lambda x: "Não" if x=="no" else "Sim")
            faf   = st.select_slider("Frequência de atividade física (dias/semana)",
                                     options=[0,1,2,3],
                                     format_func={0:"Nenhuma",1:"1–2×/sem",
                                                  2:"3–4×/sem",3:"5+/sem"}.get)
            tue   = st.select_slider("Tempo com dispositivos eletrônicos",
                                     options=[0,1,2],
                                     format_func={0:"0–2 h/dia",1:"3–5 h/dia",2:"> 5 h/dia"}.get)
            mtrans= st.selectbox("Meio de transporte habitual",
                                 ["Public_Transportation","Automobile","Walking",
                                  "Motorbike","Bike"],
                                 format_func={"Public_Transportation":"Transporte Público",
                                              "Automobile":"Carro","Walking":"A pé",
                                              "Motorbike":"Moto","Bike":"Bicicleta"}.get)

        submitted = st.form_submit_button("🔍 Analisar Paciente", use_container_width=True,
                                          type="primary")

    if submitted:
        bmi       = weight / (height ** 2)
        age_grp   = pd.cut([age], bins=[0,18,30,45,100],
                           labels=['teen','young_adult','adult','senior'])[0]
        inactive  = int(faf == 0)
        risk_score= int(favc=="yes") + inactive + int(family=="yes")

        input_df = pd.DataFrame([{
            "Gender": gender, "Age": float(age), "Height": height, "Weight": weight,
            "family_history": family, "FAVC": favc, "FCVC": fcvc, "NCP": ncp,
            "CAEC": caec, "SMOKE": smoke, "CH2O": ch2o, "SCC": scc,
            "FAF": faf, "TUE": tue, "CALC": calc, "MTRANS": mtrans,
            "BMI": bmi, "age_group": age_grp, "inactive": inactive,
            "risk_score": risk_score
        }])

        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        color      = CLASS_COLOR[prediction]
        label_pt   = CLASS_PT[prediction]
        risk_lbl   = CLASS_RISK[prediction]

        st.divider()
        st.subheader("📋 Resultado do Diagnóstico Preditivo")

        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f"""<div class="metric-card">
            <div class="value" style="color:{color}">{label_pt}</div>
            <div class="label">Diagnóstico Previsto</div></div>""", unsafe_allow_html=True)
        r2.markdown(f"""<div class="metric-card">
            <div class="value">{risk_lbl}</div>
            <div class="label">Nível de Risco</div></div>""", unsafe_allow_html=True)
        r3.markdown(f"""<div class="metric-card">
            <div class="value">{bmi:.1f}</div>
            <div class="label">IMC Calculado</div></div>""", unsafe_allow_html=True)
        conf = max(proba) * 100
        r4.markdown(f"""<div class="metric-card">
            <div class="value">{conf:.0f}%</div>
            <div class="label">Confiança do Modelo</div></div>""", unsafe_allow_html=True)

        # Result box
        st.markdown(f"""
        <div class="result-box" style="background:{color}18; border-color:{color}">
            <h3 style="color:{color}; margin:0">{risk_lbl} — {label_pt}</h3>
            <p style="margin:8px 0 0; color:#374151">
                O modelo prevê <strong>{label_pt}</strong> com {conf:.0f}% de confiança.
                IMC calculado: <strong>{bmi:.1f} kg/m²</strong>.
            </p>
        </div>""", unsafe_allow_html=True)

        # Probability distribution
        col_prob, col_rec = st.columns([1, 1])
        with col_prob:
            st.markdown("**Distribuição de Probabilidades por Classe**")
            sorted_idx = np.argsort(proba)[::-1]
            for i in sorted_idx:
                cls   = classes[i]
                p     = proba[i]
                clr   = CLASS_COLOR[cls]
                lbl   = CLASS_PT[cls]
                bw    = int(p * 100)
                st.markdown(f"""
                <div style="margin:6px 0">
                    <div style="display:flex;justify-content:space-between;font-size:.82rem">
                        <span>{'<b>' if cls==prediction else ''}{lbl}{'</b>' if cls==prediction else ''}</span>
                        <span style="color:{clr};font-weight:{'700' if cls==prediction else '400'}">{p*100:.1f}%</span>
                    </div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar" style="width:{bw}%;background:{clr}"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_rec:
            st.markdown("**🩺 Recomendações Clínicas**")
            for rec in RECOMENDACOES[prediction]:
                st.markdown(f"• {rec}")
            st.info("⚠️ Este sistema é um apoio à decisão. O diagnóstico final é responsabilidade do médico.")

        # Key factors
        st.markdown("**📌 Fatores de Risco Identificados**")
        factors = []
        if bmi >= 30:          factors.append(f"🔴 IMC elevado ({bmi:.1f})")
        elif bmi >= 25:        factors.append(f"🟡 Sobrepeso (IMC {bmi:.1f})")
        if family == "yes":    factors.append("🔴 Histórico familiar de excesso de peso")
        if favc == "yes":      factors.append("🟠 Consumo frequente de alimentos calóricos")
        if faf == 0:           factors.append("🔴 Sedentarismo — sem atividade física")
        if faf <= 1:           factors.append("🟡 Baixa frequência de atividade física")
        if smoke == "yes":     factors.append("🔴 Tabagismo")
        if calc in ["Frequently","Always"]: factors.append("🟠 Consumo frequente de álcool")
        if ch2o == 1:          factors.append("🟡 Baixo consumo de água (< 1 L/dia)")
        if tue == 2:           factors.append("🟡 Alto tempo em dispositivos eletrônicos (> 5h/dia)")
        if not factors:
            factors.append("✅ Nenhum fator de risco crítico identificado")
        cols_f = st.columns(2)
        for i, f in enumerate(factors):
            cols_f[i % 2].markdown(f)

# ══════════════════════════════════════════════════
# PAGE 2 — DASHBOARD ANALÍTICO
# ══════════════════════════════════════════════════
elif page == "📊 Dashboard Analítico":
    st.title("📊 Dashboard Analítico — Estudo de Obesidade")
    st.markdown("Visão executiva com os principais insights para a equipe médica.")
    st.divider()

    # ── KPIs ──────────────────────────────────────
    total = len(df)
    obeso = df['Obesity'].str.startswith('Obesity').sum()
    sobrepeso = df['Obesity'].str.startswith('Overweight').sum()
    normal = (df['Obesity'] == 'Normal_Weight').sum()
    baixo  = (df['Obesity'] == 'Insufficient_Weight').sum()

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown(f'<div class="metric-card"><div class="value">{total}</div><div class="label">Total de Pacientes</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><div class="value" style="color:#ef4444">{obeso}</div><div class="label">Obesidade ({obeso/total*100:.0f}%)</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><div class="value" style="color:#f97316">{sobrepeso}</div><div class="label">Sobrepeso ({sobrepeso/total*100:.0f}%)</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="metric-card"><div class="value" style="color:#22c55e">{normal}</div><div class="label">Peso Normal ({normal/total*100:.0f}%)</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="metric-card"><div class="value" style="color:#3b82f6">{baixo}</div><div class="label">Abaixo do Peso ({baixo/total*100:.0f}%)</div></div>', unsafe_allow_html=True)

    st.divider()

    # ── CHARTS ROW 1 ──────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribuição por Nível de Obesidade")
        order = ["Insufficient_Weight","Normal_Weight","Overweight_Level_I",
                 "Overweight_Level_II","Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]
        counts = df['Obesity'].value_counts().reindex(order)
        labels_pt = [CLASS_PT[o] for o in order]
        colors    = [CLASS_COLOR[o] for o in order]

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=labels_pt, y=counts.values,
            marker_color=colors,
            text=counts.values, textposition='outside'
        ))
        fig.update_layout(margin=dict(t=20,b=0), height=320,
                          xaxis_tickangle=-30, showlegend=False,
                          plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("IMC Médio por Categoria")
        bmi_class = df.groupby('Obesity')['BMI'].mean().reindex(order)
        fig2 = go.Figure(go.Bar(
            x=labels_pt, y=bmi_class.values.round(1),
            marker_color=colors,
            text=bmi_class.values.round(1), textposition='outside'
        ))
        fig2.update_layout(margin=dict(t=20,b=0), height=320,
                           xaxis_tickangle=-30, showlegend=False,
                           plot_bgcolor='white', paper_bgcolor='white')
        fig2.add_hline(y=25, line_dash="dot", line_color="#f97316",
                       annotation_text="Sobrepeso ≥25")
        fig2.add_hline(y=30, line_dash="dot", line_color="#ef4444",
                       annotation_text="Obesidade ≥30")
        st.plotly_chart(fig2, use_container_width=True)

    # ── CHARTS ROW 2 ──────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Histórico Familiar × Nível de Obesidade")
        cross = pd.crosstab(df['Obesity'], df['family_history']).reindex(order)
        cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Sem histórico', x=labels_pt,
                              y=cross_pct['no'].round(1), marker_color='#94a3b8'))
        fig3.add_trace(go.Bar(name='Com histórico', x=labels_pt,
                              y=cross_pct['yes'].round(1), marker_color='#ef4444'))
        fig3.update_layout(barmode='stack', height=320, margin=dict(t=20,b=0),
                           xaxis_tickangle=-30, plot_bgcolor='white', paper_bgcolor='white',
                           legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("Atividade Física Média por Categoria")
        faf_class = df.groupby('Obesity')['FAF'].mean().reindex(order)
        colors_faf = ['#22c55e' if v >= 1.5 else '#f97316' if v >= 0.8 else '#ef4444'
                      for v in faf_class.values]
        fig4 = go.Figure(go.Bar(
            x=labels_pt, y=faf_class.values.round(2),
            marker_color=colors_faf,
            text=faf_class.values.round(2), textposition='outside'
        ))
        fig4.update_layout(margin=dict(t=20,b=0), height=320,
                           xaxis_tickangle=-30, showlegend=False,
                           plot_bgcolor='white', paper_bgcolor='white',
                           yaxis_title="Frequência média (0–3)")
        st.plotly_chart(fig4, use_container_width=True)

    # ── CHARTS ROW 3 ──────────────────────────────
    c5, c6 = st.columns(2)

    with c5:
        st.subheader("Importância das Features — Modelo RF")
        fi_items = sorted(feat_imp.items(), key=lambda x: x[1])[-15:]
        fi_labels= [x[0] for x in fi_items]
        fi_vals  = [x[1]*100 for x in fi_items]
        fig5 = go.Figure(go.Bar(
            x=fi_vals, y=fi_labels, orientation='h',
            marker_color='#6366f1',
            text=[f"{v:.1f}%" for v in fi_vals], textposition='outside'
        ))
        fig5.update_layout(height=380, margin=dict(t=20,l=160,r=60),
                           plot_bgcolor='white', paper_bgcolor='white',
                           xaxis_title="Importância (%)")
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        st.subheader("Distribuição de IMC por Gênero e Categoria")
        import plotly.express as px
        df_plot = df.copy()
        df_plot['Categoria'] = df_plot['Obesity'].map(CLASS_PT)
        fig6 = px.box(df_plot, x='Categoria', y='BMI', color='Gender',
                      color_discrete_map={'Male':'#6366f1','Female':'#ec4899'},
                      category_orders={'Categoria': [CLASS_PT[o] for o in order]})
        fig6.update_layout(height=380, margin=dict(t=20,b=0),
                           xaxis_tickangle=-30, plot_bgcolor='white',
                           paper_bgcolor='white', legend_title='Gênero')
        st.plotly_chart(fig6, use_container_width=True)

    # ── INSIGHTS ──────────────────────────────────
    st.divider()
    st.subheader("💡 Principais Insights para a Equipe Médica")
    i1,i2,i3 = st.columns(3)
    with i1:
        fam_obese = (df[df['Obesity'].str.startswith('Obesity')]['family_history']=='yes').mean()*100
        st.markdown(f"""
        **🧬 Histórico Familiar**  
        Entre pacientes obesos, **{fam_obese:.0f}%** têm histórico familiar de excesso de peso.
        Pacientes com histórico positivo têm risco significativamente maior — rastreamento precoce é recomendado.
        """)
    with i2:
        faf_obese  = df[df['Obesity'].str.startswith('Obesity')]['FAF'].mean()
        faf_normal = df[df['Obesity']=='Normal_Weight']['FAF'].mean()
        st.markdown(f"""
        **🏃 Atividade Física**  
        Pacientes com peso normal praticam em média **{faf_normal:.1f}×/sem**, enquanto obesos praticam **{faf_obese:.1f}×/sem**.
        Sedentarismo é um dos principais fatores modificáveis identificados.
        """)
    with i3:
        bmi_ob3 = df[df['Obesity']=='Obesity_Type_III']['BMI'].mean()
        bmi_norm= df[df['Obesity']=='Normal_Weight']['BMI'].mean()
        st.markdown(f"""
        **⚖️ IMC Crítico**  
        Obesidade Grau III apresenta IMC médio de **{bmi_ob3:.1f}**, contra **{bmi_norm:.1f}** no peso normal.
        Peso e altura são as features mais preditivas — exames físicos básicos têm alto valor diagnóstico.
        """)

    i4,i5,i6 = st.columns(3)
    with i4:
        favc_ob = (df[df['Obesity'].str.startswith('Obesity')]['FAVC']=='yes').mean()*100
        st.markdown(f"""
        **🍔 Alimentação Calórica**  
        **{favc_ob:.0f}%** dos pacientes obesos consomem alimentos altamente calóricos com frequência.
        Intervenção nutricional deve ser prioridade no plano de cuidados.
        """)
    with i5:
        pct_risco = ((df['Obesity'].str.startswith('Obesity')) | (df['Obesity'].str.startswith('Overweight'))).mean()*100
        st.markdown(f"""
        **📈 Prevalência**  
        **{pct_risco:.0f}%** da amostra está em sobrepeso ou obesidade, reforçando a urgência de programas preventivos e de triagem precoce na atenção básica.
        """)
    with i6:
        gen_ob = df[df['Obesity'].str.startswith('Obesity')]['Gender'].value_counts(normalize=True)*100
        g_maior = gen_ob.idxmax()
        g_pct = gen_ob.max()
        gen_pt = "Masculino" if g_maior=="Male" else "Feminino"
        st.markdown(f"""
        **👥 Gênero**  
        Gênero **{gen_pt}** concentra **{g_pct:.0f}%** dos casos de obesidade na amostra.
        Estratégias de intervenção devem considerar abordagens específicas por gênero.
        """)

# ══════════════════════════════════════════════════
# PAGE 3 — SOBRE O MODELO
# ══════════════════════════════════════════════════
elif page == "📋 Sobre o Modelo":
    st.title("📋 Sobre o Modelo e a Pipeline")

    st.markdown("""
    ## 🏗️ Arquitetura da Pipeline de ML

    A pipeline foi construída com as seguintes etapas:

    ### 1. Feature Engineering
    - **BMI** calculado a partir de Peso / Altura²
    - **age_group**: faixa etária categorizada (teen / young_adult / adult / senior)
    - **inactive**: flag binária para sedentarismo total (FAF == 0)
    - **risk_score**: score de risco composto (alimentação calórica + sedentarismo + histórico familiar)
    - Arredondamento de variáveis ordinais com ruído decimal (FCVC, NCP, CH2O, FAF, TUE)

    ### 2. Pré-processamento
    - **Variáveis numéricas**: StandardScaler
    - **Variáveis categóricas**: OneHotEncoder (handle_unknown='ignore')
    - Implementado com **ColumnTransformer** do scikit-learn

    ### 3. Modelo
    - **Random Forest Classifier** com 300 estimadores
    - `class_weight='balanced'` para lidar com desbalanceamento entre classes
    - `random_state=42` para reprodutibilidade

    ### 4. Avaliação
    - **Holdout**: 80% treino / 20% teste, estratificado por classe
    - **Validação Cruzada**: Stratified K-Fold com 5 folds
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia (Teste)", f"{model_info['test_accuracy']*100:.2f}%")
    col2.metric("CV Média", f"{model_info['cv_mean']*100:.2f}%")
    col3.metric("CV Std", f"±{model_info['cv_std']*100:.2f}%")

    st.markdown(f"""
    ### 📊 Dataset
    - **Total de registros**: {model_info['n_train'] + model_info['n_test']}
    - **Treino**: {model_info['n_train']} registros
    - **Teste**: {model_info['n_test']} registros
    - **Features originais**: 16 | **Features engenheiradas**: +4 = 20 total
    - **Classes alvo**: 7 (Insufficient Weight → Obesity Type III)
    - **Dados faltantes**: Nenhum

    ### ⚠️ Limitações e Considerações Éticas
    - O modelo foi treinado em dados sintéticos/augmentados — validação em dados clínicos reais é necessária antes de uso clínico
    - O sistema é uma ferramenta de **apoio à decisão**, não substitui avaliação médica profissional
    - Vieses de gênero, idade e origem geográfica dos dados podem influenciar predições
    - Recomenda-se auditoria periódica do modelo com novos dados
    """)
