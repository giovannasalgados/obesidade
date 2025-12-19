
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model_pipeline.joblib')
col_order = pd.read_csv('col_order.csv', header=None)[0].astype(str).str.strip().tolist()
if '0' in col_order:
    col_order.remove('0')

st.title('Sistema Preditivo de Obesidade')

st.write(
    'Esta aplicação utiliza um modelo de Machine Learning para estimar '
    'o nível de obesidade com base em informações demográficas e comportamentais.')

# --- DICIONÁRIO DE VALORES PADRÃO ---
# Usamos valores neutros/médios para o que não for perguntado ao usuário
default_values = {
    'family_history': 'yes',
    'FAVC': 'yes',
    'FCVC': 2.0,       # Consumo de vegetais (frequência)
    'NCP': 3.0,        # Número de refeições principais
    'CAEC': 'Sometimes',
    'SMOKE': 'no',
    'CH2O': 2.0,       # Consumo de água
    'SCC': 'no',       # Monitoramento de calorias
    'TUE': 1.0,        # Tempo de uso de eletrônicos
    'CALC': 'Sometimes',
    'MTRANS': 'Public_Transportation'
}

col1, col2 = st.columns(2)

with col1:
    gender_pt = st.selectbox("Gênero", ["Masculino", "Feminino"])
    age = st.slider("Idade", 14, 61, 25)
    height = st.slider("Altura (m)", 1.45, 1.98, 1.70)
    
with col2:
    weight = st.slider("Peso (kg)", min_value=10.0, max_value=250.0, value=70.0, step=0.5)
    faf_labels = {0: "Sedentário", 1: "Baixa", 2: "Moderada", 3: "Alta"}
    faf = st.selectbox("Atividade Física", options=[0, 1, 2, 3], format_func=lambda x: faf_labels[x])

# --- PROCESSAMENTO ---
if st.button("🔍 Realizar Predição"):
    # Mapeamento para o modelo
    gender_map = {"Masculino": "Male", "Feminino": "Female"}
    
    # Cálculo do IMC (necessário pois está no seu col_order)
    imc_calculado = weight / (height ** 2)

    # Inputs da tela
    user_inputs = {
        "Gender": gender_map[gender_pt],
        "Age": age,
        "Height": height,
        "Weight": weight,
        "FAF": faf,
        "IMC": imc_calculado
    }
    
    # Mesclar: Padrão + Usuário
    final_features = default_values.copy()
    final_features.update(user_inputs)
    
    # Criar DataFrame
    input_df = pd.DataFrame([final_features])
    
    # REORDENAR e GARANTIR que todas as colunas do col_order existam
    # Se alguma coluna do CSV original faltar, preenchemos com 0
    for col in col_order:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Organiza o DataFrame na ordem exata do col_order.csv (sem o '0')
    input_df = input_df[col_order]

    try:
        # 5. Executa a predição (retorna 1, 2, 3 ou 4)
        prediction_raw = model.predict(input_df)[0]
        
        # Garante que tratamos como inteiro ou string conforme o modelo retorna
        # Se o modelo retorna números, o mapa fica assim:
        resultado_map = {
            1: "Peso Normal",
            2: "Obesidade Grau I",
            3: "Obesidade Grau II",
            4: "Obesidade Grau III"
        }

        # Busca a descrição amigável
        label_resultado = resultado_map.get(prediction_raw, f"Código {prediction_raw}")
        
        st.markdown("---")
        st.subheader("Resultado da Análise:")
        
        # Exibe o número e a descrição
        st.success(f"Nível de Obesidade Previsto: **{prediction_raw} - {label_resultado}**")
        st.info(f"IMC Calculado: **{imc_calculado:.2f}**")
        
    except Exception as e:
        st.error(f"Erro na predição: {e}")
