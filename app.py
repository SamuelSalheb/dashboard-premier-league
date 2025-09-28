import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind, pearsonr
from statsmodels.stats.proportion import proportions_ztest

# Configuração da página
st.set_page_config(
    page_title="Análise Estatística - Premier League",
    page_icon="⚽",
    layout="wide"
)

# Título principal
st.title("⚽ Análise Estatística da Premier League")

# Sidebar
st.sidebar.title("Navegação")
pagina = st.sidebar.radio("Selecione a análise:", [
    "📊 Visão Geral", 
    "📈 Análise Exploratória", 
    "🧪 Testes de Hipótese",
    "🔍 Análise por Time",
    "⚔️ Comparação entre Times"
])

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv("England CSV.csv")
    
    # Pré-processamento
    df['Referee'] = df['Referee'].fillna('Desconhecido')
    
    # Renomear colunas
    df = df.rename(columns={
        'Date': 'Data', 'Season': 'Temporada', 'HomeTeam': 'Time_Casa', 'AwayTeam': 'Time_Visitante',
        'FTH Goals': 'Gols_Time_Casa', 'FTA Goals': 'Gols_Time_Visitante', 'FT Result': 'Resultado_Final',
        'HTH Goals': 'Gols_Casa_1T', 'HTA Goals': 'Gols_Visitante_1T', 'HT Result': 'Resultado_1T',
        'H Shots': 'Chutes_Casa', 'A Shots': 'Chutes_Visitante', 'H SOT': 'Chutes_Gol_Casa',
        'A SOT': 'Chutes_Gol_Visitante', 'H Fouls': 'Faltas_Casa', 'A Fouls': 'Faltas_Visitante',
        'H Corners': 'Escanteios_Casa', 'A Corners': 'Escanteios_Visitante', 'H Yellow': 'Amarelos_Casa',
        'A Yellow': 'Amarelos_Visitante', 'H Red': 'Vermelhos_Casa', 'A Red': 'Vermelhos_Visitante',
        'Referee': 'Árbitro'
    })
    
    # Engenharia de features
    df['Dif_Gols'] = df['Gols_Time_Casa'] - df['Gols_Time_Visitante']
    df['Total_Gols'] = df['Gols_Time_Casa'] + df['Gols_Time_Visitante']
    df['Vitoria_Casa'] = (df['Gols_Time_Casa'] > df['Gols_Time_Visitante']).astype(int)
    df['Empate'] = (df['Gols_Time_Casa'] == df['Gols_Time_Visitante']).astype(int)
    df['Vitoria_Visitante'] = (df['Gols_Time_Casa'] < df['Gols_Time_Visitante']).astype(int)
    df['Intensidade_Ofensiva'] = df['Chutes_Gol_Casa'] + df['Chutes_Gol_Visitante']
    df['Total_Faltas'] = df['Faltas_Casa'] + df['Faltas_Visitante']
    df['Total_Vermelhos'] = df['Vermelhos_Casa'] + df['Vermelhos_Visitante']
    
    # Converter dados
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
    df['Resultado_Final'] = df['Resultado_Final'].map({'H': 'Vitória Casa', 'A': 'Vitória Visitante', 'D': 'Empate'})
    
    # Normalizar nomes de times
    team_mapping = {"Brighton": "Brighton & Hove Albion", "Ipswich": "Ipswich Town"}
    df["Time_Casa"] = df["Time_Casa"].replace(team_mapping)
    df["Time_Visitante"] = df["Time_Visitante"].replace(team_mapping)
    
    # FLAGS PARA CONTROLE DE QUALIDADE DOS DADOS
    colunas_estatisticas = ['Chutes_Casa', 'Chutes_Visitante', 'Chutes_Gol_Casa', 
                           'Chutes_Gol_Visitante', 'Faltas_Casa', 'Faltas_Visitante',
                           'Escanteios_Casa', 'Escanteios_Visitante', 'Amarelos_Casa', 
                           'Amarelos_Visitante', 'Vermelhos_Casa', 'Vermelhos_Visitante']
    
    # Identificar temporadas com dados estatísticos completos
    df['Estatisticas_Completas'] = ~df[colunas_estatisticas].isnull().any(axis=1)
    df['Periodo_Completo'] = df['Temporada'] >= '2000/01'
    
    return df

df = load_data()

# Configuração de análise na sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Configuração da Análise")

analise_tipo = st.sidebar.radio(
    "Escopo Temporal:",
    ["📈 Análise Histórica Completa", "🎯 Estatísticas Detalhadas (Pós-2000)"]
)

# Definir dataset baseado na escolha
if analise_tipo == "🎯 Estatísticas Detalhadas (Pós-2000)":
    df_analise = df[df['Periodo_Completo']].copy()
    st.sidebar.info("💡 Analisando período com dados estatísticos completos (2000-2024)")
else:
    df_analise = df.copy()
    st.sidebar.info("💡 Analisando todo o histórico (1993-2024)")

# Função para alertas de qualidade de dados
def alerta_qualidade_dados(df_utilizado, analise_nome):
    temporadas = df_utilizado['Temporada'].unique()
    temporadas_incompletas = ['1993/94', '1994/95', '1995/96', '1996/97', '1997/98', '1998/99', '1999/00']
    
    if any(temp in temporadas_incompletas for temp in temporadas):
        st.warning(f"""
        **⚠️ AVISO METODOLÓGICO - {analise_nome}**
        
        Esta análise inclui temporadas com dados estatísticos parciais (1993-2000). 
        **Dados disponíveis**: Resultados e gols  
        **Dados limitados**: Estatísticas detalhadas de jogo
        
        *Para análise mais robusta, considere usar 'Estatísticas Detalhadas (Pós-2000)' na sidebar.*
        """)

# Página 1: Visão Geral
if pagina == "📊 Visão Geral":
    st.header("📊 Visão Geral do Dataset")
    
    # Alertas de qualidade
    alerta_qualidade_dados(df_analise, "VISÃO GERAL")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Jogos", len(df_analise))
    
    with col2:
        st.metric("Temporadas", df_analise['Temporada'].nunique())
    
    with col3:
        st.metric("Times Únicos", len(set(df_analise['Time_Casa'].unique()) | set(df_analise['Time_Visitante'].unique())))
    
    with col4:
        periodo = f"{df_analise['Temporada'].min()} a {df_analise['Temporada'].max()}"
        st.metric("Período", periodo)
    
    # Informações do dataset
    st.subheader("📋 Nota Metodológica")
    
    if analise_tipo == "🎯 Estatísticas Detalhadas (Pós-2000)":
        st.success("""
        **✅ Período com Dados Completos (2000/01 - 2023/24)**
        - Todas as estatísticas disponíveis: chutes, faltas, cartões, escanteios
        - Análises mais robustas e confiáveis
        - Ideal para testes de hipótese e correlações
        """)
    else:
        st.info("""
        **📊 Período Histórico Completo (1993/94 - 2023/24)**
        - **Dados completos em todas as temporadas**: Resultados, gols, tabela
        - **Dados estatísticos detalhados**: Disponíveis a partir de 2000/01
        - **Ideal para**: Análise evolutiva, dominância histórica, tendências de longo prazo
        """)
    
    # Estatísticas básicas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição de Resultados")
        resultados = df_analise['Resultado_Final'].value_counts()
        fig_resultados = px.pie(values=resultados.values, names=resultados.index, 
                               title="Proporção de Resultados")
        st.plotly_chart(fig_resultados)
    
    with col2:
        st.subheader("Estatísticas de Gols")
        media_gols_casa = df_analise['Gols_Time_Casa'].mean()
        media_gols_visitante = df_analise['Gols_Time_Visitante'].mean()
        
        st.metric("Média Gols em Casa", f"{media_gols_casa:.2f}")
        st.metric("Média Gols Fora", f"{media_gols_visitante:.2f}")
        st.metric("Média Total de Gols/Jogo", f"{media_gols_casa + media_gols_visitante:.2f}")

# Página 2: Análise Exploratória
elif pagina == "📈 Análise Exploratória":
    st.header("📈 Análise Exploratória")
    
    # Alertas de qualidade
    alerta_qualidade_dados(df_analise, "ANÁLISE EXPLORATÓRIA")
    
    # Filtros
    col1, col2 = st.columns(2)
    
    with col1:
        temporada_min = st.selectbox("Temporada Mínima", sorted(df_analise['Temporada'].unique()))
    
    with col2:
        # Definir opções baseadas no período selecionado
        if analise_tipo == "🎯 Estatísticas Detalhadas (Pós-2000)":
            metricas_opcoes = ['Gols', 'Cartões Amarelos', 'Cartões Vermelhos', 'Faltas', 'Chutes a Gol', 'Escanteios']
        else:
            # Para análise histórica, manter apenas métricas com dados completos
            metricas_opcoes = ['Gols', 'Cartões Amarelos', 'Cartões Vermelhos', 'Faltas']
        
        metricas = st.selectbox("Métrica para Análise", metricas_opcoes)
    
    # Filtrar dados
    df_filtrado = df_analise[df_analise['Temporada'] >= temporada_min]
    
    # Gráficos
    tab1, tab2, tab3 = st.tabs(["📅 Por Temporada", "🏆 Top Times", "📊 Distribuições"])
    
    with tab1:
        st.subheader("Evolução por Temporada")
        
        if metricas == 'Gols':
            gols_por_temp = df_filtrado.groupby('Temporada')['Total_Gols'].mean().reset_index()
            fig = px.line(gols_por_temp, x='Temporada', y='Total_Gols', 
                         title='Evolução da Média de Gols por Partida',
                         markers=True)
            
            # Destacar períodos com dados incompletos
            if analise_tipo == "📈 Análise Histórica Completa":
                temporadas_incompletas = ['1993/94', '1994/95', '1995/96', '1996/97', '1997/98', '1998/99', '1999/00']
                for temp in temporadas_incompletas:
                    if temp in gols_por_temp['Temporada'].values:
                        fig.add_vline(x=temp, line_dash="dash", line_color="red", opacity=0.3)
            
            st.plotly_chart(fig)
            
        elif metricas == 'Cartões Amarelos':
            amarelos_por_temp = df_filtrado.groupby('Temporada')[['Amarelos_Casa', 'Amarelos_Visitante']].mean()
            amarelos_por_temp['Media_Total'] = amarelos_por_temp.mean(axis=1)
            fig = px.line(amarelos_por_temp.reset_index(), x='Temporada', y='Media_Total',
                         title='Evolução da Média de Cartões Amarelos por Partida',
                         markers=True)
            st.plotly_chart(fig)
            
        elif metricas == 'Cartões Vermelhos':
            vermelhos_por_temp = df_filtrado.groupby('Temporada')[['Vermelhos_Casa', 'Vermelhos_Visitante']].mean()
            vermelhos_por_temp['Media_Total'] = vermelhos_por_temp.mean(axis=1)
            fig = px.line(vermelhos_por_temp.reset_index(), x='Temporada', y='Media_Total',
                         title='Evolução da Média de Cartões Vermelhos por Partida',
                         markers=True)
            st.plotly_chart(fig)
                
        elif metricas == 'Faltas':
            if 'Faltas_Casa' in df_filtrado.columns and not df_filtrado['Faltas_Casa'].isnull().all():
                faltas_por_temp = df_filtrado.groupby('Temporada')[['Faltas_Casa', 'Faltas_Visitante']].mean()
                faltas_por_temp['Media_Total'] = faltas_por_temp.mean(axis=1)
                fig = px.line(faltas_por_temp.reset_index(), x='Temporada', y='Media_Total',
                             title='Evolução da Média de Faltas por Partida',
                             markers=True)
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Dados de faltas não disponíveis para o período selecionado")
                
        elif metricas == 'Chutes a Gol':
            if 'Chutes_Gol_Casa' in df_filtrado.columns and not df_filtrado['Chutes_Gol_Casa'].isnull().all():
                chutes_por_temp = df_filtrado.groupby('Temporada')[['Chutes_Gol_Casa', 'Chutes_Gol_Visitante']].mean()
                chutes_por_temp['Media_Total'] = chutes_por_temp.mean(axis=1)
                fig = px.line(chutes_por_temp.reset_index(), x='Temporada', y='Media_Total',
                             title='Evolução da Média de Chutes a Gol por Partida',
                             markers=True)
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Dados de chutes a gol não disponíveis para o período selecionado")
                
        elif metricas == 'Escanteios':
            if 'Escanteios_Casa' in df_filtrado.columns and not df_filtrado['Escanteios_Casa'].isnull().all():
                escanteios_por_temp = df_filtrado.groupby('Temporada')[['Escanteios_Casa', 'Escanteios_Visitante']].mean()
                escanteios_por_temp['Media_Total'] = escanteios_por_temp.mean(axis=1)
                fig = px.line(escanteios_por_temp.reset_index(), x='Temporada', y='Media_Total',
                             title='Evolução da Média de Escanteios por Partida',
                             markers=True)
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Dados de escanteios não disponíveis para o período selecionado")
    
    with tab2:
        st.subheader("Top Times")
        
        if metricas == 'Gols':
            gols_casa = df_filtrado.groupby('Time_Casa')['Gols_Time_Casa'].sum()
            gols_visitante = df_filtrado.groupby('Time_Visitante')['Gols_Time_Visitante'].sum()
            gols_totais = gols_casa.add(gols_visitante, fill_value=0).sort_values(ascending=False).head(15)
            
            fig = px.bar(x=gols_totais.values, y=gols_totais.index, orientation='h',
                        title='Top 15 Times com Mais Gols Marcados')
            st.plotly_chart(fig)
            
        elif metricas == 'Cartões Amarelos':
            amarelos_casa = df_filtrado.groupby('Time_Casa')['Amarelos_Casa'].sum()
            amarelos_visitante = df_filtrado.groupby('Time_Visitante')['Amarelos_Visitante'].sum()
            amarelos_totais = amarelos_casa.add(amarelos_visitante, fill_value=0).sort_values(ascending=False).head(15)
            
            fig = px.bar(x=amarelos_totais.values, y=amarelos_totais.index, orientation='h',
                        title='Top 15 Times com Mais Cartões Amarelos')
            st.plotly_chart(fig)
            
        elif metricas == 'Cartões Vermelhos':
            vermelhos_casa = df_filtrado.groupby('Time_Casa')['Vermelhos_Casa'].sum()
            vermelhos_visitante = df_filtrado.groupby('Time_Visitante')['Vermelhos_Visitante'].sum()
            vermelhos_totais = vermelhos_casa.add(vermelhos_visitante, fill_value=0).sort_values(ascending=False).head(15)
            
            fig = px.bar(x=vermelhos_totais.values, y=vermelhos_totais.index, orientation='h',
                        title='Top 15 Times com Mais Cartões Vermelhos')
            st.plotly_chart(fig)
            
        elif metricas == 'Faltas':
            if 'Faltas_Casa' in df_filtrado.columns and not df_filtrado['Faltas_Casa'].isnull().all():
                faltas_casa = df_filtrado.groupby('Time_Casa')['Faltas_Casa'].sum()
                faltas_visitante = df_filtrado.groupby('Time_Visitante')['Faltas_Visitante'].sum()
                faltas_totais = faltas_casa.add(faltas_visitante, fill_value=0).sort_values(ascending=False).head(15)
                
                fig = px.bar(x=faltas_totais.values, y=faltas_totais.index, orientation='h',
                            title='Top 15 Times com Mais Faltas Cometidas')
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Dados de faltas não disponíveis para o período selecionado")
                
        elif metricas == 'Chutes a Gol':
            if 'Chutes_Gol_Casa' in df_filtrado.columns and not df_filtrado['Chutes_Gol_Casa'].isnull().all():
                chutes_casa = df_filtrado.groupby('Time_Casa')['Chutes_Gol_Casa'].sum()
                chutes_visitante = df_filtrado.groupby('Time_Visitante')['Chutes_Gol_Visitante'].sum()
                chutes_totais = chutes_casa.add(chutes_visitante, fill_value=0).sort_values(ascending=False).head(15)
                
                fig = px.bar(x=chutes_totais.values, y=chutes_totais.index, orientation='h',
                            title='Top 15 Times com Mais Chutes a Gol')
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Dados de chutes a gol não disponíveis para o período selecionado")
                
        elif metricas == 'Escanteios':
            if 'Escanteios_Casa' in df_filtrado.columns and not df_filtrado['Escanteios_Casa'].isnull().all():
                escanteios_casa = df_filtrado.groupby('Time_Casa')['Escanteios_Casa'].sum()
                escanteios_visitante = df_filtrado.groupby('Time_Visitante')['Escanteios_Visitante'].sum()
                escanteios_totais = escanteios_casa.add(escanteios_visitante, fill_value=0).sort_values(ascending=False).head(15)
                
                fig = px.bar(x=escanteios_totais.values, y=escanteios_totais.index, orientation='h',
                            title='Top 15 Times com Mais Escanteios')
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Dados de escanteios não disponíveis para o período selecionado")
    
    with tab3:
        st.subheader("Distribuições")
        
        if metricas == 'Gols':
            fig = px.histogram(df_filtrado, x='Gols_Time_Casa', nbins=20,
                             title='Distribuição dos Gols do Time da Casa')
            st.plotly_chart(fig)
            
        elif metricas == 'Cartões Amarelos':
            fig = px.histogram(df_filtrado, x='Amarelos_Casa', nbins=10,
                             title='Distribuição de Cartões Amarelos do Time da Casa')
            st.plotly_chart(fig)
            
        elif metricas == 'Cartões Vermelhos':
            fig = px.histogram(df_filtrado, x='Total_Vermelhos', nbins=5,
                             title='Distribuição de Cartões Vermelhos por Partida')
            st.plotly_chart(fig)
            
        elif metricas == 'Faltas':
            if 'Total_Faltas' in df_filtrado.columns and not df_filtrado['Total_Faltas'].isnull().all():
                fig = px.histogram(df_filtrado, x='Total_Faltas', nbins=20,
                                 title='Distribuição do Total de Faltas por Partida')
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Dados de faltas não disponíveis para o período selecionado")
                
        elif metricas == 'Chutes a Gol':
            if 'Chutes_Gol_Casa' in df_filtrado.columns and not df_filtrado['Chutes_Gol_Casa'].isnull().all():
                fig = px.histogram(df_filtrado, x='Chutes_Gol_Casa', nbins=15,
                                 title='Distribuição de Chutes a Gol do Time da Casa')
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Dados de chutes a gol não disponíveis para o período selecionado")
                
        elif metricas == 'Escanteios':
            if 'Escanteios_Casa' in df_filtrado.columns and not df_filtrado['Escanteios_Casa'].isnull().all():
                fig = px.histogram(df_filtrado, x='Escanteios_Casa', nbins=15,
                                 title='Distribuição de Escanteios do Time da Casa')
                st.plotly_chart(fig)
            else:
                st.error("⚠️ Dados de escanteios não disponíveis para o período selecionado")

# Página 3: Testes de Hipótese (MELHORADA)
elif pagina == "🧪 Testes de Hipótese":
    st.header("🧪 Testes de Hipótese Estatísticos")
    
    # Alertas de qualidade
    alerta_qualidade_dados(df_analise, "TESTES DE HIPÓTESE")

    
    tab1, tab2, tab3 = st.tabs(["🏆 Campeões vs Outros", "🏠 Casa vs Fora", "⚽ Chutes vs Gols"])
    
    with tab1:
        st.subheader("🏆 Teste t: Campeões vs Outros Times")
        
        # Verificar qualidade dos dados para este teste
        if analise_tipo == "📈 Análise Histórica Completa":
            st.info("""
            **📊 Escopo do Teste:** Inclui todo o histórico (1993-2024)
            **✅ Dados Suficientes:** Resultados e gols disponíveis em todas as temporadas
            """)
        
        # Calcular campeões
        champions = df_analise.groupby(['Temporada', 'Time_Casa'])['Vitoria_Casa'].sum().reset_index()
        champions = champions.sort_values(['Temporada', 'Vitoria_Casa'], ascending=[True, False])
        champions = champions.groupby('Temporada').head(1)
        df_analise['Campeao'] = df_analise['Time_Casa'].isin(champions['Time_Casa'])
        
        gols_campeoes = df_analise[df_analise['Campeao']]['Gols_Time_Casa'].values
        gols_outros = df_analise[~df_analise['Campeao']]['Gols_Time_Casa'].values
        
        t_stat, p_value = ttest_ind(gols_campeoes, gols_outros, equal_var=False)
        
        st.subheader("📈 Resultados do Teste")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Estatística t", f"{t_stat:.4f}")
        with col2:
            st.metric("p-valor", f"{p_value:.4f}")
        with col3:
            st.metric("Média Campeões", f"{gols_campeoes.mean():.4f}")
        with col4:
            st.metric("Média Outros", f"{gols_outros.mean():.4f}")
        
        # Interpretação detalhada
        st.subheader("💡 Interpretação")
        
        if p_value < 0.05:
            st.success("""
            ##### ✅ CONCLUSÃO: REJEITAMOS A HIPÓTESE NULA
            
            **H₀:** Campeões e outros times têm a mesma média de gols  
            **H₁:** Campeões têm média de gols maior que outros times
            
            **Evidência:** O valor-p ({p_value:.6f}) é menor que α = 0.05, indicando que a diferença observada 
            é **estatisticamente significativa**. Há forte evidência de que times campeões realmente fazem mais gols.
            """.format(p_value=p_value))
        else:
            st.warning("""
            ##### ❌ CONCLUSÃO: NÃO REJEITAMOS A HIPÓTESE NULA
            
            **Evidência:** O valor-p ({p_value:.6f}) é maior que α = 0.05, indicando que a diferença observada 
            **pode ter ocorrido por acaso**. Não há evidência suficiente para afirmar que campeões fazem mais gols.
            """.format(p_value=p_value))
    
    with tab2:
        st.subheader("🏠 Teste z: Vantagem de Jogar em Casa")
        
        # Verificar qualidade dos dados para este teste
        if analise_tipo == "📈 Análise Histórica Completa":
            st.info("""
            **📊 Escopo do Teste:** Análise de todo o histórico (1993-2024)
            **✅ Dados Ideais:** Teste baseado em resultados, disponíveis em todas as temporadas
            """)
        
        vitorias_casa = df_analise['Vitoria_Casa'].sum()
        vitorias_visitante = df_analise['Vitoria_Visitante'].sum()
        total = vitorias_casa + vitorias_visitante
        
        stat_z, p_value_z = proportions_ztest([vitorias_casa, vitorias_visitante], [total, total], alternative='larger')
        
        st.subheader("📈 Resultados do Teste")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Estatística z", f"{stat_z:.4f}")
        with col2:
            st.metric("p-valor", f"{p_value_z:.4f}")
        with col3:
            st.metric("Vitórias Casa", f"{vitorias_casa/total:.1%}")
        with col4:
            st.metric("Vitórias Fora", f"{vitorias_visitante/total:.1%}")
        
        st.subheader("💡 Interpretação")
        
        if p_value_z < 0.05:
            st.success("""
            ##### ✅ VANTAGEM SIGNIFICATIVA EM CASA
            
            **Evidência forte** de que times têm melhor desempenho jogando em casa. 
            A diferença de {diff:.1f}% é **estatisticamente significativa**.
            """.format(diff=(vitorias_casa/total - vitorias_visitante/total)*100))
        else:
            st.warning("""
            ##### ❌ SEM EVIDÊNCIA DE VANTAGEM
            
            **Não há evidência suficiente** para afirmar que há vantagem em jogar em casa.
            A diferença observada pode ser devido ao acaso.
            """)
    
    with tab3:
        st.subheader("⚽ Correlação: Chutes a Gol vs Gols Marcados")
        
        # Verificar se os dados estão disponíveis
        if analise_tipo == "📈 Análise Histórica Completa":
            st.error("""
            **⚠️ DADOS INSUFICIENTES PARA ESTE TESTE**
            
            Para análise de correlação entre chutes a gol e gols, é necessário usar o período 
            com dados estatísticos completos (pós-2000).
            
            **Por favor, selecione 'Estatísticas Detalhadas (Pós-2000)' na sidebar.**
            """)
        else:
            st.success("""
            **✅ DADOS ADEQUADOS PARA ANÁLISE**
            Período com estatísticas completas de chutes e gols (2000-2024)
            """)
            
            r_stat, p_value_r = pearsonr(df_analise['Chutes_Gol_Casa'].values, df_analise['Gols_Time_Casa'].values)
            
            st.subheader("📈 Resultados do Teste")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Coeficiente r", f"{r_stat:.4f}")
            with col2:
                st.metric("p-valor", f"{p_value_r:.4f}")
            with col3:
                st.metric("R² (Variância Explicada)", f"{r_stat**2:.1%}")
            
            # Gráfico
            fig = px.scatter(df_analise, x='Chutes_Gol_Casa', y='Gols_Time_Casa',
                            trendline='ols', title='Relação entre Chutes a Gol e Gols Marcados',
                            labels={'Chutes_Gol_Casa': 'Chutes a Gol', 'Gols_Time_Casa': 'Gols Marcados'})
            st.plotly_chart(fig)
            
            st.subheader("💡 Interpretação")
            
            if p_value_r < 0.05:
                if abs(r_stat) > 0.7:
                    força = "FORTE"
                elif abs(r_stat) > 0.3:
                    força = "MODERADA"
                else:
                    força = "FRACA"
                
                st.success(f"""
                ##### ✅ CORRELAÇÃO {força} E SIGNIFICATIVA
                
                **Relação positiva**: Mais chutes a gol estão associados a mais gols marcados.
                **Apenas {r_stat**2:.1%} da variação** nos gols é explicada pelos chutes a gol.
                """)
            else:
                st.warning("""
                ##### ❌ CORRELAÇÃO NÃO SIGNIFICATIVA
                
                **Não há evidência** de relação linear entre chutes a gol e gols marcados.
                A correlação observada pode ser devido ao acaso.
                """)

# Página 4: Análise por Time
elif pagina == "🔍 Análise por Time":
    st.header("🔍 Análise por Time")
    
    # Alertas de qualidade
    alerta_qualidade_dados(df_analise, "ANÁLISE POR TIME")
    
    # Selecionar time
    todos_times = sorted(set(df_analise['Time_Casa'].unique()) | set(df_analise['Time_Visitante'].unique()))
    time_selecionado = st.selectbox("Selecione um time:", todos_times)
    
    if time_selecionado:
        # Filtrar jogos do time
        jogos_time = df_analise[(df_analise['Time_Casa'] == time_selecionado) | (df_analise['Time_Visitante'] == time_selecionado)]
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Jogos", len(jogos_time))
        
        with col2:
            vitorias = len(jogos_time[
                ((jogos_time['Time_Casa'] == time_selecionado) & (jogos_time['Resultado_Final'] == 'Vitória Casa')) |
                ((jogos_time['Time_Visitante'] == time_selecionado) & (jogos_time['Resultado_Final'] == 'Vitória Visitante'))
            ])
            st.metric("Vitórias", vitorias)
        
        with col3:
            empates = len(jogos_time[jogos_time['Resultado_Final'] == 'Empate'])
            st.metric("Empates", empates)
        
        with col4:
            aproveitamento = (vitorias * 3 + empates) / (len(jogos_time) * 3) * 100
            st.metric("Aproveitamento", f"{aproveitamento:.1f}%")
        
        # Abas de análise
        tab1, tab2, tab3 = st.tabs(["📈 Performance Temporal", "⚽ Estatísticas", "🎯 Confrontos Diretos"])
        
        with tab1:
            st.subheader("📈 Evolução da Performance")
            
            # Performance por temporada
            performance_data = []
            for temp in sorted(jogos_time['Temporada'].unique()):
                jogos_temp = jogos_time[jogos_time['Temporada'] == temp]
                vitorias_temp = len(jogos_temp[
                    ((jogos_temp['Time_Casa'] == time_selecionado) & (jogos_temp['Resultado_Final'] == 'Vitória Casa')) |
                    ((jogos_temp['Time_Visitante'] == time_selecionado) & (jogos_temp['Resultado_Final'] == 'Vitória Visitante'))
                ])
                performance_data.append({
                    'Temporada': temp, 
                    'Vitórias': vitorias_temp, 
                    'Jogos': len(jogos_temp),
                    'Aproveitamento': (vitorias_temp * 3 + len(jogos_temp[jogos_temp['Resultado_Final'] == 'Empate'])) / (len(jogos_temp) * 3) * 100
                })
            
            df_performance = pd.DataFrame(performance_data)
            
            fig = px.line(df_performance, x='Temporada', y='Aproveitamento',
                         title=f'Aproveitamento do {time_selecionado} por Temporada',
                         markers=True)
            
            # Destacar períodos com dados incompletos
            if analise_tipo == "📈 Análise Histórica Completa":
                temporadas_incompletas = ['1993/94', '1994/95', '1995/96', '1996/97', '1997/98', '1998/99', '1999/00']
                for temp in temporadas_incompletas:
                    if temp in df_performance['Temporada'].values:
                        fig.add_vline(x=temp, line_dash="dash", line_color="red", opacity=0.3)
            
            st.plotly_chart(fig)
        
        with tab2:
            st.subheader("⚽ Estatísticas Detalhadas")
            
            # Estatísticas básicas (sempre disponíveis)
            gols_marcados = jogos_time[
                jogos_time['Time_Casa'] == time_selecionado
            ]['Gols_Time_Casa'].sum() + jogos_time[
                jogos_time['Time_Visitante'] == time_selecionado
            ]['Gols_Time_Visitante'].sum()
            
            gols_sofridos = jogos_time[
                jogos_time['Time_Casa'] == time_selecionado
            ]['Gols_Time_Visitante'].sum() + jogos_time[
                jogos_time['Time_Visitante'] == time_selecionado
            ]['Gols_Time_Casa'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Gols Marcados", gols_marcados)
            with col2:
                st.metric("Gols Sofridos", gols_sofridos)
            with col3:
                st.metric("Saldo de Gols", gols_marcados - gols_sofridos)
            
            # Estatísticas detalhadas (apenas se disponíveis)
            if analise_tipo == "🎯 Estatísticas Detalhadas (Pós-2000)":
                st.subheader("📊 Estatísticas de Jogo")
                
                # Calcular estatísticas avançadas
                chutes_gol_feitos = jogos_time[
                    jogos_time['Time_Casa'] == time_selecionado
                ]['Chutes_Gol_Casa'].sum() + jogos_time[
                    jogos_time['Time_Visitante'] == time_selecionado
                ]['Chutes_Gol_Visitante'].sum()
                
                eficiencia = (gols_marcados / chutes_gol_feitos * 100) if chutes_gol_feitos > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chutes a Gol Feitos", chutes_gol_feitos)
                with col2:
                    st.metric("Eficiência", f"{eficiencia:.1f}%")
                with col3:
                    st.metric("Cartões Amarelos", 
                             jogos_time[jogos_time['Time_Casa'] == time_selecionado]['Amarelos_Casa'].sum() +
                             jogos_time[jogos_time['Time_Visitante'] == time_selecionado]['Amarelos_Visitante'].sum())
            
            # Distribuição de resultados (sempre disponível)
            resultados_time = jogos_time['Resultado_Final'].value_counts()
            fig_pizza = px.pie(values=resultados_time.values, names=resultados_time.index,
                              title=f'Distribuição de Resultados - {time_selecionado}')
            st.plotly_chart(fig_pizza)
        
        with tab3:
            st.subheader("🎯 Confrontos Diretos")
            
            # Times adversários
            adversarios = st.selectbox("Selecione um adversário:", 
                                     [t for t in todos_times if t != time_selecionado])
            
            if adversarios:
                confrontos = df_analise[
                    ((df_analise['Time_Casa'] == time_selecionado) & (df_analise['Time_Visitante'] == adversarios)) |
                    ((df_analise['Time_Casa'] == adversarios) & (df_analise['Time_Visitante'] == time_selecionado))
                ]
                
                if len(confrontos) > 0:
                    vitorias_time = len(confrontos[
                        ((confrontos['Time_Casa'] == time_selecionado) & (confrontos['Resultado_Final'] == 'Vitória Casa')) |
                        ((confrontos['Time_Visitante'] == time_selecionado) & (confrontos['Resultado_Final'] == 'Vitória Visitante'))
                    ])
                    
                    vitorias_adversario = len(confrontos) - vitorias_time - len(confrontos[confrontos['Resultado_Final'] == 'Empate'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Vitórias {time_selecionado}", vitorias_time)
                    with col2:
                        st.metric("Empates", len(confrontos[confrontos['Resultado_Final'] == 'Empate']))
                    with col3:
                        st.metric(f"Vitórias {adversarios}", vitorias_adversario)
                    
                    # Últimos confrontos
                    st.subheader("Últimos 5 Confrontos")
                    ultimos_confrontos = confrontos.sort_values('Data', ascending=False).head(5)
                    for _, jogo in ultimos_confrontos.iterrows():
                        emoji = "🏠" if jogo['Time_Casa'] == time_selecionado else "✈️"
                        st.write(f"{emoji} {jogo['Data'].strftime('%d/%m/%Y')}: {jogo['Time_Casa']} {jogo['Gols_Time_Casa']}-{jogo['Gols_Time_Visitante']} {jogo['Time_Visitante']}")
                else:
                    st.info("Nenhum confronto encontrado entre estes times.")

# Página 5: Comparação entre Times (SIMPLIFICADA)
elif pagina == "⚔️ Comparação entre Times":
    st.header("⚔️ Comparação entre Dois Times")
    
    # Alertas de qualidade
    alerta_qualidade_dados(df_analise, "COMPARAÇÃO ENTRE TIMES")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time1 = st.selectbox("Selecione o primeiro time:", 
                           sorted(set(df_analise['Time_Casa'].unique()) | set(df_analise['Time_Visitante'].unique())))
    
    with col2:
        # Filtrar para não selecionar o mesmo time
        outros_times = [t for t in set(df_analise['Time_Casa'].unique()) | set(df_analise['Time_Visitante'].unique()) if t != time1]
        time2 = st.selectbox("Selecione o segundo time:", sorted(outros_times))
    
    if time1 and time2:
        st.markdown(f"## {time1} vs {time2}")
        
        # Dados dos dois times
        jogos_time1 = df_analise[(df_analise['Time_Casa'] == time1) | (df_analise['Time_Visitante'] == time1)]
        jogos_time2 = df_analise[(df_analise['Time_Casa'] == time2) | (df_analise['Time_Visitante'] == time2)]
        
        # Métricas comparativas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Jogos", 
                     f"{len(jogos_time1)} | {len(jogos_time2)}", 
                     f"{len(jogos_time1) - len(jogos_time2)}")
        
        with col2:
            vitorias_time1 = len(jogos_time1[
                ((jogos_time1['Time_Casa'] == time1) & (jogos_time1['Resultado_Final'] == 'Vitória Casa')) |
                ((jogos_time1['Time_Visitante'] == time1) & (jogos_time1['Resultado_Final'] == 'Vitória Visitante'))
            ])
            
            vitorias_time2 = len(jogos_time2[
                ((jogos_time2['Time_Casa'] == time2) & (jogos_time2['Resultado_Final'] == 'Vitória Casa')) |
                ((jogos_time2['Time_Visitante'] == time2) & (jogos_time2['Resultado_Final'] == 'Vitória Visitante'))
            ])
            
            st.metric("Vitórias", 
                     f"{vitorias_time1} | {vitorias_time2}", 
                     f"{vitorias_time1 - vitorias_time2}")
        
        with col3:
            empates_time1 = len(jogos_time1[jogos_time1['Resultado_Final'] == 'Empate'])
            empates_time2 = len(jogos_time2[jogos_time2['Resultado_Final'] == 'Empate'])
            
            st.metric("Empates", 
                     f"{empates_time1} | {empates_time2}", 
                     f"{empates_time1 - empates_time2}")
        
        with col4:
            ap_time1 = (vitorias_time1 * 3 + empates_time1) / (len(jogos_time1) * 3) * 100
            ap_time2 = (vitorias_time2 * 3 + empates_time2) / (len(jogos_time2) * 3) * 100
            
            st.metric("Aproveitamento", 
                     f"{ap_time1:.1f}% | {ap_time2:.1f}%", 
                     f"{ap_time1 - ap_time2:.1f}%")
        
        # Abas de comparação - REMOVIDA A ABA DE CONFRONTOS DIRETOS
        tab1, tab2 = st.tabs(["📊 Estatísticas Gerais", "📈 Evolução Temporal"])
        
        with tab1:
            # Gráfico de comparação
            comparacao_data = {
                'Metrica': ['Total Jogos', 'Vitórias', 'Empates', 'Aproveitamento'],
                time1: [len(jogos_time1), vitorias_time1, empates_time1, ap_time1],
                time2: [len(jogos_time2), vitorias_time2, empates_time2, ap_time2]
            }
            
            df_comparacao = pd.DataFrame(comparacao_data)
            df_melted = df_comparacao.melt(id_vars=['Metrica'], var_name='Time', value_name='Valor')
            
            fig = px.bar(df_melted, x='Metrica', y='Valor', color='Time', barmode='group',
                        title=f'Comparação Geral: {time1} vs {time2}')
            st.plotly_chart(fig)
            
            # Estatísticas adicionais se disponíveis
            if analise_tipo == "🎯 Estatísticas Detalhadas (Pós-2000)":
                st.subheader("📊 Estatísticas de Jogo Comparadas")
                
                # Calcular estatísticas avançadas para ambos times
                def calcular_estatisticas_avancadas(time, df):
                    jogos_time = df[(df['Time_Casa'] == time) | (df['Time_Visitante'] == time)]
                    
                    gols_marcados = jogos_time[
                        jogos_time['Time_Casa'] == time
                    ]['Gols_Time_Casa'].sum() + jogos_time[
                        jogos_time['Time_Visitante'] == time
                    ]['Gols_Time_Visitante'].sum()
                    
                    chutes_gol_feitos = jogos_time[
                        jogos_time['Time_Casa'] == time
                    ]['Chutes_Gol_Casa'].sum() + jogos_time[
                        jogos_time['Time_Visitante'] == time
                    ]['Chutes_Gol_Visitante'].sum()
                    
                    eficiencia = (gols_marcados / chutes_gol_feitos * 100) if chutes_gol_feitos > 0 else 0
                    
                    return {
                        'Eficiência (%)': eficiencia,
                        'Chutes a Gol': chutes_gol_feitos,
                        'Cartões Amarelos': jogos_time[jogos_time['Time_Casa'] == time]['Amarelos_Casa'].sum() +
                                          jogos_time[jogos_time['Time_Visitante'] == time]['Amarelos_Visitante'].sum()
                    }
                
                stats_time1 = calcular_estatisticas_avancadas(time1, df_analise)
                stats_time2 = calcular_estatisticas_avancadas(time2, df_analise)
                
                # Gráfico de comparação de estatísticas avançadas
                stats_comparacao = {
                    'Metrica': list(stats_time1.keys()),
                    time1: list(stats_time1.values()),
                    time2: list(stats_time2.values())
                }
                
                df_stats_comparacao = pd.DataFrame(stats_comparacao)
                df_stats_melted = df_stats_comparacao.melt(id_vars=['Metrica'], var_name='Time', value_name='Valor')
                
                fig_stats = px.bar(df_stats_melted, x='Metrica', y='Valor', color='Time', barmode='group',
                                 title=f'Estatísticas de Jogo: {time1} vs {time2}')
                st.plotly_chart(fig_stats)
        
        with tab2:
            st.subheader("Evolução Comparada por Temporada")
            
            # Aproveitamento por temporada
            evolucao_data = []
            
            for temp in sorted(df_analise['Temporada'].unique()):
                # Time 1
                jogos_temp1 = jogos_time1[jogos_time1['Temporada'] == temp]
                if len(jogos_temp1) > 0:
                    vitorias_temp1 = len(jogos_temp1[
                        ((jogos_temp1['Time_Casa'] == time1) & (jogos_temp1['Resultado_Final'] == 'Vitória Casa')) |
                        ((jogos_temp1['Time_Visitante'] == time1) & (jogos_temp1['Resultado_Final'] == 'Vitória Visitante'))
                    ])
                    empates_temp1 = len(jogos_temp1[jogos_temp1['Resultado_Final'] == 'Empate'])
                    ap_temp1 = (vitorias_temp1 * 3 + empates_temp1) / (len(jogos_temp1) * 3) * 100
                    
                    evolucao_data.append({'Temporada': temp, 'Time': time1, 'Aproveitamento': ap_temp1})
                
                # Time 2
                jogos_temp2 = jogos_time2[jogos_time2['Temporada'] == temp]
                if len(jogos_temp2) > 0:
                    vitorias_temp2 = len(jogos_temp2[
                        ((jogos_temp2['Time_Casa'] == time2) & (jogos_temp2['Resultado_Final'] == 'Vitória Casa')) |
                        ((jogos_temp2['Time_Visitante'] == time2) & (jogos_temp2['Resultado_Final'] == 'Vitória Visitante'))
                    ])
                    empates_temp2 = len(jogos_temp2[jogos_temp2['Resultado_Final'] == 'Empate'])
                    ap_temp2 = (vitorias_temp2 * 3 + empates_temp2) / (len(jogos_temp2) * 3) * 100
                    
                    evolucao_data.append({'Temporada': temp, 'Time': time2, 'Aproveitamento': ap_temp2})
            
            if evolucao_data:
                df_evolucao = pd.DataFrame(evolucao_data)
                fig = px.line(df_evolucao, x='Temporada', y='Aproveitamento', color='Time',
                             title=f'Evolução do Aproveitamento: {time1} vs {time2}',
                             markers=True)
                
                # Destacar períodos com dados incompletos
                if analise_tipo == "📈 Análise Histórica Completa":
                    temporadas_incompletas = ['1993/94', '1994/95', '1995/96', '1996/97', '1997/98', '1998/99', '1999/00']
                    for temp in temporadas_incompletas:
                        if temp in df_evolucao['Temporada'].values:
                            fig.add_vline(x=temp, line_dash="dash", line_color="red", opacity=0.3)
                
                st.plotly_chart(fig)