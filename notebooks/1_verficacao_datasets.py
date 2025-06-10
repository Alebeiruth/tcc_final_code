# ============================================================================
# PASSO 1: VERIFICAÇÃO DOS DATASETS JAFFE E CK+
# ============================================================================

# Célula 1: Imports e Configurações Iniciais
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuração dos plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configurações globais
RANDOM_SEEDS = [42, 50]  # Para reprodutibilidade
IMAGE_SIZE = (96, 96)    # Tamanho padrão para todas as imagens
EXPECTED_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

print("✓ Imports carregados com sucesso!")
print(f"✓ Seeds configurados: {RANDOM_SEEDS}")
print(f"✓ Tamanho padrão de imagem: {IMAGE_SIZE}")
print(f"✓ Classes esperadas: {len(EXPECTED_CLASSES)} classes")

# Célula 2: Configuração dos Caminhos dos Datasets
# Definir os caminhos dos datasets
DATASET_PATHS = {
    'jaffe': '.\data\jaffe',
    'ck+': '.\data\ck',
    'jaffe_augmented': '.\data\combined_cross_balanced'  # Dataset aumentado
}

def verificar_estrutura_diretorios():
    """
    Verifica se os diretórios dos datasets existem e têm a estrutura correta.
    """
    resultados = {}
    
    for nome, caminho in DATASET_PATHS.items():
        print(f"\n📁 Verificando dataset: {nome.upper()}")
        print(f"   Caminho: {caminho}")
        
        if not os.path.exists(caminho):
            print(f"   ❌ ERRO: Diretório não encontrado!")
            resultados[nome] = {'existe': False, 'classes': [], 'estrutura_ok': False}
            continue
            
        # Listar subdiretórios (classes)
        try:
            classes_encontradas = [d for d in os.listdir(caminho) 
                                 if os.path.isdir(os.path.join(caminho, d))]
            classes_encontradas.sort()
            
            print(f"   ✓ Diretório existe")
            print(f"   ✓ Classes encontradas: {len(classes_encontradas)}")
            print(f"   📋 Classes: {classes_encontradas}")
            
            # Verificar se as classes esperadas estão presentes
            classes_faltando = set(EXPECTED_CLASSES) - set(classes_encontradas)
            classes_extras = set(classes_encontradas) - set(EXPECTED_CLASSES)
            
            if classes_faltando:
                print(f"   ⚠️  Classes faltando: {list(classes_faltando)}")
            if classes_extras:
                print(f"   ⚠️  Classes extras: {list(classes_extras)}")
            
            estrutura_ok = len(classes_faltando) == 0
            print(f"   {'✓' if estrutura_ok else '❌'} Estrutura {'OK' if estrutura_ok else 'INCORRETA'}")
            
            resultados[nome] = {
                'existe': True,
                'classes': classes_encontradas,
                'estrutura_ok': estrutura_ok,
                'classes_faltando': list(classes_faltando),
                'classes_extras': list(classes_extras)
            }
            
        except Exception as e:
            print(f"   ❌ ERRO ao acessar diretório: {e}")
            resultados[nome] = {'existe': True, 'classes': [], 'estrutura_ok': False, 'erro': str(e)}
    
    return resultados

# Executar verificação
estrutura_datasets = verificar_estrutura_diretorios()

# Célula 3: Análise Detalhada da Distribuição de Imagens
def analisar_distribuicao_imagens(dataset_path, dataset_name):
    """
    Analisa a distribuição de imagens por classe e coleta metadados.
    """
    print(f"\n🔍 ANÁLISE DETALHADA: {dataset_name.upper()}")
    print("=" * 50)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return None
    
    distribuicao = {}
    metadados = {
        'total_imagens': 0,
        'extensoes': Counter(),
        'tamanhos': [],
        'sujeitos': set(),
        'problemas': []
    }
    
    for classe in EXPECTED_CLASSES:
        caminho_classe = os.path.join(dataset_path, classe)
        
        if not os.path.exists(caminho_classe):
            print(f"   ⚠️  Classe '{classe}' não encontrada")
            distribuicao[classe] = 0
            continue
        
        # Contar imagens na classe
        arquivos = [f for f in os.listdir(caminho_classe) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        distribuicao[classe] = len(arquivos)
        metadados['total_imagens'] += len(arquivos)
        
        # Analisar metadados das imagens
        for arquivo in arquivos[:5]:  # Amostra dos primeiros 5 arquivos
            caminho_img = os.path.join(caminho_classe, arquivo)
            try:
                # Ler imagem para verificar integridade
                img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    metadados['problemas'].append(f"Não foi possível ler: {arquivo}")
                    continue
                
                # Coletar metadados
                metadados['extensoes'][arquivo.split('.')[-1].lower()] += 1
                metadados['tamanhos'].append(img.shape)
                
                # Extrair ID do sujeito (diferentes para JAFFE e CK+)
                if 'jaffe' in dataset_name.lower():
                    # JAFFE: primeiros 2 caracteres (ex: "KA")
                    sujeito = arquivo[:2]
                else:
                    # CK+: formato "S###_..."
                    sujeito = arquivo.split('_')[0] if '_' in arquivo else 'unknown'
                
                metadados['sujeitos'].add(sujeito)
                
            except Exception as e:
                metadados['problemas'].append(f"Erro em {arquivo}: {str(e)}")
    
    # Exibir resultados
    print(f"📊 DISTRIBUIÇÃO DE CLASSES:")
    for classe, count in distribuicao.items():
        print(f"   {classe:>10}: {count:>3} imagens")
    
    print(f"\n📈 ESTATÍSTICAS GERAIS:")
    print(f"   Total de imagens: {metadados['total_imagens']}")
    print(f"   Total de sujeitos: {len(metadados['sujeitos'])}")
    print(f"   Extensões: {dict(metadados['extensoes'])}")
    
    if metadados['tamanhos']:
        tamanhos_unicos = list(set(metadados['tamanhos']))
        print(f"   Tamanhos encontrados: {tamanhos_unicos}")
    
    if metadados['problemas']:
        print(f"\n⚠️  PROBLEMAS ENCONTRADOS:")
        for problema in metadados['problemas'][:10]:  # Mostrar até 10 problemas
            print(f"   - {problema}")
    
    # Calcular estatísticas de balanceamento
    counts = list(distribuicao.values())
    if counts:
        balanceamento = {
            'min': min(counts),
            'max': max(counts),
            'média': np.mean(counts),
            'std': np.std(counts),
            'coef_variacao': np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
        }
        
        print(f"\n⚖️  ANÁLISE DE BALANCEAMENTO:")
        print(f"   Mínimo: {balanceamento['min']} imagens")
        print(f"   Máximo: {balanceamento['max']} imagens")
        print(f"   Média: {balanceamento['média']:.1f} imagens")
        print(f"   Desvio padrão: {balanceamento['std']:.1f}")
        print(f"   Coef. variação: {balanceamento['coef_variacao']:.3f}")
        
        if balanceamento['coef_variacao'] > 0.3:
            print(f"   ⚠️  Dataset desbalanceado (CV > 0.3)")
        else:
            print(f"   ✓ Dataset relativamente balanceado")
    
    return {
        'distribuicao': distribuicao,
        'metadados': metadados,
        'balanceamento': balanceamento if 'balanceamento' in locals() else None
    }

# Analisar cada dataset
resultados_analise = {}
for nome, caminho in DATASET_PATHS.items():
    if estrutura_datasets.get(nome, {}).get('existe', False):
        resultados_analise[nome] = analisar_distribuicao_imagens(caminho, nome)
    else:
        print(f"\n⏭️  Pulando análise de {nome} (não encontrado)")

# Célula 4: Visualização da Distribuição dos Datasets
def plotar_distribuicao_datasets():
    """
    Cria visualizações comparativas da distribuição dos datasets.
    """
    # Preparar dados para visualização
    dados_plot = []
    
    for nome, resultado in resultados_analise.items():
        if resultado is None:
            continue
        
        for classe, count in resultado['distribuicao'].items():
            dados_plot.append({
                'Dataset': nome.upper(),
                'Classe': classe,
                'Quantidade': count
            })
    
    if not dados_plot:
        print("❌ Nenhum dado disponível para visualização")
        return
    
    df_plot = pd.DataFrame(dados_plot)
    
    # Criar subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análise Comparativa dos Datasets JAFFE e CK+', fontsize=16, fontweight='bold')
    
    # 1. Distribuição por classe (barplot agrupado)
    ax1 = axes[0, 0]
    sns.barplot(data=df_plot, x='Classe', y='Quantidade', hue='Dataset', ax=ax1)
    ax1.set_title('Distribuição de Classes por Dataset')
    ax1.set_xlabel('Classes de Emoções')
    ax1.set_ylabel('Número de Imagens')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Dataset')
    
    # 2. Totais por dataset (barplot simples)
    ax2 = axes[0, 1]
    totais = df_plot.groupby('Dataset')['Quantidade'].sum().reset_index()
    sns.barplot(data=totais, x='Dataset', y='Quantidade', ax=ax2)
    ax2.set_title('Total de Imagens por Dataset')
    ax2.set_ylabel('Total de Imagens')
    
    # Adicionar valores nas barras
    for i, v in enumerate(totais['Quantidade']):
        ax2.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 3. Coeficiente de variação (balanceamento)
    ax3 = axes[1, 0]
    coef_var_data = []
    for nome, resultado in resultados_analise.items():
        if resultado and resultado['balanceamento']:
            coef_var_data.append({
                'Dataset': nome.upper(),
                'Coef_Variacao': resultado['balanceamento']['coef_variacao']
            })
    
    if coef_var_data:
        df_cv = pd.DataFrame(coef_var_data)
        bars = sns.barplot(data=df_cv, x='Dataset', y='Coef_Variacao', ax=ax3)
        ax3.set_title('Coeficiente de Variação (Balanceamento)')
        ax3.set_ylabel('Coeficiente de Variação')
        ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Limite Desbalanceamento')
        ax3.legend()
        
        # Adicionar valores nas barras
        for i, v in enumerate(df_cv['Coef_Variacao']):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Coeficiente de Variação (Balanceamento)')
    
    # 4. Heatmap de distribuição
    ax4 = axes[1, 1]
    pivot_data = df_plot.pivot(index='Classe', columns='Dataset', values='Quantidade')
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Nº de Imagens'})
        ax4.set_title('Heatmap: Distribuição por Classe e Dataset')
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Classes de Emoções')
    else:
        ax4.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.show()
    
    return df_plot

# Gerar visualizações
df_distribuicao = plotar_distribuicao_datasets()

# Célula 5: Verificação de Integridade das Imagens
def verificar_integridade_imagens(dataset_path, dataset_name, amostra_size=50):
    """
    Verifica a integridade e características das imagens em uma amostra.
    """
    print(f"\n🔍 VERIFICAÇÃO DE INTEGRIDADE: {dataset_name.upper()}")
    print("=" * 50)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return None
    
    problemas = []
    imagens_ok = 0
    caracteristicas = {
        'tamanhos': Counter(),
        'min_pixel': [],
        'max_pixel': [],
        'mean_pixel': [],
        'std_pixel': []
    }
    
    # Coletar amostra de imagens de cada classe
    amostras_por_classe = amostra_size // len(EXPECTED_CLASSES)
    
    for classe in EXPECTED_CLASSES:
        caminho_classe = os.path.join(dataset_path, classe)
        if not os.path.exists(caminho_classe):
            continue
        
        arquivos = [f for f in os.listdir(caminho_classe) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # Pegar amostra aleatória
        np.random.seed(42)  # Para reprodutibilidade
        arquivos_amostra = np.random.choice(arquivos, 
                                          min(amostras_por_classe, len(arquivos)), 
                                          replace=False)
        
        for arquivo in arquivos_amostra:
            caminho_img = os.path.join(caminho_classe, arquivo)
            
            try:
                # Tentar ler a imagem
                img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    problemas.append(f"Não foi possível ler: {classe}/{arquivo}")
                    continue
                
                # Verificar se a imagem não está corrompida
                if img.size == 0:
                    problemas.append(f"Imagem vazia: {classe}/{arquivo}")
                    continue
                
                # Coletar características
                caracteristicas['tamanhos'][img.shape] += 1
                caracteristicas['min_pixel'].append(img.min())
                caracteristicas['max_pixel'].append(img.max())
                caracteristicas['mean_pixel'].append(img.mean())
                caracteristicas['std_pixel'].append(img.std())
                
                imagens_ok += 1
                
            except Exception as e:
                problemas.append(f"Erro em {classe}/{arquivo}: {str(e)}")
    
    # Relatório de integridade
    total_verificadas = imagens_ok + len(problemas)
    taxa_sucesso = (imagens_ok / total_verificadas * 100) if total_verificadas > 0 else 0
    
    print(f"📊 RESULTADO DA VERIFICAÇÃO:")
    print(f"   Imagens verificadas: {total_verificadas}")
    print(f"   Imagens OK: {imagens_ok}")
    print(f"   Problemas encontrados: {len(problemas)}")
    print(f"   Taxa de sucesso: {taxa_sucesso:.1f}%")
    
    if caracteristicas['tamanhos']:
        print(f"\n📐 CARACTERÍSTICAS DAS IMAGENS:")
        print(f"   Tamanhos encontrados:")
        for tamanho, count in caracteristicas['tamanhos'].most_common(5):
            print(f"     {tamanho}: {count} imagens")
        
        if caracteristicas['mean_pixel']:
            print(f"   Estatísticas de pixels:")
            print(f"     Valor mínimo: {np.mean(caracteristicas['min_pixel']):.1f} ± {np.std(caracteristicas['min_pixel']):.1f}")
            print(f"     Valor máximo: {np.mean(caracteristicas['max_pixel']):.1f} ± {np.std(caracteristicas['max_pixel']):.1f}")
            print(f"     Média: {np.mean(caracteristicas['mean_pixel']):.1f} ± {np.std(caracteristicas['mean_pixel']):.1f}")
            print(f"     Desvio padrão: {np.mean(caracteristicas['std_pixel']):.1f} ± {np.std(caracteristicas['std_pixel']):.1f}")
    
    if problemas:
        print(f"\n⚠️  PROBLEMAS ENCONTRADOS:")
        for problema in problemas[:10]:  # Mostrar até 10 problemas
            print(f"   - {problema}")
        if len(problemas) > 10:
            print(f"   ... e mais {len(problemas) - 10} problemas")
    
    return {
        'total_verificadas': total_verificadas,
        'imagens_ok': imagens_ok,
        'problemas': problemas,
        'taxa_sucesso': taxa_sucesso,
        'caracteristicas': caracteristicas
    }

# Verificar integridade de cada dataset
resultados_integridade = {}
for nome, caminho in DATASET_PATHS.items():
    if estrutura_datasets.get(nome, {}).get('existe', False):
        resultados_integridade[nome] = verificar_integridade_imagens(caminho, nome)


# Célula 6: Relatório Final de Verificação
def gerar_relatorio_final():
    """
    Gera um relatório consolidado da verificação dos datasets.
    """
    print("\n" + "="*70)
    print("🎯 RELATÓRIO FINAL - VERIFICAÇÃO DOS DATASETS")
    print("="*70)
    
    datasets_disponiveis = []
    datasets_problemas = []
    
    for nome, estrutura in estrutura_datasets.items():
        print(f"\n📋 DATASET: {nome.upper()}")
        print("-" * 40)
        
        if not estrutura.get('existe', False):
            print("❌ Status: NÃO ENCONTRADO")
            datasets_problemas.append(nome)
            continue
        
        if not estrutura.get('estrutura_ok', False):
            print("⚠️  Status: PROBLEMAS NA ESTRUTURA")
            if estrutura.get('classes_faltando'):
                print(f"   Classes faltando: {estrutura['classes_faltando']}")
            datasets_problemas.append(nome)
            continue
        
        # Dataset OK - coletar estatísticas
        analise = resultados_analise.get(nome)
        integridade = resultados_integridade.get(nome)
        
        if analise and integridade:
            total_imagens = analise['metadados']['total_imagens']
            total_sujeitos = len(analise['metadados']['sujeitos'])
            taxa_sucesso = integridade['taxa_sucesso']
            
            print("✅ Status: DISPONÍVEL")
            print(f"   📊 Total de imagens: {total_imagens}")
            print(f"   👥 Total de sujeitos: {total_sujeitos}")
            print(f"   🎯 Taxa de integridade: {taxa_sucesso:.1f}%")
            
            if analise['balanceamento']:
                cv = analise['balanceamento']['coef_variacao']
                balanceado = "SIM" if cv <= 0.3 else "NÃO"
                print(f"   ⚖️  Balanceado: {balanceado} (CV: {cv:.3f})")
            
            if taxa_sucesso >= 95 and (not analise['balanceamento'] or analise['balanceamento']['coef_variacao'] <= 0.5):
                datasets_disponiveis.append(nome)
                print("   🎉 Recomendado para uso")
            else:
                datasets_problemas.append(nome)
                if taxa_sucesso < 95:
                    print(f"   ⚠️  Baixa taxa de integridade ({taxa_sucesso:.1f}%)")
                if analise['balanceamento'] and analise['balanceamento']['coef_variacao'] > 0.5:
                    print(f"   ⚠️  Muito desbalanceado (CV: {analise['balanceamento']['coef_variacao']:.3f})")
        else:
            print("❌ Status: ERRO NA ANÁLISE")
            datasets_problemas.append(nome)
    
    # Resumo final
    print(f"\n" + "="*70)
    print("📈 RESUMO EXECUTIVO")
    print("="*70)
    print(f"✅ Datasets disponíveis para uso: {len(datasets_disponiveis)}")
    if datasets_disponiveis:
        for dataset in datasets_disponiveis:
            print(f"   - {dataset.upper()}")
    
    print(f"⚠️  Datasets com problemas: {len(datasets_problemas)}")
    if datasets_problemas:
        for dataset in datasets_problemas:
            print(f"   - {dataset.upper()}")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    if len(datasets_disponiveis) >= 2:
        print("   ✓ Datasets suficientes para análise cross-dataset")
        print("   ✓ Prosseguir para Passo 2: Data Augmentation")
    elif len(datasets_disponiveis) == 1:
        print("   ⚠️  Apenas um dataset disponível")
        print("   ⚠️  Considerar corrigir problemas nos outros datasets")
    else:
        print("   ❌ Nenhum dataset adequado encontrado")
        print("   ❌ Necessário corrigir problemas antes de prosseguir")
    
    return {
        'disponiveis': datasets_disponiveis,
        'problemas': datasets_problemas,
        'pronto_para_passo2': len(datasets_disponiveis) >= 2
    }

# Gerar relatório final
relatorio_final = gerar_relatorio_final()

# Salvar resultados para próximos passos
print(f"\n💾 Salvando resultados da verificação...")
verificacao_datasets = {
    'estrutura': estrutura_datasets,
    'analise': resultados_analise,
    'integridade': resultados_integridade,
    'relatorio': relatorio_final,
    'config': {
        'random_seeds': RANDOM_SEEDS,
        'image_size': IMAGE_SIZE,
        'expected_classes': EXPECTED_CLASSES,
        'dataset_paths': DATASET_PATHS
    }
}

print("✅ Passo 1 concluído com sucesso!")
print(f"📋 Próximo passo: {'Data Augmentation' if relatorio_final['pronto_para_passo2'] else 'Corrigir problemas nos datasets'}")