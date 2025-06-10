import pickle
import pandas as pd
import numpy as np

# ============ CARREGAR O ARQUIVO PKL ============
def carregar_resultados_pkl(caminho_arquivo):
    """Carrega os resultados do arquivo pickle"""
    try:
        with open(caminho_arquivo, 'rb') as f:
            resultados = pickle.load(f)
        print("✅ Arquivo PKL carregado com sucesso!")
        return resultados
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {caminho_arquivo}")
        return None
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")
        return None

# ============ EXPLORAR ESTRUTURA DO ARQUIVO ============
def explorar_estrutura_pkl(resultados):
    """Explora a estrutura dos dados no arquivo PKL"""
    print("\n📋 ESTRUTURA DOS DADOS NO ARQUIVO PKL:")
    print("=" * 50)
    
    if isinstance(resultados, dict):
        print("📁 Tipo: Dicionário Python")
        print(f"📊 Chaves principais: {list(resultados.keys())}")
        
        for chave in resultados.keys():
            print(f"\n📂 {chave}:")
            valor = resultados[chave]
            
            if isinstance(valor, dict):
                print(f"   📁 Dicionário com {len(valor)} itens")
                print(f"   🔑 Sub-chaves: {list(valor.keys())[:5]}...")  # Primeiras 5
            elif isinstance(valor, list):
                print(f"   📋 Lista com {len(valor)} elementos")
            elif isinstance(valor, (int, float)):
                print(f"   🔢 Número: {valor}")
            elif isinstance(valor, str):
                print(f"   📝 Texto: {valor[:50]}...")
            else:
                print(f"   ❓ Tipo: {type(valor)}")
    else:
        print(f"📁 Tipo: {type(resultados)}")

# ============ EXTRAIR MÉTRICAS ESPECÍFICAS ============
def extrair_metricas_do_pkl(resultados):
    """Extrai métricas específicas do arquivo PKL"""
    print("\n📊 MÉTRICAS EXTRAÍDAS DO PKL:")
    print("=" * 40)
    
    metricas_encontradas = {}
    
    # Estratégia Separada
    if 'separated_strategy_results' in resultados:
        sep_results = resultados['separated_strategy_results']
        print("\n🎯 ESTRATÉGIA SEPARADA:")
        
        for seed, dados in sep_results.items():
            print(f"\n  SEED {seed}:")
            
            # LOSO intra-dataset
            intra = dados.get('separated_strategy', {}).get('intra_dataset', {})
            for dataset, info in intra.items():
                if 'summary' in info:
                    summary = info['summary']
                    acc = summary.get('mean_accuracy', 0)
                    f1 = summary.get('mean_f1_macro', 0)
                    std_acc = summary.get('std_accuracy', 0)
                    n_folds = summary.get('n_folds', 0)
                    
                    print(f"    📈 LOSO {dataset}: Acc={acc:.3f}±{std_acc:.3f}, F1={f1:.3f}, Folds={n_folds}")
                    
                    metricas_encontradas[f"{seed}_{dataset}_loso"] = {
                        'accuracy': acc,
                        'f1_macro': f1,
                        'std_accuracy': std_acc,
                        'n_folds': n_folds
                    }
            
            # Cross-dataset
            cross = dados.get('separated_strategy', {}).get('cross_dataset', {})
            for cross_name, info in cross.items():
                acc = info.get('accuracy', 0)
                f1 = info.get('f1_macro', 0)
                print(f"    📈 Cross {cross_name}: Acc={acc:.3f}, F1={f1:.3f}")
                
                metricas_encontradas[f"{seed}_{cross_name}_cross"] = {
                    'accuracy': acc,
                    'f1_macro': f1
                }
    
    # Estratégia Combinada
    if 'combined_strategy_results' in resultados:
        comb_results = resultados['combined_strategy_results']
        print("\n🎯 ESTRATÉGIA COMBINADA:")
        
        for seed, dados in comb_results.items():
            print(f"\n  SEED {seed}:")
            
            # LOSO unificado
            unified = dados.get('combined_strategy', {}).get('unified_loso', {})
            for dataset, info in unified.items():
                if 'summary' in info:
                    summary = info['summary']
                    acc = summary.get('mean_accuracy', 0)
                    f1 = summary.get('mean_f1_macro', 0)
                    print(f"    📈 LOSO Unificado: Acc={acc:.3f}, F1={f1:.3f}")
                    
                    metricas_encontradas[f"{seed}_unified_loso"] = {
                        'accuracy': acc,
                        'f1_macro': f1
                    }
            
            # Cross-origin
            cross_origin = dados.get('combined_strategy', {}).get('cross_origin', {})
            for cross_name, info in cross_origin.items():
                acc = info.get('accuracy', 0)
                f1 = info.get('f1_macro', 0)
                print(f"    📈 Cross-origin {cross_name}: Acc={acc:.3f}, F1={f1:.3f}")
    
    # Informações de tempo
    if 'execution_stats' in resultados:
        exec_stats = resultados['execution_stats']
        feature_stats = exec_stats.get('feature_extraction_summary', {})
        
        print(f"\n⏱️ TEMPOS:")
        print(f"    Tempo total: {feature_stats.get('total_extraction_time', 0):.2f}s")
        print(f"    Imagens processadas: {feature_stats.get('total_images_processed', 0)}")
        print(f"    Tempo/imagem: {feature_stats.get('avg_time_per_image', 0):.4f}s")
        print(f"    Memória máxima: {feature_stats.get('max_memory_usage_mb', 0):.1f}MB")
        
        metricas_encontradas['tempo_info'] = feature_stats
    
    return metricas_encontradas

# ============ CONVERTER PARA FORMATO TABULAR ============
def criar_tabela_metricas(metricas_dict):
    """Converte métricas para DataFrame pandas"""
    rows = []
    
    for experimento, dados in metricas_dict.items():
        if experimento == 'tempo_info':
            continue
            
        row = {
            'Experimento': experimento,
            'Accuracy': dados.get('accuracy', 0),
            'F1_Macro': dados.get('f1_macro', 0),
            'Std_Accuracy': dados.get('std_accuracy', 0),
            'N_Folds': dados.get('n_folds', 0)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

# ============ EXEMPLO DE USO ============
if __name__ == "__main__":
    # Caminho para seu arquivo

    # caminho = "./results/strategy1_cross_dataset/strategy1_complete_results.pkl"
    caminho = "./results/passo3_lbp_svm/joao_dual_strategy_results_real.pkl"
    # caminho = "./results/passo3_lbp_svm/dual_strategy_results_real.pkl"

    # caminho = "./results/passo3_lbp_svm/01_dual_strategy_results_real.pkl"
    # caminho = "./multi_dataset_comprehensive_analysis/JAFFE_vs_Combined_Cross_Balanced/JAFFE_vs_Combined_Cross_Balanced_statistical_results.pkl"
    # caminho = "./multi_dataset_comprehensive_analysis/JAFFE_vs_CK+/JAFFE_vs_CK+_statistical_results.pkl"
    # caminho = "./multi_dataset_comprehensive_analysis/CK+_vs_Combined_Cross_Balanced/CK+_vs_Combined_Cross_Balanced_statistical_results.pkl"

    
    
    print("🔍 ANALISANDO ARQUIVO PKL - SVM + LBP")
    print("=" * 60)
    
    # 1. Carregar arquivo
    resultados = carregar_resultados_pkl(caminho)
    
    if resultados:
        # 2. Explorar estrutura
        explorar_estrutura_pkl(resultados)
        
        # 3. Extrair métricas
        metricas = extrair_metricas_do_pkl(resultados)
        
        # 4. Criar tabela
        if metricas:
            df_metricas = criar_tabela_metricas(metricas)
            
            print(f"\n📊 TABELA DE MÉTRICAS:")
            print(df_metricas.to_string(index=False))
            
            # 5. Salvar como CSV (mais legível)
            df_metricas.to_csv("metricas_svm_lbp.csv", index=False)
            print(f"\n💾 Métricas salvas em: metricas_svm_lbp.csv")
    
    print("\n✅ Análise concluída!")