# CROSS-DATASET DRIFT ANALYSIS - VERSÃO CORRIGIDA
# Análise estatística rigorosa de dataset drift entre JAFFE e CK+
# Compatível com a estrutura de dados do notebook fornecido
# ============================================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp, entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import warnings
from tqdm import tqdm
from collections import Counter
warnings.filterwarnings('ignore')

# Configurações globais
DRIFT_ANALYSIS_CONFIG = {
    'statistical_tests': {
        'kolmogorov_smirnov': True,
        'wasserstein_distance': True,
        'kl_divergence': True,
    },
    'visualization': {
        'figure_size': (20, 16),
        'dpi': 300,
    },
    'performance': {
        'max_samples_ks': 2000,
        'max_samples_wasserstein': 1000,
        'max_features_kl': 500,
        'max_samples_per_class': 300
    }
}

EXPECTED_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

print("✅ Importações e configurações carregadas!")

# ============================================================================
# FUNÇÃO DE CARREGAMENTO DE DADOS
# ============================================================================

def load_images_from_directory(dataset_path, image_size=(96, 96)):
    """
    Carrega imagens de um diretório organizado por subpastas (uma para cada classe).
    
    Parameters:
    -----------
    dataset_path : str
        Caminho para o diretório raiz do dataset
    image_size : tuple
        Tamanho desejado das imagens (largura, altura)
    
    Returns:
    --------
    X : np.ndarray
        Array contendo as imagens processadas
    y : np.ndarray
        Array contendo os rótulos (nomes das classes)
    """
    print(f"📁 Carregando de: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Diretório não encontrado: {dataset_path}")

    X = []
    y = []

    # Obtém lista das classes a partir do nome das subpastas
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"📂 Classes encontradas: {classes}")

    # Itera sobre cada classe (subpasta)
    for label in classes:
        class_dir = os.path.join(dataset_path, label)
        files = [f for f in os.listdir(class_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"   📊 Classe '{label}': {len(files)} arquivos")

        # Itera sobre cada arquivo de imagem na subpasta
        for filename in tqdm(files, desc=f'Carregando {label}', leave=False):
            filepath = os.path.join(class_dir, filename)

            try:
                # Lê a imagem
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                # Redimensiona a imagem para o tamanho padrão
                image = cv2.resize(image, image_size)

                # Normaliza os pixels para o intervalo [0, 1]
                image = image / 255.0

                # Armazena a imagem e seu respectivo rótulo
                X.append(image)
                y.append(label)
                
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar {filename}: {e}")
                continue

    print(f"✅ Total carregado: {len(X)} imagens")
    return np.array(X), np.array(y)


# ================== EXEMPLO DE USO ADICIONAL ==================

def analyze_specific_dataset_pair():
    """
    Função para analisar um par específico de datasets rapidamente.
    Útil para testes ou análises focadas.
    """
    print("\n🔍 ANÁLISE RÁPIDA DE PAR ESPECÍFICO")
    print("=" * 50)
    
    # Exemplo: analisar apenas JAFFE vs Combined_Cross_Balanced
    try:
        # Carregar datasets específicos
        print("📁 Carregando JAFFE...")
        X_jaffe, y_jaffe = load_images_from_directory(r'.\data\jaffe', image_size=(96, 96))
        
        print("📁 Carregando Combined Cross Balanced...")
        X_combined, y_combined = load_images_from_directory(r'.\data\combine_cross_balanced', image_size=(96, 96))
        
        # Análise rápida
        results = run_comprehensive_drift_analysis(
            X_jaffe, X_combined,
            y_jaffe, y_combined,
            dataset1_name="JAFFE",
            dataset2_name="Combined_Cross_Balanced",
            save_results=True,
            output_dir="quick_analysis_jaffe_vs_combined"
        )
        
        plt.show()
        print("✅ Análise rápida concluída!")
        
    except Exception as e:
        print(f"❌ Erro na análise rápida: {e}")

def generate_summary_table():
    """
    Gera tabela resumo das métricas de drift para inclusão em papers.
    """
    
    # Exemplo de como criar tabela para LaTeX/papers
    summary_data = {
        'Dataset Pair': ['JAFFE vs CK+', 'JAFFE vs Combined', 'CK+ vs Combined'],
        'KS Drift (%)': [92.3, 85.7, 78.4],  # Valores exemplo
        'Wasserstein Distance': [0.234, 0.198, 0.156],
        'KL Divergence': [1.45, 1.23, 0.98],
        'Drift Level': ['CRÍTICO', 'ALTO', 'ALTO']
    }
    
    df = pd.DataFrame(summary_data)
    
    print("\n📊 TABELA RESUMO PARA PUBLICAÇÃO:")
    print("=" * 50)
    print(df.to_string(index=False))
    
    # Salvar em formato LaTeX
    latex_table = df.to_latex(index=False, caption="Cross-Dataset Drift Analysis Results", 
                             label="tab:drift_analysis")
    
    with open("drift_analysis_table.tex", "w") as f:
        f.write(latex_table)
    
    print("\n💾 Tabela LaTeX salva em: drift_analysis_table.tex")
    
    return df

print("✅ Multi-Dataset Drift Analysis Pipeline implementado com sucesso!")
print("🎯 Execute o script para analisar drift entre JAFFE, CK+ e Combined Cross Balanced!")
print()
print("📚 FUNÇÕES DISPONÍVEIS:")
print("   • run_multi_dataset_drift_analysis() - Análise completa multi-dataset")
print("   • analyze_specific_dataset_pair() - Análise rápida de um par específico") 
print("   • generate_summary_table() - Gera tabela resumo para publicação")
print()
print("🔗 OUTPUTS PRINCIPAIS:")
print("   • consolidated_drift_analysis.png - Visualização consolidada")
print("   • consolidated_report.txt - Relatório executivo completo")
print("   • drift_comparison_matrix.csv - Matriz de comparação em CSV")
print("   • Relatórios individuais para cada par de datasets")
print()
print("📈 MÉTRICAS IMPLEMENTADAS:")
print("   • Kolmogorov-Smirnov Test - Compara distribuições cumulativas")
print("   • Wasserstein Distance - Earth Mover's Distance entre distribuições")
print("   • Kullback-Leibler Divergence - Assimetria informacional")
print("   • Jensen-Shannon Divergence - Versão simétrica da KL")
print("   • Análise por classe de emoção")
print("   • Ranking de similaridade entre datasets")# ============================================================================


# ================== CONFIGURAÇÕES DE ANÁLISE ==================

DRIFT_ANALYSIS_CONFIG = {
    'statistical_tests': {
        'kolmogorov_smirnov': True,
        'wasserstein_distance': True,
        'kl_divergence': True,
        'jensen_shannon': True,
        'mann_whitney_u': True,
        'anderson_darling': True
    },
    'dimensionality_reduction': {
        'pca_components': [2, 3, 10, 50],
        'tsne_perplexity': [30, 50, 100],
        'lda_components': 2
    },
    'clustering_analysis': {
        'kmeans_clusters': [2, 5, 7, 10],
        'silhouette_analysis': True
    },
    'visualization': {
        'figure_size': (20, 16),
        'dpi': 300,
        'style': 'scientific',
        'color_palette': 'Set2'
    },
    'significance_levels': [0.001, 0.01, 0.05]
}

EXPECTED_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

print("✓ Configurações de análise de drift carregadas!")

# ================== FUNÇÃO CORRIGIDA DE CARREGAMENTO ==================

def load_images_from_directory(dataset_path, image_size=(96, 96)):
    """
    Carrega imagens de um diretório organizado por subpastas (uma para cada classe).
    Versão corrigida compatível com a estrutura do notebook.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Diretório não encontrado: {dataset_path}")

    X = []
    y = []

    # Obtém lista das classes a partir do nome das subpastas
    classes = sorted(os.listdir(dataset_path))
    print(f"📁 Diretório: {dataset_path}")
    print(f"📂 Classes encontradas: {classes}")

    # Itera sobre cada classe (subpasta)
    for label in classes:
        class_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(class_dir):
            continue

        files = os.listdir(class_dir)
        print(f"   📊 Classe '{label}': {len(files)} arquivos")

        # Itera sobre cada arquivo de imagem na subpasta
        for filename in tqdm(files, desc=f'Carregando {label}'):
            filepath = os.path.join(class_dir, filename)

            # Lê a imagem
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            # Redimensiona a imagem para o tamanho padrão
            image = cv2.resize(image, image_size)

            # Normaliza os pixels para o intervalo [0, 1]
            image = image / 255.0

            # Armazena a imagem e seu respectivo rótulo
            X.append(image)
            y.append(label)

    print(f"✅ Total carregado: {len(X)} imagens")
    return np.array(X), np.array(y)

# ================== ANALISADOR DE DRIFT ESTATÍSTICO ==================

class StatisticalDriftAnalyzer:
    """
    Analisador estatístico avançado para quantificar drift entre datasets.
    """
    
    def __init__(self, config=DRIFT_ANALYSIS_CONFIG):
        self.config = config
        self.results = {
            'feature_level_drift': {},
            'distribution_distances': {},
            'statistical_tests': {},
            'class_level_analysis': {},
            'summary_statistics': {}
        }
        
        # Configurar estilo visual
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(config['visualization']['color_palette'])
    
    def kolmogorov_smirnov_test(self, X1, X2, feature_names=None):
        """Executa teste Kolmogorov-Smirnov para cada feature."""
        print("📊 Executando Kolmogorov-Smirnov Test...")
        
        n_features = X1.shape[1]
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        ks_results = {
            'statistics': [],
            'p_values': [],
            'significant_features': [],
            'drift_percentages': []
        }
        
        # Amostragem para acelerar o cálculo se necessário
        max_samples = 5000
        if len(X1) > max_samples:
            idx1 = np.random.choice(len(X1), max_samples, replace=False)
            X1_sample = X1[idx1]
        else:
            X1_sample = X1
            
        if len(X2) > max_samples:
            idx2 = np.random.choice(len(X2), max_samples, replace=False)
            X2_sample = X2[idx2]
        else:
            X2_sample = X2
        
        for i in tqdm(range(n_features), desc="KS Test por feature"):
            # KS test para cada feature
            ks_stat, p_value = ks_2samp(X1_sample[:, i], X2_sample[:, i])
            
            ks_results['statistics'].append(ks_stat)
            ks_results['p_values'].append(p_value)
            
            # Determinar significância
            is_significant = p_value < 0.05
            if is_significant:
                ks_results['significant_features'].append(i)
            
            # Calcular percentual de drift
            drift_percentage = ks_stat * 100
            ks_results['drift_percentages'].append(drift_percentage)
        
        # Estatísticas resumo
        avg_ks_stat = np.mean(ks_results['statistics'])
        avg_drift_percentage = np.mean(ks_results['drift_percentages'])
        significant_ratio = len(ks_results['significant_features']) / n_features
        
        ks_summary = {
            'average_ks_statistic': avg_ks_stat,
            'average_drift_percentage': avg_drift_percentage,
            'significant_features_ratio': significant_ratio,
            'total_significant_features': len(ks_results['significant_features']),
            'max_drift_percentage': max(ks_results['drift_percentages']),
            'min_drift_percentage': min(ks_results['drift_percentages'])
        }
        
        self.results['statistical_tests']['kolmogorov_smirnov'] = {
            'detailed_results': ks_results,
            'summary': ks_summary
        }
        
        print(f"   ✅ KS Test concluído:")
        print(f"   📈 Drift médio: {avg_drift_percentage:.1f}%")
        print(f"   📈 Features significativas: {len(ks_results['significant_features'])}/{n_features}")
        print(f"   📈 Maior drift: {max(ks_results['drift_percentages']):.1f}%")
        
        return ks_results, ks_summary
    
    def wasserstein_distance_analysis(self, X1, X2):
        """Calcula Wasserstein Distance entre distribuições."""
        print("📊 Calculando Wasserstein Distance...")
        
        n_features = X1.shape[1]
        wasserstein_distances = []
        
        # Amostragem para otimizar performance
        max_samples = 2000
        if len(X1) > max_samples:
            idx1 = np.random.choice(len(X1), max_samples, replace=False)
            X1_sample = X1[idx1]
        else:
            X1_sample = X1
            
        if len(X2) > max_samples:
            idx2 = np.random.choice(len(X2), max_samples, replace=False)
            X2_sample = X2[idx2]
        else:
            X2_sample = X2
        
        for i in tqdm(range(n_features), desc="Wasserstein por feature"):
            w_distance = wasserstein_distance(X1_sample[:, i], X2_sample[:, i])
            wasserstein_distances.append(w_distance)
        
        # Estatísticas
        avg_wasserstein = np.mean(wasserstein_distances)
        max_wasserstein = np.max(wasserstein_distances)
        min_wasserstein = np.min(wasserstein_distances)
        std_wasserstein = np.std(wasserstein_distances)
        
        # Interpretação
        if avg_wasserstein > 0.3:
            interpretation = "DRIFT MUITO ALTO - Distribuições extremamente diferentes"
        elif avg_wasserstein > 0.15:
            interpretation = "DRIFT ALTO - Diferenças significativas"
        elif avg_wasserstein > 0.05:
            interpretation = "DRIFT MODERADO - Algumas diferenças detectadas"
        else:
            interpretation = "DRIFT BAIXO - Distribuições similares"
        
        wasserstein_results = {
            'distances_per_feature': wasserstein_distances,
            'average_distance': avg_wasserstein,
            'max_distance': max_wasserstein,
            'min_distance': min_wasserstein,
            'std_distance': std_wasserstein,
            'interpretation': interpretation
        }
        
        self.results['distribution_distances']['wasserstein'] = wasserstein_results
        
        print(f"   ✅ Wasserstein Distance calculada:")
        print(f"   📈 Distância média: {avg_wasserstein:.4f}")
        print(f"   📈 Distância máxima: {max_wasserstein:.4f}")
        print(f"   📊 Interpretação: {interpretation}")
        
        return wasserstein_results
    
    def kl_divergence_analysis(self, X1, X2):
        """Calcula Kullback-Leibler Divergence entre distribuições."""
        print("📊 Calculando KL Divergence...")
        
        n_features = X1.shape[1]
        kl_divergences = []
        js_divergences = []
        
        # Selecionar amostra de features para análise
        max_features = min(1000, n_features)
        feature_indices = np.random.choice(n_features, max_features, replace=False)
        
        for i in tqdm(feature_indices, desc="KL Divergence por feature"):
            try:
                # Criar histogramas normalizados
                min_val = min(np.min(X1[:, i]), np.min(X2[:, i]))
                max_val = max(np.max(X1[:, i]), np.max(X2[:, i]))
                
                if max_val == min_val:
                    continue
                    
                bins = np.linspace(min_val, max_val, 50)
                
                hist1, _ = np.histogram(X1[:, i], bins=bins, density=True)
                hist2, _ = np.histogram(X2[:, i], bins=bins, density=True)
                
                # Normalizar e adicionar epsilon
                epsilon = 1e-10
                hist1 = hist1 + epsilon
                hist2 = hist2 + epsilon
                hist1 = hist1 / np.sum(hist1)
                hist2 = hist2 / np.sum(hist2)
                
                # KL Divergence
                kl_div = entropy(hist1, hist2)
                if not np.isnan(kl_div) and not np.isinf(kl_div):
                    kl_divergences.append(kl_div)
                
                # Jensen-Shannon Divergence
                m = 0.5 * (hist1 + hist2)
                js_div = 0.5 * entropy(hist1, m) + 0.5 * entropy(hist2, m)
                if not np.isnan(js_div) and not np.isinf(js_div):
                    js_divergences.append(js_div)
                
            except Exception as e:
                continue
        
        # Estatísticas
        avg_kl = np.mean(kl_divergences) if kl_divergences else 0
        max_kl = np.max(kl_divergences) if kl_divergences else 0
        avg_js = np.mean(js_divergences) if js_divergences else 0
        max_js = np.max(js_divergences) if js_divergences else 0
        
        # Interpretação
        if avg_kl > 1.0:
            kl_interpretation = "DIVERGÊNCIA ALTA - Datasets muito diferentes"
        elif avg_kl > 0.5:
            kl_interpretation = "DIVERGÊNCIA MODERADA - Diferenças significativas"
        else:
            kl_interpretation = "DIVERGÊNCIA BAIXA - Datasets similares"
        
        kl_results = {
            'kl_divergences': kl_divergences,
            'js_divergences': js_divergences,
            'average_kl': avg_kl,
            'max_kl': max_kl,
            'average_js': avg_js,
            'max_js': max_js,
            'kl_interpretation': kl_interpretation,
            'valid_features': len(kl_divergences)
        }
        
        self.results['distribution_distances']['kl_divergence'] = kl_results
        
        print(f"   ✅ KL Divergence calculada:")
        print(f"   📈 KL Divergence média: {avg_kl:.4f}")
        print(f"   📈 JS Divergence média: {avg_js:.4f}")
        print(f"   📊 Interpretação: {kl_interpretation}")
        
        return kl_results
    
    def class_level_drift_analysis(self, X1, y1, X2, y2):
        """Analisa drift específico por classe de emoção."""
        print("📊 Analisando drift por classe...")
        
        class_drift_results = {}
        
        # Encontrar classes comuns
        classes1 = set(y1)
        classes2 = set(y2)
        common_classes = classes1.intersection(classes2)
        
        print(f"   📂 Classes em comum: {sorted(common_classes)}")
        
        for emotion_class in common_classes:
            # Filtrar dados por classe
            mask1 = y1 == emotion_class
            mask2 = y2 == emotion_class
            
            X1_class = X1[mask1]
            X2_class = X2[mask2]
            
            if len(X1_class) < 10 or len(X2_class) < 10:
                print(f"   ⚠️ Poucos exemplos para classe {emotion_class}")
                continue
            
            # Amostragem se necessário
            max_samples_class = 500
            if len(X1_class) > max_samples_class:
                idx = np.random.choice(len(X1_class), max_samples_class, replace=False)
                X1_class = X1_class[idx]
            if len(X2_class) > max_samples_class:
                idx = np.random.choice(len(X2_class), max_samples_class, replace=False)
                X2_class = X2_class[idx]
            
            # Selecionar subset de features para análise rápida
            n_features_sample = min(100, X1_class.shape[1])
            feature_indices = np.random.choice(X1_class.shape[1], n_features_sample, replace=False)
            
            # KS test para esta classe específica
            ks_stats = []
            for i in feature_indices:
                try:
                    ks_stat, _ = ks_2samp(X1_class[:, i], X2_class[:, i])
                    if not np.isnan(ks_stat):
                        ks_stats.append(ks_stat)
                except:
                    continue
            
            avg_ks_class = np.mean(ks_stats) if ks_stats else 0
            
            # Wasserstein distance para esta classe
            w_distances = []
            for i in feature_indices[:50]:  # Limitar para performance
                try:
                    w_dist = wasserstein_distance(X1_class[:, i], X2_class[:, i])
                    if not np.isnan(w_dist):
                        w_distances.append(w_dist)
                except:
                    continue
            
            avg_w_class = np.mean(w_distances) if w_distances else 0
            
            # Estatísticas da classe
            class_stats = {
                'sample_sizes': (len(X1_class), len(X2_class)),
                'avg_ks_statistic': avg_ks_class,
                'avg_wasserstein_distance': avg_w_class,
                'drift_percentage': avg_ks_class * 100,
                'feature_means_diff': np.mean(np.abs(np.mean(X1_class, axis=0) - np.mean(X2_class, axis=0)))
            }
            
            class_drift_results[emotion_class] = class_stats
            
            print(f"   📈 {emotion_class.upper()}: Drift={avg_ks_class*100:.1f}%, Wasserstein={avg_w_class:.4f}")
        
        self.results['class_level_analysis'] = class_drift_results
        return class_drift_results
    
    def comprehensive_statistical_analysis(self, X1, X2, y1=None, y2=None, 
                                         dataset1_name="Dataset1", dataset2_name="Dataset2"):
        """Executa análise estatística completa de drift entre dois datasets."""
        print(f"\n🔬 ANÁLISE ESTATÍSTICA COMPLETA: {dataset1_name} vs {dataset2_name}")
        print("=" * 80)
        
        # Flatten das imagens se necessário
        if len(X1.shape) > 2:
            X1_flat = X1.reshape(X1.shape[0], -1)
            print(f"   📊 {dataset1_name} - Imagens flatten: {X1.shape} -> {X1_flat.shape}")
        else:
            X1_flat = X1
            
        if len(X2.shape) > 2:
            X2_flat = X2.reshape(X2.shape[0], -1)
            print(f"   📊 {dataset2_name} - Imagens flatten: {X2.shape} -> {X2_flat.shape}")
        else:
            X2_flat = X2
        
        # 1. Kolmogorov-Smirnov Test
        ks_results, ks_summary = self.kolmogorov_smirnov_test(X1_flat, X2_flat)
        
        # 2. Wasserstein Distance
        wasserstein_results = self.wasserstein_distance_analysis(X1_flat, X2_flat)
        
        # 3. KL Divergence
        kl_results = self.kl_divergence_analysis(X1_flat, X2_flat)
        
        # 4. Análise por classe (se rótulos disponíveis)
        if y1 is not None and y2 is not None:
            class_results = self.class_level_drift_analysis(X1_flat, y1, X2_flat, y2)
        
        # 5. Estatísticas gerais dos datasets
        dataset_stats = self._compute_dataset_statistics(X1_flat, X2_flat, dataset1_name, dataset2_name)
        
        # 6. Resumo executivo
        executive_summary = self._generate_executive_summary()
        
        self.results['summary_statistics'] = {
            'dataset_statistics': dataset_stats,
            'executive_summary': executive_summary
        }
        
        print(f"\n✅ ANÁLISE ESTATÍSTICA CONCLUÍDA")
        print("=" * 50)
        
        return self.results
    
    def _compute_dataset_statistics(self, X1, X2, name1, name2):
        """Calcula estatísticas descritivas dos datasets."""
        stats1 = {
            'name': name1,
            'n_samples': X1.shape[0],
            'n_features': X1.shape[1],
            'mean_values': np.mean(X1, axis=0),
            'std_values': np.std(X1, axis=0),
            'feature_ranges': np.ptp(X1, axis=0)
        }
        
        stats2 = {
            'name': name2,
            'n_samples': X2.shape[0],
            'n_features': X2.shape[1],
            'mean_values': np.mean(X2, axis=0),
            'std_values': np.std(X2, axis=0),
            'feature_ranges': np.ptp(X2, axis=0)
        }
        
        # Comparações
        mean_diff = np.mean(np.abs(stats1['mean_values'] - stats2['mean_values']))
        std_diff = np.mean(np.abs(stats1['std_values'] - stats2['std_values']))
        
        comparison = {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'sample_size_ratio': stats1['n_samples'] / stats2['n_samples']
        }
        
        return {
            'dataset1': stats1,
            'dataset2': stats2,
            'comparison': comparison
        }
    
    def _generate_executive_summary(self):
        """Gera resumo executivo da análise de drift."""
        
        # Extrair métricas principais
        ks_summary = self.results['statistical_tests']['kolmogorov_smirnov']['summary']
        wasserstein = self.results['distribution_distances']['wasserstein']
        kl_results = self.results['distribution_distances']['kl_divergence']
        
        avg_drift = ks_summary['average_drift_percentage']
        avg_wasserstein = wasserstein['average_distance']
        avg_kl = kl_results['average_kl']
        
        # Determinar nível geral de drift
        if avg_drift > 80:
            drift_level = "CRÍTICO"
            drift_color = "🔴"
        elif avg_drift > 60:
            drift_level = "ALTO"
            drift_color = "🟡"
        elif avg_drift > 30:
            drift_level = "MODERADO"
            drift_color = "🟠"
        else:
            drift_level = "BAIXO"
            drift_color = "🟢"
        
        # Interpretação científica
        scientific_interpretation = self._generate_scientific_interpretation(avg_drift, avg_wasserstein, avg_kl)
        
        summary = {
            'drift_level': drift_level,
            'drift_color': drift_color,
            'key_metrics': {
                'ks_drift_percentage': avg_drift,
                'wasserstein_distance': avg_wasserstein,
                'kl_divergence': avg_kl
            },
            'scientific_interpretation': scientific_interpretation,
            'recommendations': self._generate_recommendations(avg_drift, avg_wasserstein)
        }
        
        return summary
    
    def _generate_scientific_interpretation(self, ks_drift, wasserstein, kl_div):
        """Gera interpretação científica dos resultados."""
        
        interpretation = []
        
        # Interpretação KS
        if ks_drift > 90:
            interpretation.append(f"KS Drift de {ks_drift:.1f}% indica distribuições completamente diferentes entre os datasets.")
        elif ks_drift > 70:
            interpretation.append(f"KS Drift de {ks_drift:.1f}% sugere diferenças substanciais nas distribuições.")
        else:
            interpretation.append(f"KS Drift de {ks_drift:.1f}% indica diferenças moderadas nas distribuições.")
        
        # Interpretação Wasserstein
        if wasserstein > 0.3:
            interpretation.append(f"Wasserstein Distance de {wasserstein:.4f} confirma que os datasets ocupam espaços feature distintos.")
        elif wasserstein > 0.15:
            interpretation.append(f"Wasserstein Distance de {wasserstein:.4f} indica diferenças significativas no espaço feature.")
        else:
            interpretation.append(f"Wasserstein Distance de {wasserstein:.4f} sugere similaridade relativa entre datasets.")
        
        # Interpretação KL
        if kl_div > 1.0:
            interpretation.append(f"KL Divergence de {kl_div:.4f} sugere que um dataset não pode ser usado como proxy do outro.")
        elif kl_div > 0.5:
            interpretation.append(f"KL Divergence de {kl_div:.4f} indica assimetria informacional entre datasets.")
        else:
            interpretation.append(f"KL Divergence de {kl_div:.4f} sugere compatibilidade informacional razoável.")
        
        return " ".join(interpretation)
    
    def _generate_recommendations(self, ks_drift, wasserstein):
        """Gera recomendações baseadas no nível de drift detectado."""
        
        recommendations = []
        
        if ks_drift > 80:
            recommendations.extend([
                "Evitar transferência direta de modelos entre datasets",
                "Implementar técnicas de domain adaptation",
                "Considerar feature engineering específico para cada dataset",
                "Usar ensemble methods que considerem múltiplos domínios"
            ])
        elif ks_drift > 50:
            recommendations.extend([
                "Aplicar técnicas de normalização robustas",
                "Considerar domain adaptation leve",
                "Validar performance com cross-dataset evaluation",
                "Monitorar drift em produção"
            ])
        else:
            recommendations.extend([
                "Datasets podem ser usados em conjunto com cautela",
                "Aplicar normalização padrão",
                "Monitorar performance em validação cruzada"
            ])
        
        return recommendations

# ================== VISUALIZADOR SIMPLIFICADO ==================

class DriftVisualizationEngine:
    """Engine de visualização para análise de drift."""
    
    def __init__(self, config=DRIFT_ANALYSIS_CONFIG):
        self.config = config
        self.fig_size = config['visualization']['figure_size']
        self.dpi = config['visualization']['dpi']
        
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
    
    def create_summary_visualization(self, statistical_results, dataset1_name="JAFFE", dataset2_name="CK+"):
        """Cria visualização resumida dos resultados de drift."""
        print("🎨 Criando visualização de drift analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Cross-Dataset Drift Analysis: {dataset1_name} vs {dataset2_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Métricas principais
        self._plot_main_metrics(axes[0, 0], statistical_results)
        
        # 2. Distribuição de drift por features
        self._plot_drift_distribution(axes[0, 1], statistical_results)
        
        # 3. Análise por classe
        self._plot_class_analysis(axes[1, 0], statistical_results)
        
        # 4. Resumo executivo
        self._plot_executive_summary(axes[1, 1], statistical_results)
        
        plt.tight_layout()
        return fig
    
    def _plot_main_metrics(self, ax, results):
        """Plot das métricas principais."""
        executive_summary = results['summary_statistics']['executive_summary']
        
        metrics = ['KS Drift (%)', 'Wasserstein', 'KL Divergence']
        values = [
            executive_summary['key_metrics']['ks_drift_percentage'],
            executive_summary['key_metrics']['wasserstein_distance'] * 100,
            executive_summary['key_metrics']['kl_divergence'] * 100
        ]
        
        colors = ['#E74C3C', '#F39C12', '#9B59B6']
        bars = ax.bar(metrics, values, color=colors, alpha=0.8)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Metric Value')
        ax.set_title('Principal Drift Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_drift_distribution(self, ax, results):
        """Plot da distribuição de drift por features."""
        ks_results = results['statistical_tests']['kolmogorov_smirnov']['detailed_results']
        drift_percentages = ks_results['drift_percentages']
        
        # Histogram de drift
        ax.hist(drift_percentages, bins=30, alpha=0.7, color='#3498DB', edgecolor='black')
        ax.axvline(np.mean(drift_percentages), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(drift_percentages):.1f}%')
        ax.axvline(np.median(drift_percentages), color='orange', linestyle='--', 
                  label=f'Median: {np.median(drift_percentages):.1f}%')
        
        ax.set_xlabel('Drift Percentage (%)')
        ax.set_ylabel('Number of Features')
        ax.set_title('Distribution of Feature-Level Drift')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_class_analysis(self, ax, results):
        """Plot da análise por classe."""
        if 'class_level_analysis' not in results or not results['class_level_analysis']:
            ax.text(0.5, 0.5, 'Class analysis not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Class-Level Analysis')
            return
        
        class_results = results['class_level_analysis']
        classes = list(class_results.keys())
        drift_values = [class_results[cls]['drift_percentage'] for cls in classes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = ax.bar(range(len(classes)), drift_values, color=colors, alpha=0.8)
        
        ax.set_xlabel('Emotion Classes')
        ax.set_ylabel('Drift Percentage (%)')
        ax.set_title('Class-Level Drift Analysis')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([cls.capitalize() for cls in classes], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Linha de drift médio
        avg_drift = np.mean(drift_values)
        ax.axhline(y=avg_drift, color='red', linestyle='--', alpha=0.7,
                  label=f'Average: {avg_drift:.1f}%')
        ax.legend()
    
    def _plot_executive_summary(self, ax, results):
        """Plot do resumo executivo."""
        executive_summary = results['summary_statistics']['executive_summary']
        
        # Remover eixos
        ax.axis('off')
        
        # Título
        ax.text(0.5, 0.9, 'Executive Summary', ha='center', va='top', 
               fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        # Nível de drift
        drift_level = executive_summary['drift_level']
        drift_color = executive_summary['drift_color']
        ax.text(0.5, 0.75, f'{drift_color} Drift Level: {drift_level}', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Métricas principais
        metrics_text = (
            f"KS Drift: {executive_summary['key_metrics']['ks_drift_percentage']:.1f}%\n"
            f"Wasserstein: {executive_summary['key_metrics']['wasserstein_distance']:.4f}\n"
            f"KL Divergence: {executive_summary['key_metrics']['kl_divergence']:.4f}"
        )
        ax.text(0.5, 0.55, metrics_text, ha='center', va='center', 
               fontsize=10, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Recomendações principais
        recommendations = executive_summary['recommendations'][:3]  # Top 3
        rec_text = "Top Recommendations:\n" + "\n".join([f"• {rec[:40]}..." for rec in recommendations])
        ax.text(0.5, 0.25, rec_text, ha='center', va='center', 
               fontsize=9, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# ================== PIPELINE MULTI-DATASET ==================

def run_multi_dataset_drift_analysis(datasets_dict, save_results=True, output_dir="multi_dataset_drift_analysis"):
    """
    Pipeline para análise completa de drift entre múltiplos datasets.
    
    Parameters:
    -----------
    datasets_dict : dict
        Dicionário com formato: {nome: (X, y)}
    save_results : bool
        Se deve salvar os resultados
    output_dir : str
        Diretório para salvar resultados
    
    Returns:
    --------
    dict : Resultados completos da análise multi-dataset
    """
    
    dataset_names = list(datasets_dict.keys())
    n_datasets = len(dataset_names)
    
    print(f"\n🚀 INICIANDO ANÁLISE MULTI-DATASET DRIFT")
    print(f"📊 Datasets: {', '.join(dataset_names)}")
    print("=" * 100)
    
    all_results = {}
    comparison_matrix = {}
    
    # Análise cruzada entre todos os pares de datasets
    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            dataset1_name = dataset_names[i]
            dataset2_name = dataset_names[j]
            
            X1, y1 = datasets_dict[dataset1_name]
            X2, y2 = datasets_dict[dataset2_name]
            
            print(f"\n🔬 Analisando: {dataset1_name} vs {dataset2_name}")
            print("-" * 60)
            
            # Executar análise bilateral
            results = run_comprehensive_drift_analysis(
                X1, X2, y1, y2,
                dataset1_name=dataset1_name,
                dataset2_name=dataset2_name,
                save_results=save_results,
                output_dir=os.path.join(output_dir, f"{dataset1_name}_vs_{dataset2_name}")
            )
            
            comparison_key = f"{dataset1_name}_vs_{dataset2_name}"
            all_results[comparison_key] = results
            
            # Extrair métricas para matriz de comparação
            executive_summary = results['statistical_results']['summary_statistics']['executive_summary']
            comparison_matrix[comparison_key] = {
                'ks_drift': executive_summary['key_metrics']['ks_drift_percentage'],
                'wasserstein': executive_summary['key_metrics']['wasserstein_distance'],
                'kl_divergence': executive_summary['key_metrics']['kl_divergence'],
                'drift_level': executive_summary['drift_level']
            }
    
    # Criar análise consolidada
    print(f"\n📊 FASE FINAL: Análise Consolidada Multi-Dataset")
    print("-" * 60)
    
    consolidated_results = create_consolidated_analysis(
        all_results, comparison_matrix, dataset_names, 
        save_results, output_dir
    )
    
    return {
        'individual_results': all_results,
        'comparison_matrix': comparison_matrix,
        'consolidated_analysis': consolidated_results,
        'output_directory': output_dir if save_results else None
    }

def run_comprehensive_drift_analysis(X1, X2, y1, y2, 
                                    dataset1_name="JAFFE", dataset2_name="CK+",
                                    save_results=True, output_dir="drift_analysis_results"):
    """
    Pipeline principal para análise completa de drift entre dois datasets.
    """
    
    print(f"\n🔬 ANÁLISE: {dataset1_name} vs {dataset2_name}")
    
    # 1. Análise Estatística
    analyzer = StatisticalDriftAnalyzer()
    statistical_results = analyzer.comprehensive_statistical_analysis(
        X1, X2, y1, y2, dataset1_name, dataset2_name
    )
    
    # 2. Visualizações
    visualizer = DriftVisualizationEngine()
    analysis_fig = visualizer.create_summary_visualization(
        statistical_results, dataset1_name, dataset2_name
    )
    
    # 3. Salvar Resultados
    if save_results:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Salvar visualização
        fig_path = os.path.join(output_dir, f"{dataset1_name}_vs_{dataset2_name}_drift_analysis.png")
        analysis_fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # Salvar resultados estatísticos
        results_file = os.path.join(output_dir, f"{dataset1_name}_vs_{dataset2_name}_statistical_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(statistical_results, f)
        
        # Salvar relatório executivo
        report_file = os.path.join(output_dir, f"{dataset1_name}_vs_{dataset2_name}_executive_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(generate_executive_report(statistical_results, dataset1_name, dataset2_name))
    
    # 4. Resumo
    executive_summary = statistical_results['summary_statistics']['executive_summary']
    print(f"   🎯 Drift: {executive_summary['drift_color']} {executive_summary['drift_level']}")
    print(f"   📈 KS: {executive_summary['key_metrics']['ks_drift_percentage']:.1f}%")
    print(f"   📈 Wasserstein: {executive_summary['key_metrics']['wasserstein_distance']:.4f}")
    print(f"   📈 KL: {executive_summary['key_metrics']['kl_divergence']:.4f}")
    
    plt.close(analysis_fig)  # Fechar figura para economizar memória
    
    return {
        'statistical_results': statistical_results,
        'visualization_figure': analysis_fig,
        'output_directory': output_dir if save_results else None
    }

def create_consolidated_analysis(all_results, comparison_matrix, dataset_names, 
                               save_results=True, output_dir="multi_dataset_drift_analysis"):
    """
    Cria análise consolidada de todos os resultados multi-dataset.
    """
    
    print("📊 Criando análise consolidada...")
    
    # 1. Matriz de comparação
    drift_matrix = create_drift_comparison_matrix(comparison_matrix, dataset_names)
    
    # 2. Ranking de similaridade
    similarity_ranking = create_similarity_ranking(comparison_matrix, dataset_names)
    
    # 3. Visualização consolidada
    consolidated_fig = create_consolidated_visualization(
        drift_matrix, similarity_ranking, dataset_names, comparison_matrix
    )
    
    # 4. Relatório consolidado
    consolidated_report = generate_consolidated_report(
        comparison_matrix, similarity_ranking, dataset_names
    )
    
    if save_results:
        # Salvar visualização consolidada
        fig_path = os.path.join(output_dir, "consolidated_drift_analysis.png")
        consolidated_fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Análise consolidada salva: {fig_path}")
        
        # Salvar relatório consolidado
        report_path = os.path.join(output_dir, "consolidated_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(consolidated_report)
        print(f"   📄 Relatório consolidado salvo: {report_path}")
        
        # Salvar matriz de drift
        matrix_path = os.path.join(output_dir, "drift_comparison_matrix.csv")
        drift_matrix.to_csv(matrix_path)
        print(f"   📊 Matriz de drift salva: {matrix_path}")
    
    return {
        'drift_matrix': drift_matrix,
        'similarity_ranking': similarity_ranking,
        'consolidated_figure': consolidated_fig,
        'consolidated_report': consolidated_report
    }

def create_drift_comparison_matrix(comparison_matrix, dataset_names):
    """
    Cria matriz de comparação de drift entre datasets.
    """
    n_datasets = len(dataset_names)
    
    # Matrizes para diferentes métricas
    ks_matrix = np.zeros((n_datasets, n_datasets))
    wasserstein_matrix = np.zeros((n_datasets, n_datasets))
    kl_matrix = np.zeros((n_datasets, n_datasets))
    
    for i, dataset1 in enumerate(dataset_names):
        for j, dataset2 in enumerate(dataset_names):
            if i == j:
                # Diagonal = 0 (dataset consigo mesmo)
                ks_matrix[i, j] = 0
                wasserstein_matrix[i, j] = 0
                kl_matrix[i, j] = 0
            elif i < j:
                # Parte superior da matriz
                key = f"{dataset1}_vs_{dataset2}"
                if key in comparison_matrix:
                    ks_matrix[i, j] = comparison_matrix[key]['ks_drift']
                    wasserstein_matrix[i, j] = comparison_matrix[key]['wasserstein']
                    kl_matrix[i, j] = comparison_matrix[key]['kl_divergence']
            else:
                # Parte inferior da matriz (simétrica)
                key = f"{dataset2}_vs_{dataset1}"
                if key in comparison_matrix:
                    ks_matrix[i, j] = comparison_matrix[key]['ks_drift']
                    wasserstein_matrix[i, j] = comparison_matrix[key]['wasserstein']
                    kl_matrix[i, j] = comparison_matrix[key]['kl_divergence']
                else:
                    # Se não existe, usar a simétrica
                    ks_matrix[i, j] = ks_matrix[j, i]
                    wasserstein_matrix[i, j] = wasserstein_matrix[j, i]
                    kl_matrix[i, j] = kl_matrix[j, i]
    
    # Criar DataFrame consolidado
    matrices_data = []
    for i, dataset1 in enumerate(dataset_names):
        for j, dataset2 in enumerate(dataset_names):
            matrices_data.append({
                'Dataset1': dataset1,
                'Dataset2': dataset2,
                'KS_Drift': ks_matrix[i, j],
                'Wasserstein_Distance': wasserstein_matrix[i, j],
                'KL_Divergence': kl_matrix[i, j]
            })
    
    return pd.DataFrame(matrices_data)

def create_similarity_ranking(comparison_matrix, dataset_names):
    """
    Cria ranking de similaridade entre datasets.
    """
    similarities = []
    
    for key, metrics in comparison_matrix.items():
        dataset1, dataset2 = key.split('_vs_')
        
        # Score de similaridade (inverso do drift)
        # Normalizar métricas e calcular score composto
        ks_norm = max(0, 100 - metrics['ks_drift']) / 100
        wasserstein_norm = max(0, 1 - metrics['wasserstein'])
        kl_norm = max(0, 1 - metrics['kl_divergence'])
        
        similarity_score = (ks_norm + wasserstein_norm + kl_norm) / 3
        
        similarities.append({
            'Comparison': key,
            'Dataset1': dataset1,
            'Dataset2': dataset2,
            'Similarity_Score': similarity_score,
            'KS_Drift': metrics['ks_drift'],
            'Wasserstein': metrics['wasserstein'],
            'KL_Divergence': metrics['kl_divergence'],
            'Drift_Level': metrics['drift_level']
        })
    
    # Ordenar por score de similaridade (maior = mais similar)
    similarity_df = pd.DataFrame(similarities)
    similarity_df = similarity_df.sort_values('Similarity_Score', ascending=False)
    
    return similarity_df

def create_consolidated_visualization(drift_matrix, similarity_ranking, dataset_names, comparison_matrix):
    """
    Cria visualização consolidada de todos os resultados.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Multi-Dataset Drift Analysis - Consolidated View', fontsize=16, fontweight='bold')
    
    # 1. Heatmap de KS Drift
    ks_pivot = drift_matrix.pivot(index='Dataset1', columns='Dataset2', values='KS_Drift')
    sns.heatmap(ks_pivot, annot=True, fmt='.1f', cmap='Reds', ax=axes[0, 0])
    axes[0, 0].set_title('KS Drift Matrix (%)')
    
    # 2. Heatmap de Wasserstein Distance
    wasserstein_pivot = drift_matrix.pivot(index='Dataset1', columns='Dataset2', values='Wasserstein_Distance')
    sns.heatmap(wasserstein_pivot, annot=True, fmt='.3f', cmap='Oranges', ax=axes[0, 1])
    axes[0, 1].set_title('Wasserstein Distance Matrix')
    
    # 3. Heatmap de KL Divergence
    kl_pivot = drift_matrix.pivot(index='Dataset1', columns='Dataset2', values='KL_Divergence')
    sns.heatmap(kl_pivot, annot=True, fmt='.3f', cmap='Purples', ax=axes[0, 2])
    axes[0, 2].set_title('KL Divergence Matrix')
    
    # 4. Ranking de Similaridade
    similarity_ranking_top = similarity_ranking.head(10)  # Top 10
    bars = axes[1, 0].barh(similarity_ranking_top['Comparison'], 
                          similarity_ranking_top['Similarity_Score'], 
                          color='skyblue', alpha=0.8)
    axes[1, 0].set_xlabel('Similarity Score')
    axes[1, 0].set_title('Dataset Similarity Ranking')
    axes[1, 0].set_xlim(0, 1)
    
    # 5. Distribuição de Drift Levels
    drift_levels = [metrics['drift_level'] for metrics in comparison_matrix.values()]
    level_counts = pd.Series(drift_levels).value_counts()
    axes[1, 1].pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Distribution of Drift Levels')
    
    # 6. Scatter Plot: KS vs Wasserstein
    ks_values = [metrics['ks_drift'] for metrics in comparison_matrix.values()]
    wasserstein_values = [metrics['wasserstein'] for metrics in comparison_matrix.values()]
    
    axes[1, 2].scatter(ks_values, wasserstein_values, alpha=0.7, s=100, c='green')
    axes[1, 2].set_xlabel('KS Drift (%)')
    axes[1, 2].set_ylabel('Wasserstein Distance')
    axes[1, 2].set_title('KS Drift vs Wasserstein Distance')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Adicionar labels nos pontos
    for i, key in enumerate(comparison_matrix.keys()):
        axes[1, 2].annotate(key.replace('_vs_', '\nvs\n'), 
                           (ks_values[i], wasserstein_values[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    return fig

def generate_consolidated_report(comparison_matrix, similarity_ranking, dataset_names):
    """
    Gera relatório consolidado da análise multi-dataset.
    """
    
    report = f"""
# MULTI-DATASET DRIFT ANALYSIS - CONSOLIDATED REPORT

## EXECUTIVE SUMMARY
This report presents a comprehensive analysis of dataset drift across {len(dataset_names)} datasets:
{', '.join(dataset_names)}

## OVERALL STATISTICS
- **Total Comparisons**: {len(comparison_matrix)}
- **Average KS Drift**: {np.mean([m['ks_drift'] for m in comparison_matrix.values()]):.2f}%
- **Average Wasserstein Distance**: {np.mean([m['wasserstein'] for m in comparison_matrix.values()]):.4f}
- **Average KL Divergence**: {np.mean([m['kl_divergence'] for m in comparison_matrix.values()]):.4f}

## SIMILARITY RANKING
Most similar dataset pairs:
"""
    
    for i, row in similarity_ranking.head(5).iterrows():
        report += f"{i+1}. {row['Comparison']}: Similarity Score = {row['Similarity_Score']:.3f}\n"
    
    report += f"""
## DRIFT LEVEL DISTRIBUTION
"""
    
    drift_levels = [metrics['drift_level'] for metrics in comparison_matrix.values()]
    level_counts = pd.Series(drift_levels).value_counts()
    
    for level, count in level_counts.items():
        percentage = (count / len(drift_levels)) * 100
        report += f"- **{level}**: {count} comparisons ({percentage:.1f}%)\n"
    
    report += f"""
## DETAILED COMPARISON MATRIX

| Comparison | KS Drift (%) | Wasserstein | KL Divergence | Drift Level |
|------------|--------------|-------------|---------------|-------------|
"""
    
    for key, metrics in comparison_matrix.items():
        report += f"| {key} | {metrics['ks_drift']:.2f} | {metrics['wasserstein']:.4f} | {metrics['kl_divergence']:.4f} | {metrics['drift_level']} |\n"
    
    report += f"""
## RECOMMENDATIONS

### High-Level Insights:
1. **Dataset Compatibility**: Use similarity ranking to identify most compatible dataset pairs
2. **Transfer Learning**: Lower drift indicates better potential for cross-dataset model transfer
3. **Domain Adaptation**: High drift levels suggest need for domain adaptation techniques

### Technical Recommendations:
- For CRITICAL drift: Implement strong domain adaptation or train separate models
- For HIGH drift: Apply domain adaptation techniques and careful validation
- For MODERATE drift: Use standard transfer learning with monitoring
- For LOW drift: Datasets can be combined with standard preprocessing

Generated by Multi-Dataset Drift Analysis Engine
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report

def generate_executive_report(statistical_results, dataset1_name, dataset2_name):
    """Gera relatório executivo em formato texto."""
    executive_summary = statistical_results['summary_statistics']['executive_summary']
    
    report = f"""
# CROSS-DATASET DRIFT ANALYSIS REPORT
## {dataset1_name} vs {dataset2_name}

### EXECUTIVE SUMMARY
- **Drift Level**: {executive_summary['drift_level']}
- **Overall Assessment**: {executive_summary['drift_color']} 

### KEY METRICS
- **Kolmogorov-Smirnov Drift**: {executive_summary['key_metrics']['ks_drift_percentage']:.2f}%
- **Wasserstein Distance**: {executive_summary['key_metrics']['wasserstein_distance']:.4f}
- **Kullback-Leibler Divergence**: {executive_summary['key_metrics']['kl_divergence']:.4f}

### SCIENTIFIC INTERPRETATION
{executive_summary['scientific_interpretation']}

### RECOMMENDATIONS
"""
    
    for i, rec in enumerate(executive_summary['recommendations'], 1):
        report += f"{i}. {rec}\n"
    
    dataset_stats = statistical_results['summary_statistics']['dataset_statistics']
    
    report += f"""
### DATASET STATISTICS
- **{dataset1_name}**: {dataset_stats['dataset1']['n_samples']} samples, {dataset_stats['dataset1']['n_features']} features
- **{dataset2_name}**: {dataset_stats['dataset2']['n_samples']} samples, {dataset_stats['dataset2']['n_features']} features
- **Sample Size Ratio**: {dataset_stats['comparison']['sample_size_ratio']:.2f}
- **Mean Difference**: {dataset_stats['comparison']['mean_difference']:.4f}
- **Std Difference**: {dataset_stats['comparison']['std_difference']:.4f}

### TECHNICAL DETAILS
- **Analysis Method**: Multi-metric statistical drift detection
- **Statistical Tests**: Kolmogorov-Smirnov, Wasserstein Distance, KL Divergence
- **Confidence Level**: 95%
- **Sampling Strategy**: Optimized for large datasets

Generated by Advanced Drift Analysis Engine
"""
    
    return report

# ================== SCRIPT PRINCIPAL MULTI-DATASET ==================


if __name__ == "__main__":
    print("🔬 Multi-Dataset Cross-Dataset Drift Analysis")
    print("=" * 60)
    
    # Definir caminhos dos datasets
    jaffe_path = r'.\data\jaffe'
    ck_path = r'.\data\ck'
    combined_cross_path = r'.\data\combined_cross_balanced'
    
    dataset_paths = {
        'JAFFE': jaffe_path,
        'CK+': ck_path,
        'Combined_Cross_Balanced': combined_cross_path
    }
    
    # Verificar se os diretórios existem
    valid_datasets = {}
    
    for name, path in dataset_paths.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
            valid_datasets[name] = path
        else:
            print(f"❌ {name} não encontrado: {path}")
            # Tentar ajustar caminho automaticamente
            corrected_path = input(f"Digite o caminho correto para {name} (ou ENTER para pular): ").strip()
            if corrected_path and os.path.exists(corrected_path):
                valid_datasets[name] = corrected_path
                print(f"✅ {name} corrigido: {corrected_path}")
    
    if len(valid_datasets) < 2:
        print("❌ Erro: Pelo menos 2 datasets são necessários para análise de drift")
        exit(1)
    
    try:
        # Carregar todos os datasets válidos
        print(f"\n📁 Carregando {len(valid_datasets)} datasets...")
        loaded_datasets = {}
        
        for name, path in valid_datasets.items():
            print(f"\n📊 Carregando {name} de: {path}")
            X, y = load_images_from_directory(path, image_size=(96, 96))
            
            if len(X) == 0:
                print(f"⚠️ Aviso: Nenhuma imagem carregada para {name}")
                continue
                
            loaded_datasets[name] = (X, y)
            print(f"   ✅ {name}: {len(X)} imagens, shape: {X.shape}")
            print(f"   📂 Classes: {sorted(set(y))}")
        
        if len(loaded_datasets) < 2:
            raise ValueError("Menos de 2 datasets carregados com sucesso")
        
        print(f"\n✅ Todos os datasets carregados com sucesso!")
        print(f"📊 Total de datasets: {len(loaded_datasets)}")
        print(f"📂 Datasets carregados: {list(loaded_datasets.keys())}")
        
        # Executar análise multi-dataset
        print(f"\n🚀 Iniciando análise cruzada entre todos os datasets...")
        
        multi_results = run_multi_dataset_drift_analysis(
            datasets_dict=loaded_datasets,
            save_results=True,
            output_dir="multi_dataset_comprehensive_analysis"
        )
        
        print(f"\n🎉 ANÁLISE MULTI-DATASET COMPLETA!")
        print("=" * 60)
        
        # Resumo dos resultados
        comparison_matrix = multi_results['comparison_matrix']
        
        print(f"📊 RESUMO EXECUTIVO:")
        print(f"   🔍 Total de comparações: {len(comparison_matrix)}")
        
        # Estatísticas gerais
        ks_drifts = [m['ks_drift'] for m in comparison_matrix.values()]
        wasserstein_dists = [m['wasserstein'] for m in comparison_matrix.values()]
        kl_divs = [m['kl_divergence'] for m in comparison_matrix.values()]
        
        print(f"   📈 KS Drift médio: {np.mean(ks_drifts):.1f}% (min: {min(ks_drifts):.1f}%, max: {max(ks_drifts):.1f}%)")
        print(f"   📈 Wasserstein médio: {np.mean(wasserstein_dists):.4f} (min: {min(wasserstein_dists):.4f}, max: {max(wasserstein_dists):.4f})")
        print(f"   📈 KL Divergence médio: {np.mean(kl_divs):.4f} (min: {min(kl_divs):.4f}, max: {max(kl_divs):.4f})")
        
        # Mostrar par mais e menos similar
        similarity_ranking = multi_results['consolidated_analysis']['similarity_ranking']
        
        print(f"\n🏆 PARES DE DATASETS:")
        most_similar = similarity_ranking.iloc[0]
        least_similar = similarity_ranking.iloc[-1]
        
        print(f"   🟢 Mais similares: {most_similar['Comparison']} (Score: {most_similar['Similarity_Score']:.3f})")
        print(f"   🔴 Menos similares: {least_similar['Comparison']} (Score: {least_similar['Similarity_Score']:.3f})")
        
        # Distribuição de drift levels
        drift_levels = [m['drift_level'] for m in comparison_matrix.values()]
        level_counts = pd.Series(drift_levels).value_counts()
        
        print(f"\n📊 DISTRIBUIÇÃO DE DRIFT LEVELS:")
        for level, count in level_counts.items():
            percentage = (count / len(drift_levels)) * 100
            emoji = {'BAIXO': '🟢', 'MODERADO': '🟡', 'ALTO': '🟠', 'CRÍTICO': '🔴'}.get(level, '⚪')
            print(f"   {emoji} {level}: {count} comparações ({percentage:.1f}%)")
        
        print(f"\n💾 Resultados salvos em: {multi_results['output_directory']}")
        print(f"📊 Análise consolidada disponível em: consolidated_drift_analysis.png")
        print(f"📄 Relatório completo em: consolidated_report.txt")
        
        # Mostrar visualização consolidada
        plt.show()
        
        print(f"\n✨ Análise multi-dataset concluída com sucesso!")
        
        # Recomendações finais baseadas nos resultados
        print(f"\n🎯 RECOMENDAÇÕES PRINCIPAIS:")
        
        avg_ks = np.mean(ks_drifts)
        if avg_ks > 80:
            print("   🔴 DRIFT CRÍTICO: Recomenda-se domain adaptation robusta")
        elif avg_ks > 60:
            print("   🟡 DRIFT ALTO: Considerar técnicas de domain adaptation")
        elif avg_ks > 30:
            print("   🟠 DRIFT MODERADO: Transfer learning com cuidado")
        else:
            print("   🟢 DRIFT BAIXO: Datasets podem ser combinados")
        
        print("   📚 Para detalhes técnicos, consulte os relatórios individuais")
        print("   📊 Use a matriz de similaridade para escolher pares compatíveis")
        
    except FileNotFoundError as e:
        print(f"❌ Erro de arquivo: {e}")
        print("📝 Verifique os caminhos dos datasets")
    except ValueError as e:
        print(f"❌ Erro nos dados: {e}")
        print("📝 Verifique se os datasets contêm imagens válidas")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()

# ================== SCRIPT PRINCIPAL ==================

if __name__ == "__main__":
    print("🔬 Cross-Dataset Drift Analysis - Versão Corrigida")
    print("=" * 60)
    
    # Definir caminhos dos datasets
    jaffe_path = r'.\data\jaffe'  # Ajuste conforme necessário
    ck_path = r'.\data\ck'  # Ajuste conforme necessário
    
    # Verificar se os diretórios existem
    if not os.path.exists(jaffe_path):
        print(f"❌ Diretório JAFFE não encontrado: {jaffe_path}")
        print("📝 Ajuste o caminho na variável 'jaffe_path'")
        jaffe_path = input("Digite o caminho correto para JAFFE augmented: ").strip()
    
    if not os.path.exists(ck_path):
        print(f"❌ Diretório CK+ não encontrado: {ck_path}")
        print("📝 Ajuste o caminho na variável 'ck_path'")
        ck_path = input("Digite o caminho correto para CK+: ").strip()
    
    try:
        # 1. Carregar dados JAFFE
        print(f"\n📁 Carregando JAFFE Augmented de: {jaffe_path}")
        X_jaffe, y_jaffe = load_images_from_directory(jaffe_path, image_size=(96, 96))
        
        # 2. Carregar dados CK+
        print(f"\n📁 Carregando CK+ de: {ck_path}")
        X_ck, y_ck = load_images_from_directory(ck_path, image_size=(96, 96))
        
        # 3. Verificar se os dados foram carregados
        if len(X_jaffe) == 0:
            raise ValueError("Nenhuma imagem foi carregada do dataset JAFFE")
        if len(X_ck) == 0:
            raise ValueError("Nenhuma imagem foi carregada do dataset CK+")
        
        print(f"\n✅ Dados carregados com sucesso!")
        print(f"   📊 JAFFE: {len(X_jaffe)} imagens, shape: {X_jaffe.shape}")
        print(f"   📊 CK+: {len(X_ck)} imagens, shape: {X_ck.shape}")
        print(f"   📂 Classes JAFFE: {sorted(set(y_jaffe))}")
        print(f"   📂 Classes CK+: {sorted(set(y_ck))}")
        
        # 4. Executar análise completa
        results = run_comprehensive_drift_analysis(
            X_jaffe, X_ck, 
            y_jaffe, y_ck,
            dataset1_name="JAFFE_Augmented", 
            dataset2_name="CK+",
            save_results=True,
            output_dir="comprehensive_drift_analysis"
        )
        
        # 5. Mostrar visualização
        plt.show()
        
        print(f"\n🎉 Análise completa finalizada!")
        print(f"📁 Resultados salvos em: {results['output_directory']}")
        
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
        print("📝 Verifique os caminhos dos datasets")
    except ValueError as e:
        print(f"❌ Erro nos dados: {e}")
        print("📝 Verifique se os datasets contêm imagens válidas")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()

print("✅ Drift Analysis Pipeline implementado com sucesso!")
print("🎯 Execute o script para analisar drift entre JAFFE e CK+!")