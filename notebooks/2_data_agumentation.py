# CÓDIGO EXECUTÁVEL - Cross-Dataset Data Augmentation
# Para equilibrar JAFFE com CK+ usando oversampling inteligente

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import shutil
from tqdm import tqdm
import random
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

# ================== CONFIGURAÇÕES ==================
# Configuração específica para seu caso (JAFFE → CK+ balance)
CROSS_DATASET_CONFIG = {
    'estrategia': 'cross_dataset_balance',  # Equiparar ao dataset com mais amostras
    'rotation_angles': [-15, -10, -5, 5, 10, 15],
    'brightness_factors': [0.7, 0.8, 1.2, 1.3],
    'noise_levels': [0.01, 0.02, 0.03],
    'blur_kernels': [(3,3), (5,5)],
    'preserve_original': True,
    'min_augmentations_per_image': 1,
    'max_augmentations_per_image': 8
}

# Classes esperadas (suas 7 emoções)
EXPECTED_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
IMAGE_SIZE = (96, 96)

# Diretórios de entrada (ADAPTE PARA SEUS CAMINHOS)
DATASET_PATHS = {
    'jaffe': r'.\data\jaffe',  # Seu JAFFE original
    'ck+': r'.\data\ck'                  # Seu CK+ original
}

# Diretórios de saída (onde salvar os resultados)
OUTPUT_DIRS = {
    'jaffe_cross_balanced': r'.\data\_cross_balanced',
    'ck_cross_balanced': r'.\data\_cross_balanced', 
    'combined_cross_balanced': r'.\data\combined_cross_balanced'
}

# ================== FUNÇÕES PRINCIPAIS ==================

def verificar_distribuicao_atual():
    """Verifica a distribuição atual dos datasets"""
    print("🔍 VERIFICANDO DISTRIBUIÇÃO ATUAL DOS DATASETS")
    print("=" * 60)
    
    distribuicoes = {}
    
    for dataset_name, path in DATASET_PATHS.items():
        print(f"\n📊 {dataset_name.upper()}:")
        
        if not os.path.exists(path):
            print(f"   ❌ Caminho não encontrado: {path}")
            continue
        
        distribuicao = {}
        for classe in EXPECTED_CLASSES:
            class_path = os.path.join(path, classe)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                distribuicao[classe] = count
                print(f"   {classe:>10}: {count:>3} imagens")
            else:
                distribuicao[classe] = 0
                print(f"   {classe:>10}: {0:>3} imagens (pasta não encontrada)")
        
        distribuicoes[dataset_name] = distribuicao
    
    return distribuicoes

def calcular_metas_cross_dataset(distribuicoes):
    """Calcula as metas de balanceamento cross-dataset"""
    print(f"\n🎯 CALCULANDO METAS DE BALANCEAMENTO")
    print("=" * 50)
    
    metas = {}
    deficit_por_dataset = {}
    
    for classe in EXPECTED_CLASSES:
        # Meta = maior quantidade entre todos os datasets
        counts = [dist.get(classe, 0) for dist in distribuicoes.values()]
        meta = max(counts) if counts else 0
        metas[classe] = meta
        
        print(f"\n🎭 {classe.upper()}:")
        print(f"   Meta (máximo): {meta}")
        
        # Calcular déficit por dataset
        for dataset_name, distribuicao in distribuicoes.items():
            atual = distribuicao.get(classe, 0)
            deficit = max(0, meta - atual)
            
            if dataset_name not in deficit_por_dataset:
                deficit_por_dataset[dataset_name] = {}
            
            deficit_por_dataset[dataset_name][classe] = {
                'atual': atual,
                'meta': meta,
                'deficit': deficit
            }
            
            if deficit > 0:
                print(f"   {dataset_name:>6}: {atual:>3} → {meta:>3} (+{deficit:>3} needed)")
            else:
                print(f"   {dataset_name:>6}: {atual:>3} → {meta:>3} (✓ OK)")
    
    return metas, deficit_por_dataset

class SmartAugmenter:
    """Augmenter otimizado para expressões faciais"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    def horizontal_flip(self, image):
        """Flip horizontal preservando expressão"""
        return cv2.flip(image, 1)
    
    def rotate_image(self, image, angle):
        """Rotação suave"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), 
                             borderMode=cv2.BORDER_REFLECT101)
    
    def adjust_brightness(self, image, factor):
        """Ajuste de brilho gamma"""
        normalized = image.astype(np.float32) / 255.0
        adjusted = np.power(normalized, factor)
        return (adjusted * 255).astype(np.uint8)
    
    def add_noise(self, image, intensity=0.02):
        """Ruído gaussiano realístico"""
        noise = np.random.normal(0, intensity * 50, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def generate_augmentations(self, image, num_needed):
        """Gera augmentations inteligentes"""
        augmentations = []
        
        # Técnicas disponíveis com probabilidades
        techniques = [
            ('flip', 0.6),
            ('rotate', 0.5), 
            ('brightness', 0.4),
            ('noise', 0.3)
        ]
        
        for i in range(num_needed):
            aug_image = image.copy()
            applied_techniques = []
            
            # Aplicar técnicas randomicamente
            for technique, prob in techniques:
                if random.random() < prob:
                    
                    if technique == 'flip':
                        aug_image = self.horizontal_flip(aug_image)
                        applied_techniques.append('flip')
                    
                    elif technique == 'rotate':
                        angle = random.choice([-10, -5, 5, 10])
                        aug_image = self.rotate_image(aug_image, angle)
                        applied_techniques.append(f'rot{angle}')
                    
                    elif technique == 'brightness':
                        factor = random.choice([0.8, 0.9, 1.1, 1.2])
                        aug_image = self.adjust_brightness(aug_image, factor)
                        applied_techniques.append(f'bright{factor}')
                    
                    elif technique == 'noise':
                        aug_image = self.add_noise(aug_image)
                        applied_techniques.append('noise')
            
            # Se nenhuma técnica foi aplicada, fazer flip simples
            if not applied_techniques:
                aug_image = self.horizontal_flip(aug_image)
                applied_techniques = ['flip']
            
            augmentations.append((aug_image, '_'.join(applied_techniques)))
        
        return augmentations

def executar_augmentation_dataset(dataset_name, input_path, output_path, deficit_info):
    """Executa augmentation para um dataset específico"""
    print(f"\n🚀 EXECUTANDO AUGMENTATION: {dataset_name.upper()}")
    print("=" * 60)
    
    # Criar diretório de saída
    os.makedirs(output_path, exist_ok=True)
    
    # Inicializar augmenter
    augmenter = SmartAugmenter()
    
    # Estatísticas
    stats = {
        'originais_copiadas': 0,
        'augmentations_geradas': 0,
        'classes_processadas': 0
    }
    
    for classe in EXPECTED_CLASSES:
        info_classe = deficit_info.get(classe, {})
        deficit = info_classe.get('deficit', 0)
        atual = info_classe.get('atual', 0)
        meta = info_classe.get('meta', 0)
        
        print(f"\n📁 {classe.upper()}: {atual} → {meta} (+{deficit} needed)")
        
        # Caminhos da classe
        input_class_path = os.path.join(input_path, classe)
        output_class_path = os.path.join(output_path, classe)
        
        if not os.path.exists(input_class_path):
            print(f"   ⚠️ Pasta não encontrada: {input_class_path}")
            continue
        
        os.makedirs(output_class_path, exist_ok=True)
        
        # Listar imagens originais
        imagens_originais = [f for f in os.listdir(input_class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Copiar originais
        print(f"   📋 Copiando {len(imagens_originais)} imagens originais...")
        for img_file in imagens_originais:
            src = os.path.join(input_class_path, img_file)
            dst = os.path.join(output_class_path, f"orig_{img_file}")
            shutil.copy2(src, dst)
            stats['originais_copiadas'] += 1
        
        # Gerar augmentations se necessário
        if deficit > 0:
            print(f"   🎨 Gerando {deficit} augmentations...")
            
            # Distribuir augmentations pelas imagens disponíveis
            num_imagens = len(imagens_originais)
            augs_per_image = deficit // num_imagens
            augs_extras = deficit % num_imagens
            
            contador_geradas = 0
            
            for i, img_file in enumerate(tqdm(imagens_originais, desc="   Augmentando")):
                # Carregar imagem
                img_path = os.path.join(input_class_path, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    continue
                
                # Redimensionar
                image = cv2.resize(image, IMAGE_SIZE)
                
                # Calcular quantas augmentations para esta imagem
                num_augs = augs_per_image
                if i < augs_extras:
                    num_augs += 1
                
                # Gerar augmentations
                augmentations = augmenter.generate_augmentations(image, num_augs)
                
                # Salvar augmentations
                base_name = os.path.splitext(img_file)[0]
                for j, (aug_img, technique) in enumerate(augmentations):
                    aug_filename = f"aug_{base_name}_{j:02d}_{technique}.png"
                    aug_path = os.path.join(output_class_path, aug_filename)
                    
                    cv2.imwrite(aug_path, aug_img)
                    contador_geradas += 1
                    stats['augmentations_geradas'] += 1
                    
                    if contador_geradas >= deficit:
                        break
                
                if contador_geradas >= deficit:
                    break
            
            print(f"   ✅ Geradas {contador_geradas} augmentations")
        else:
            print(f"   ✅ Classe já balanceada")
        
        stats['classes_processadas'] += 1
    
    return stats

def criar_dataset_combinado(output_dirs):
    """Cria dataset combinado final"""
    print(f"\n🔄 CRIANDO DATASET COMBINADO FINAL")
    print("=" * 50)
    
    combined_path = output_dirs['combined_cross_balanced']
    os.makedirs(combined_path, exist_ok=True)
    
    contador_total = 0
    
    for classe in EXPECTED_CLASSES:
        combined_class_path = os.path.join(combined_path, classe)
        os.makedirs(combined_class_path, exist_ok=True)
        
        contador_classe = 0
        
        # Combinar de todos os datasets balanceados
        for dataset_name, dataset_output in [('jaffe', output_dirs['jaffe_cross_balanced']), 
                                           ('ck+', output_dirs['ck_cross_balanced'])]:
            
            dataset_class_path = os.path.join(dataset_output, classe)
            
            if os.path.exists(dataset_class_path):
                for img_file in os.listdir(dataset_class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        src = os.path.join(dataset_class_path, img_file)
                        dst = os.path.join(combined_class_path, f"{dataset_name}_{img_file}")
                        shutil.copy2(src, dst)
                        contador_classe += 1
                        contador_total += 1
        
        print(f"   {classe:>10}: {contador_classe} imagens")
    
    print(f"\n   📊 TOTAL COMBINADO: {contador_total} imagens")
    return contador_total

def main():
    """Função principal - executa todo o pipeline"""
    print("🎬 INICIANDO CROSS-DATASET DATA AUGMENTATION")
    print("=" * 80)
    
    # 1. Verificar distribuições atuais
    distribuicoes = verificar_distribuicao_atual()
    
    # 2. Calcular metas
    metas, deficit_por_dataset = calcular_metas_cross_dataset(distribuicoes)
    
    # 3. Executar augmentation para cada dataset
    resultados = {}
    
    for dataset_name, dataset_path in DATASET_PATHS.items():
        if dataset_name in deficit_por_dataset:
            output_key = f'{dataset_name}_cross_balanced'
            if output_key not in OUTPUT_DIRS and dataset_name == 'ck+':
                output_key = 'ck_cross_balanced'
            output_path = OUTPUT_DIRS[output_key]

            stats = executar_augmentation_dataset(
                dataset_name, 
                dataset_path, 
                output_path, 
                deficit_por_dataset[dataset_name]
            )

            resultados[dataset_name] = stats
    output_key = f'{dataset_name}_cross_balanced'
    if output_key not in OUTPUT_DIRS and dataset_name == 'ck+':
        output_key = 'ck_cross_balanced'
    output_path = OUTPUT_DIRS[output_key]

    
    # 4. Criar dataset combinado
    total_combinado = criar_dataset_combinado(OUTPUT_DIRS)
    
    # 5. Relatório final
    print(f"\n📈 RELATÓRIO FINAL")
    print("=" * 40)
    
    print(f"\n🎯 EXEMPLO ESPECÍFICO - CLASSE 'ANGER':")
    if 'anger' in metas:
        meta_anger = metas['anger']
        print(f"   Meta definida: {meta_anger} imagens por dataset")
        
        for dataset_name in distribuicoes:
            if dataset_name in deficit_por_dataset:
                original = deficit_por_dataset[dataset_name]['anger']['atual']
                geradas = deficit_por_dataset[dataset_name]['anger']['deficit'] 
                final = original + geradas
                print(f"   {dataset_name.upper()}: {original} + {geradas} = {final}")
        
        print(f"   TOTAL COMBINADO 'ANGER': {meta_anger * len(DATASET_PATHS)} imagens")
    
    print(f"\n📁 RESULTADOS SALVOS EM:")
    for output_name, output_path in OUTPUT_DIRS.items():
        if os.path.exists(output_path):
            total_imgs = sum(len([f for f in os.listdir(os.path.join(output_path, classe)) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                           for classe in EXPECTED_CLASSES 
                           if os.path.exists(os.path.join(output_path, classe)))
            print(f"   {output_path}: {total_imgs} imagens")
    
    print(f"\n✅ PIPELINE CONCLUÍDO COM SUCESSO!")

# ================== EXECUÇÃO ==================
if __name__ == "__main__":
    main()