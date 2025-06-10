# ========================================
# CARREGAMENTO E BALANCEAMENTO HÍBRIDO DOS DADOS
# ========================================

def carregar_imagens_por_sujeito(pasta, image_size=(96, 96)):
    """
    Carrega imagens de um diretório organizado por subpastas (uma para cada classe).
    
    Args:
        pasta: Caminho para o diretório raiz do dataset
        image_size: Tupla com o tamanho desejado das imagens (largura, altura)
    
    Returns:
        Lista de tuplas (imagem, classe, sujeito)
    """
    if not os.path.exists(pasta):
        raise FileNotFoundError(f"Diretório não encontrado: {pasta}")
    
    dados = []
    classes = sorted(os.listdir(pasta))
    
    for classe in classes:
        caminho_classe = os.path.join(pasta, classe)
        if not os.path.isdir(caminho_classe):
            continue
        
        for arquivo in tqdm(os.listdir(caminho_classe), desc=f"Carregando {classe}"):
            caminho_imagem = os.path.join(caminho_classe, arquivo)
            imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                continue
            
            imagem = cv2.resize(imagem, image_size)
            imagem = imagem / 255.0  # Normalizar para [0, 1]
            
            # Extrair identificador do sujeito baseado no dataset
            if 'jaffe' in pasta.lower():
                sujeito = arquivo[:2]  # Ex: "KA"
            elif 'ck' in pasta.lower():
                sujeito = arquivo.split('_')[0]  # Ex: "S010"
            else:
                sujeito = 'unknown'
            
            dados.append((imagem, classe, sujeito))
    
    return dados

def shannon_entropy(image):
    """Calcula entropia de Shannon para uma imagem."""
    histogram, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
    histogram = histogram[histogram > 0]  # Remove zeros para evitar log(0)
    return entropy(histogram, base=2)

def extrair_features_lbp_multiescala(imagem, radius_list, neighbors_list, method='uniform', grid_size=(3, 3)):
    """
    Extrai características LBP multi-escala com grid espacial.
    
    Args:
        imagem: Imagem de entrada (numpy array)
        radius_list: Lista de raios para LBP
        neighbors_list: Lista de vizinhos para LBP
        method: Método LBP ('uniform', 'default')
        grid_size: Tamanho da grade espacial (rows, cols)
    
    Returns:
        Vetor de características concatenado
    """
    if imagem.dtype != np.uint8:
        if imagem.max() <= 1.0:
            imagem = (imagem * 255).astype(np.uint8)
        else:
            imagem = imagem.astype(np.uint8)
    
    features = []
    h, w = imagem.shape
    grid_h, grid_w = grid_size
    
    # Dividir imagem em grid
    for i in range(grid_h):
        for j in range(grid_w):
            # Coordenadas da região
            start_h = i * h // grid_h
            end_h = (i + 1) * h // grid_h
            start_w = j * w // grid_w
            end_w = (j + 1) * w // grid_w
            
            regiao = imagem[start_h:end_h, start_w:end_w]
            
            # Extrair LBP para cada escala
            for radius, neighbors in zip(radius_list, neighbors_list):
                lbp = local_binary_pattern(regiao, neighbors, radius, method=method)
                
                # Calcular histograma
                if method == 'uniform':
                    n_bins = neighbors + 2
                else:
                    n_bins = 2 ** neighbors
                
                hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                features.extend(hist)
    
    return np.array(features)

def processar_dataset_lbp(dados, usar_multiescala=True, mostrar_progresso=True):
    """
    Processa dataset extraindo características LBP.
    
    Args:
        dados: Lista de tuplas (imagem, classe, sujeito)
        usar_multiescala: Se True, usa LBP multi-escala
        mostrar_progresso: Se True, mostra barra de progresso
    
    Returns:
        X: Array de características
        y: Array de classes
        subjects: Array de sujeitos
    """
    X = []
    y = []
    subjects = []
    
    iterador = tqdm(dados, desc="Extraindo LBP") if mostrar_progresso else dados
    
    for imagem, classe, sujeito in iterador:
        if usar_multiescala:
            features = extrair_features_lbp_multiescala(
                imagem,
                CONFIG_FINAL['lbp_radius'],
                CONFIG_FINAL['lbp_neighbors'],
                CONFIG_FINAL['lbp_method'],
                CONFIG_FINAL['grid_size']
            )
        else:
            # LBP simples
            if imagem.dtype != np.uint8:
                if imagem.max() <= 1.0:
                    imagem = (imagem * 255).astype(np.uint8)
                else:
                    imagem = imagem.astype(np.uint8)
            
            lbp = local_binary_pattern(imagem, 8, 1, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
            features = hist
        
        X.append(features)
        y.append(classe)
        subjects.append(sujeito)
    
    return np.array(X), np.array(y), np.array(subjects)

# ========================================
# CARREGAMENTO DOS DADOS ORIGINAIS
# ========================================

print("📂 CARREGAMENTO DOS DADOS ORIGINAIS")
print("=" * 50)

# Ajustar caminhos conforme sua estrutura
pasta_jaffe = '../../../datasets/jaffe'  # JAFFE original
pasta_ck = '../../../datasets/ck+'       # CK+ original

# Verificar se os diretórios existem
if not os.path.exists(pasta_jaffe):
    print(f"❌ JAFFE não encontrado: {pasta_jaffe}")
    # Tentar caminhos alternativos
    pasta_jaffe = '../data/jaffe'
    if not os.path.exists(pasta_jaffe):
        print(f"❌ JAFFE também não encontrado em: {pasta_jaffe}")

if not os.path.exists(pasta_ck):
    print(f"❌ CK+ não encontrado: {pasta_ck}")
    # Tentar caminhos alternativos
    pasta_ck = '../data/ck+'
    if not os.path.exists(pasta_ck):
        print(f"❌ CK+ também não encontrado em: {pasta_ck}")

# Carregar dados se os diretórios existem
if os.path.exists(pasta_jaffe) and os.path.exists(pasta_ck):
    print(f"✅ Carregando JAFFE de: {pasta_jaffe}")
    dados_jaffe_orig = carregar_imagens_por_sujeito(pasta_jaffe, CONFIG_FINAL['image_size'])
    
    print(f"✅ Carregando CK+ de: {pasta_ck}")
    dados_ck_orig = carregar_imagens_por_sujeito(pasta_ck, CONFIG_FINAL['image_size'])
    
    print(f"\n📊 ESTATÍSTICAS INICIAIS:")
    print(f"   • JAFFE original: {len(dados_jaffe_orig)} amostras")
    print(f"   • CK+ original: {len(dados_ck_orig)} amostras")
    
    # Mostrar distribuição de classes
    jaffe_classes = Counter([cls for _, cls, _ in dados_jaffe_orig])
    ck_classes = Counter([cls for _, cls, _ in dados_ck_orig])
    
    print(f"\n📈 DISTRIBUIÇÃO JAFFE:")
    for classe, count in sorted(jaffe_classes.items()):
        print(f"   • {classe}: {count} amostras")
    
    print(f"\n📈 DISTRIBUIÇÃO CK+:")
    for classe, count in sorted(ck_classes.items()):
        print(f"   • {classe}: {count} amostras")

else:
    print("❌ Não foi possível encontrar os diretórios dos datasets!")
    print("🔧 Ajuste as variáveis pasta_jaffe e pasta_ck com os caminhos corretos")

# ========================================
# EXECUÇÃO DO BALANCEAMENTO HÍBRIDO
# ========================================

if 'dados_jaffe_orig' in locals() and 'dados_ck_orig' in locals():
    print(f"\n⚖️ EXECUTANDO BALANCEAMENTO HÍBRIDO CROSS-DATASET")
    print("=" * 60)
    
    # Executar balanceamento híbrido com resolução otimizada
    dados_jaffe_balanced, dados_ck_balanced, stats_balanceamento = executar_balanceamento_hibrido_com_resolucao_otimizada(
        dados_jaffe_orig, 
        dados_ck_orig,
        classes_alvo=CONFIG_FINAL['classes_alvo'],
        target_size=CONFIG_FINAL['image_size'],
        usar_preprocessamento=True,
        random_state=CONFIG_FINAL['random_state']
    )
    
    print(f"\n✅ BALANCEAMENTO CONCLUÍDO!")
    print(f"   • JAFFE balanceado: {len(dados_jaffe_balanced)} amostras")
    print(f"   • CK+ processado: {len(dados_ck_balanced)} amostras")
    
    # Verificar distribuição final
    jaffe_balanced_classes = Counter([cls for _, cls, _ in dados_jaffe_balanced])
    ck_balanced_classes = Counter([cls for _, cls, _ in dados_ck_balanced])
    
    print(f"\n📊 DISTRIBUIÇÃO FINAL:")
    print("Classe       JAFFE    CK+      Diferença")
    print("-" * 45)
    
    todas_classes = sorted(set(jaffe_balanced_classes.keys()) | set(ck_balanced_classes.keys()))
    for classe in todas_classes:
        j_count = jaffe_balanced_classes.get(classe, 0)
        c_count = ck_balanced_classes.get(classe, 0)
        diff = abs(j_count - c_count)
        print(f"{classe:12} {j_count:6d}   {c_count:6d}   {diff:9d}")
    
    print("-" * 45)
    print(f"{'TOTAL':12} {len(dados_jaffe_balanced):6d}   {len(dados_ck_balanced):6d}")
    
    # Calcular ratio de balanceamento
    total_amostras = len(dados_jaffe_balanced) + len(dados_ck_balanced)
    if total_amostras > 0:
        ratio_jaffe = (len(dados_jaffe_balanced) / total_amostras) * 100
        ratio_ck = (len(dados_ck_balanced) / total_amostras) * 100
        desvio_50_50 = abs(50 - ratio_jaffe)
        
        print(f"\n🎯 EFICIÊNCIA DO BALANCEAMENTO:")
        print(f"   • Ratio JAFFE: {ratio_jaffe:.1f}%")
        print(f"   • Ratio CK+: {ratio_ck:.1f}%")
        print(f"   • Desvio do 50/50: {desvio_50_50:.1f} pontos percentuais")
        
        if desvio_50_50 < 5:
            print("   🎉 EXCELENTE: Balanceamento muito bom!")
        elif desvio_50_50 < 10:
            print("   ✅ BOM: Balanceamento adequado.")
        else:
            print("   ⚠️ ACEITÁVEL: Pequeno desvio do ideal.")

else:
    print("❌ Dados não carregados - execute o carregamento primeiro!")

print(f"\n🚀 Próximo passo: Extrair características LBP dos dados balanceados")