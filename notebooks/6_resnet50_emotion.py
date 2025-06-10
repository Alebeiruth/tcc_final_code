import os
import numpy as np
import pandas as pd
import cv2
import random
import time
import psutil
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter

# Configurações
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.3

# Paths dos datasets
FER2013_PATH = r".\data\FER2013"  # Ajustar para seu caminho
RAF_DB_PATH = r".\data\RAF-DB\DATASET"    # Ajustar para seu caminho

# Mapeamento das emoções
EMOTION_LABELS = {
    'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 
    'neutral': 4, 'sadness': 5, 'surprise': 6
}

class TrainingMonitor:
    """Classe para monitorar tempo e memória durante o treinamento"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory_mb = 0
        self.initial_memory_mb = 0
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Inicia o monitoramento"""
        self.start_time = time.time()
        self.initial_memory_mb = self._get_memory_usage()
        self.peak_memory_mb = self.initial_memory_mb
        print(f"🚀 Iniciando treinamento ResNet50...")
        print(f"⏰ Horário de início: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Memória inicial: {self.initial_memory_mb:.2f} MB")
        print("-" * 50)
        
    def update_peak_memory(self):
        """Atualiza o pico de memória se necessário"""
        current_memory = self._get_memory_usage()
        if current_memory > self.peak_memory_mb:
            self.peak_memory_mb = current_memory
            
    def _get_memory_usage(self):
        """Retorna uso atual de memória em MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def end_monitoring(self):
        """Finaliza o monitoramento e exibe estatísticas"""
        self.end_time = time.time()
        
        # Calcula tempo total
        total_time_seconds = self.end_time - self.start_time
        total_time_formatted = str(timedelta(seconds=int(total_time_seconds)))
        
        # Memória final
        final_memory_mb = self._get_memory_usage()
        memory_increase = final_memory_mb - self.initial_memory_mb
        
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE MONITORAMENTO DE TREINAMENTO")
        print("="*60)
        print(f"⏱️  Tempo total de treinamento: {total_time_formatted}")
        print(f"⏱️  Tempo em segundos: {total_time_seconds:.2f}s")
        print(f"💾 Memória inicial: {self.initial_memory_mb:.2f} MB")
        print(f"💾 Memória final: {final_memory_mb:.2f} MB")
        print(f"💾 Pico de memória: {self.peak_memory_mb:.2f} MB")
        print(f"📈 Aumento de memória: {memory_increase:.2f} MB")
        print(f"📈 Fator de aumento: {(final_memory_mb/self.initial_memory_mb):.2f}x")
        print("="*60)
        
        return {
            'total_time_seconds': total_time_seconds,
            'total_time_formatted': total_time_formatted,
            'initial_memory_mb': self.initial_memory_mb,
            'final_memory_mb': final_memory_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_increase_mb': memory_increase
        }

class MemoryCallback(tf.keras.callbacks.Callback):
    """Callback para monitorar memória durante o treinamento"""
    
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        
    def on_epoch_end(self, epoch, logs=None):
        self.monitor.update_peak_memory()
        if epoch % 5 == 0:  # Exibe a cada 5 épocas
            current_memory = self.monitor._get_memory_usage()
            print(f"Época {epoch+1} - Memória atual: {current_memory:.2f} MB")

def load_and_preprocess_image(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Carrega e preprocessa uma imagem"""
    try:
        # Carrega a imagem
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Converte para escala de cinza
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Redimensiona
        img = cv2.resize(img, target_size)
        
        # Converte para 3 canais (RGB) para ResNet50
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Normaliza
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"Erro ao processar imagem {image_path}: {e}")
        return None

def load_fer2013_data():
    """Carrega dados do FER2013"""
    images = []
    labels = []
    
    print(f"Verificando FER2013 em: {FER2013_PATH}")
    if not os.path.exists(FER2013_PATH):
        print(f"❌ Caminho não encontrado: {FER2013_PATH}")
        return np.array([]), []
    
    for emotion, label in EMOTION_LABELS.items():
        emotion_path = os.path.join(FER2013_PATH, emotion)
        if not os.path.exists(emotion_path):
            print(f"⚠️ Pasta {emotion} não encontrada")
            continue
            
        count = 0
        for filename in os.listdir(emotion_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(emotion_path, filename)
                
                # Determina se é treino ou teste baseado no nome
                is_test = 'test' in filename.lower()
                
                img = load_and_preprocess_image(image_path)
                if img is not None:
                    images.append(img)
                    labels.append((label, is_test))
                    count += 1
        
        print(f"✅ {emotion}: {count} imagens carregadas")
    
    return np.array(images), labels

def load_raf_db_data():
    """Carrega dados do RAF-DB"""
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    print(f"Verificando RAF-DB em: {RAF_DB_PATH}")
    if not os.path.exists(RAF_DB_PATH):
        print(f"❌ Caminho RAF-DB não encontrado: {RAF_DB_PATH}")
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    
    # Carrega dados de treino
    train_path = os.path.join(RAF_DB_PATH, 'train')
    if os.path.exists(train_path):
        train_total = 0
        for emotion, label in EMOTION_LABELS.items():
            emotion_path = os.path.join(train_path, emotion)
            if not os.path.exists(emotion_path):
                print(f"⚠️ Pasta {emotion} não encontrada em train")
                continue
                
            count = 0
            for filename in os.listdir(emotion_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(emotion_path, filename)
                    img = load_and_preprocess_image(image_path)
                    if img is not None:
                        train_images.append(img)
                        train_labels.append(label)
                        count += 1
                        train_total += 1
            
            print(f"✅ RAF-DB Train {emotion}: {count} imagens")
        print(f"✅ RAF-DB Train Total: {train_total} imagens carregadas")
    else:
        print(f"⚠️ Pasta de treino RAF-DB não encontrada: {train_path}")
    
    # Carrega dados de teste
    test_path = os.path.join(RAF_DB_PATH, 'test')
    if os.path.exists(test_path):
        test_total = 0
        for emotion, label in EMOTION_LABELS.items():
            emotion_path = os.path.join(test_path, emotion)
            if not os.path.exists(emotion_path):
                print(f"⚠️ Pasta {emotion} não encontrada em test")
                continue
                
            count = 0
            for filename in os.listdir(emotion_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(emotion_path, filename)
                    img = load_and_preprocess_image(image_path)
                    if img is not None:
                        test_images.append(img)
                        test_labels.append(label)
                        count += 1
                        test_total += 1
            
            print(f"✅ RAF-DB Test {emotion}: {count} imagens")
        print(f"✅ RAF-DB Test Total: {test_total} imagens carregadas")
    else:
        print(f"⚠️ Pasta de teste RAF-DB não encontrada: {test_path}")
    
    return (np.array(train_images), np.array(train_labels), 
            np.array(test_images), np.array(test_labels))

def balance_datasets(fer_data, raf_data):
    """Balanceia os datasets usando undersampling 50/50"""
    fer_images, fer_labels = fer_data
    raf_train_images, raf_train_labels, raf_test_images, raf_test_labels = raf_data
    
    # Separa FER2013 em treino e teste
    fer_train_images, fer_train_labels = [], []
    fer_test_images, fer_test_labels = [], []
    
    for img, (label, is_test) in zip(fer_images, fer_labels):
        if is_test:
            fer_test_images.append(img)
            fer_test_labels.append(label)
        else:
            fer_train_images.append(img)
            fer_train_labels.append(label)
    
    fer_train_images = np.array(fer_train_images)
    fer_train_labels = np.array(fer_train_labels)
    fer_test_images = np.array(fer_test_images)
    fer_test_labels = np.array(fer_test_labels)
    
    # Conta amostras por emoção em cada dataset
    fer_train_counts = Counter(fer_train_labels)
    raf_train_counts = Counter(raf_train_labels)
    
    print("Contagem original por emoção:")
    print("FER2013 Train:", fer_train_counts)
    print("RAF-DB Train:", raf_train_counts)
    
    # Verifica se RAF-DB tem dados
    if len(raf_train_images) == 0:
        print("⚠️ RAF-DB vazio, usando apenas FER2013")
        combined_train_images = fer_train_images
        combined_train_labels = fer_train_labels
        combined_test_images = fer_test_images
        combined_test_labels = fer_test_labels
    else:
        # Balanceamento 50/50
        balanced_fer_train_images, balanced_fer_train_labels = [], []
        balanced_raf_train_images, balanced_raf_train_labels = [], []
        
        for emotion in range(7):
            fer_count = fer_train_counts.get(emotion, 0)
            raf_count = raf_train_counts.get(emotion, 0)
            
            if fer_count > 0 and raf_count > 0:
                # Determina o número mínimo para balanceamento 50/50
                min_count = min(fer_count, raf_count)
                
                # Seleciona amostras aleatórias do FER2013
                fer_emotion_indices = np.where(fer_train_labels == emotion)[0]
                selected_fer_indices = np.random.choice(fer_emotion_indices, min_count, replace=False)
                
                for idx in selected_fer_indices:
                    balanced_fer_train_images.append(fer_train_images[idx])
                    balanced_fer_train_labels.append(fer_train_labels[idx])
                
                # Seleciona amostras aleatórias do RAF-DB
                raf_emotion_indices = np.where(raf_train_labels == emotion)[0]
                selected_raf_indices = np.random.choice(raf_emotion_indices, min_count, replace=False)
                
                for idx in selected_raf_indices:
                    balanced_raf_train_images.append(raf_train_images[idx])
                    balanced_raf_train_labels.append(raf_train_labels[idx])
        
        # Combina os datasets balanceados
        if len(balanced_fer_train_images) > 0 and len(balanced_raf_train_images) > 0:
            combined_train_images = np.concatenate([
                np.array(balanced_fer_train_images), 
                np.array(balanced_raf_train_images)
            ])
            combined_train_labels = np.concatenate([
                np.array(balanced_fer_train_labels), 
                np.array(balanced_raf_train_labels)
            ])
        else:
            print("⚠️ Balanceamento falhou, usando apenas FER2013")
            combined_train_images = fer_train_images
            combined_train_labels = fer_train_labels
        
        # Combina dados de teste (verifica se ambos têm dados)
        if len(raf_test_images) > 0:
            combined_test_images = np.concatenate([fer_test_images, raf_test_images])
            combined_test_labels = np.concatenate([fer_test_labels, raf_test_labels])
        else:
            combined_test_images = fer_test_images
            combined_test_labels = fer_test_labels
    
    print(f"\nDados finais:")
    print(f"Treino: {len(combined_train_images)} amostras")
    print(f"Teste: {len(combined_test_images)} amostras")
    print(f"Distribuição treino: {Counter(combined_train_labels)}")
    
    return (combined_train_images, combined_train_labels, 
            combined_test_images, combined_test_labels)

def create_resnet50_model():
    """Cria modelo ResNet50 para classificação de emoções"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Congela as camadas base
    base_model.trainable = False
    
    # Adiciona camadas customizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(7, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, monitor):
    """Treina o modelo"""
    print("Iniciando treinamento do ResNet50...")
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    memory_callback = MemoryCallback(monitor)
    
    # Treina o modelo
    training_start = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, memory_callback],
        verbose=1
    )
    training_time = time.time() - training_start
    
    print(f"⏱️ Treinamento concluído em: {timedelta(seconds=int(training_time))}")
    
    # Adiciona tempo de treinamento ao histórico
    history.training_time = training_time
    
    return history

def plot_training_history(history):
    """Plota histórico de treinamento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy - ResNet50')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss - ResNet50')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('resnet50_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo"""
    # Predições
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Relatório de classificação
    emotion_names = list(EMOTION_LABELS.keys())
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=emotion_names))
    
    # Matriz de confusão
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names,
                yticklabels=emotion_names)
    plt.title('Confusion Matrix - ResNet50')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('resnet50_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred, y_pred_classes, y_true_classes

def calculate_cross_dataset_drift(fer_features, raf_features):
    """Calcula métricas de drift entre datasets"""
    
    # Verifica se ambos os datasets têm dados
    if len(fer_features) == 0 or len(raf_features) == 0:
        print("⚠️ Não é possível calcular drift - um dos datasets está vazio")
        return None, None
    
    def flatten_features(features):
        return features.reshape(features.shape[0], -1)
    
    fer_flat = flatten_features(fer_features)
    raf_flat = flatten_features(raf_features)
    
    # Kolmogorov-Smirnov test
    ks_statistics = []
    for i in range(min(fer_flat.shape[1], 1000)):  # Limita para evitar muitos testes
        ks_stat, _ = stats.ks_2samp(fer_flat[:, i], raf_flat[:, i])
        ks_statistics.append(ks_stat)
    
    avg_ks = np.mean(ks_statistics)
    
    # Kullback-Leibler divergence (aproximada)
    def calculate_kl_divergence(X1, X2, bins=50):
        kl_divs = []
        for i in range(min(X1.shape[1], 100)):  # Limita features
            # Calcula histogramas
            range_min = min(X1[:, i].min(), X2[:, i].min())
            range_max = max(X1[:, i].max(), X2[:, i].max())
            
            hist1, _ = np.histogram(X1[:, i], bins=bins, range=(range_min, range_max))
            hist2, _ = np.histogram(X2[:, i], bins=bins, range=(range_min, range_max))
            
            # Normaliza
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # Adiciona pequeno epsilon para evitar log(0)
            epsilon = 1e-10
            hist1 += epsilon
            hist2 += epsilon
            
            # Calcula KL divergence
            kl = np.sum(hist1 * np.log(hist1 / hist2))
            kl_divs.append(kl)
        
        return np.mean(kl_divs)
    
    kl_divergence = calculate_kl_divergence(fer_flat, raf_flat)
    
    print(f"\nCross-Dataset Drift Analysis (ResNet50):")
    print(f"Average Kolmogorov-Smirnov statistic: {avg_ks:.4f}")
    print(f"Average Kullback-Leibler divergence: {kl_divergence:.4f}")
    
    return avg_ks, kl_divergence

def save_training_report(monitor_stats, history, model_accuracy):
    """Salva relatório completo do treinamento"""
    
    report = f"""
# RELATÓRIO DE TREINAMENTO - RESNET50 EMOTION CLASSIFIER
================================================================================

## INFORMAÇÕES GERAIS
- Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Modelo: ResNet50
- Tamanho da imagem: {IMG_SIZE}x{IMG_SIZE}
- Batch size: {BATCH_SIZE}
- Épocas máximas: {EPOCHS}

## PERFORMANCE DE TEMPO
- Tempo total: {monitor_stats['total_time_formatted']}
- Tempo em segundos: {monitor_stats['total_time_seconds']:.2f}s
- Tempo de treinamento: {timedelta(seconds=int(getattr(history, 'training_time', 0)))}

## PERFORMANCE DE MEMÓRIA
- Memória inicial: {monitor_stats['initial_memory_mb']:.2f} MB
- Memória final: {monitor_stats['final_memory_mb']:.2f} MB
- Pico de memória: {monitor_stats['peak_memory_mb']:.2f} MB
- Aumento de memória: {monitor_stats['memory_increase_mb']:.2f} MB
- Fator de multiplicação: {(monitor_stats['final_memory_mb']/monitor_stats['initial_memory_mb']):.2f}x

## RESULTADO FINAL
- Acurácia no teste: {model_accuracy:.4f} ({model_accuracy*100:.2f}%)
- Épocas executadas: {len(history.history['accuracy'])}

## OBSERVAÇÕES
- GPU utilizada: {tf.config.list_physical_devices('GPU')}
- Versão TensorFlow: {tf.__version__}
- Arquitetura: Transfer Learning com ResNet50

================================================================================
    """
    
    # Salva o relatório
    with open('resnet50_training_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📝 Relatório salvo em: resnet50_training_report.txt")

def main():
    """Função principal"""
    print("Carregando ResNet50 para classificação de emoções...")
    
    # Inicializa monitor
    monitor = TrainingMonitor()
    monitor.start_monitoring()
    
    try:
        # Carrega dados
        print("Carregando dados do FER2013...")
        fer_images, fer_labels = load_fer2013_data()
        monitor.update_peak_memory()
        
        if len(fer_images) == 0:
            print("❌ Nenhum dado carregado do FER2013. Verifique o caminho dos dados.")
            return
        
        print("Carregando dados do RAF-DB...")
        raf_data = load_raf_db_data()
        monitor.update_peak_memory()
        
        # Balanceia datasets
        print("Balanceando datasets...")
        X_train, y_train, X_test, y_test = balance_datasets((fer_images, fer_labels), raf_data)
        monitor.update_peak_memory()
        
        if len(X_train) == 0:
            print("❌ Nenhum dado de treino disponível")
            return
        
        # Divide dados de treino para validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=VALIDATION_SPLIT, 
            stratify=y_train, random_state=42
        )
        
        # Converte labels para categorical
        y_train = to_categorical(y_train, 7)
        y_val = to_categorical(y_val, 7)
        y_test = to_categorical(y_test, 7)
        
        print(f"Dados de treino: {X_train.shape}")
        print(f"Dados de validação: {X_val.shape}")
        print(f"Dados de teste: {X_test.shape}")
        
        # Cria e treina modelo
        print("Criando modelo ResNet50...")
        model = create_resnet50_model()
        model.summary()
        monitor.update_peak_memory()
        
        print("Treinando modelo...")
        history = train_model(model, X_train, y_train, X_val, y_val, monitor)
        
        # Salva modelo
        model.save('resnet50_emotion_model.h5')
        monitor.update_peak_memory()
        
        # Plota histórico
        plot_training_history(history)
        
        # Avalia modelo
        print("Avaliando modelo...")
        y_pred, y_pred_classes, y_true_classes = evaluate_model(model, X_test, y_test)
        model_accuracy = np.mean(y_pred_classes == y_true_classes)
        
        # Análise de drift (apenas se tivermos ambos os datasets)
        if len(raf_data[0]) > 0:  # Se RAF-DB tem dados
            print("Calculando cross-dataset drift...")
            
            # Extrai features para análise de drift
            feature_extractor = Model(inputs=model.input, 
                                    outputs=model.layers[-3].output)  # Antes da última camada
            
            # Separa dados por dataset para análise de drift
            sample_size = min(1000, len(X_test) // 2)
            fer_sample = X_test[:sample_size]
            raf_sample = X_test[sample_size:sample_size*2]
            
            fer_features = feature_extractor.predict(fer_sample)
            raf_features = feature_extractor.predict(raf_sample)
            
            calculate_cross_dataset_drift(fer_features, raf_features)
        else:
            print("⚠️ Análise de drift pulada - apenas FER2013 disponível")
        
        # Finaliza monitoramento
        monitor_stats = monitor.end_monitoring()
        
        # Salva relatório completo
        save_training_report(monitor_stats, history, model_accuracy)
        
        print("Treinamento ResNet50 concluído!")
        
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        monitor.end_monitoring()
        raise

if __name__ == "__main__":
    # Define seed para reprodutibilidade
    tf.random.set_seed(50)
    np.random.seed(50)
    random.seed(50)
    
    main()