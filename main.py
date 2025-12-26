import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
import os
import warnings

# Отключаем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

# --- КОНФИГУРАЦИЯ ---
DATA_DIR = '/app/data'
OUTPUT_DIR = '/app/results'
N_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")
WIDTH_MULTIPLIER = 1.05

# Если запускаем локально без Docker, меняем пути:
if not os.path.exists(DATA_DIR):
    DATA_DIR = 'data'        # Локальная папка с данными
    OUTPUT_DIR = 'results'   # Локальная папка для результатов
    os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Running on: {DEVICE}")

# --- 1. КЛАССЫ НЕЙРОСЕТИ ---

class PricingDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class IntervalNet(nn.Module):
    def __init__(self, input_dim):
        super(IntervalNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out = self.net(x)
        mid = out[:, 0]
        # Ширина должна быть положительной -> Softplus
        width = torch.nn.functional.softplus(out[:, 1])
        return mid, width

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred_mid, pred_width, target_mid, target_width):
        pred_low = pred_mid - pred_width / 2
        pred_high = pred_mid + pred_width / 2
        
        target_low = target_mid - target_width / 2
        target_high = target_mid + target_width / 2

        # Intersection
        inter_low = torch.max(pred_low, target_low)
        inter_high = torch.min(pred_high, target_high)
        intersection = torch.clamp(inter_high - inter_low, min=0)

        # Union
        pred_area = pred_high - pred_low
        target_area = target_high - target_low
        union = pred_area + target_area - intersection + 1e-6

        iou = intersection / union
        return 1.0 - torch.mean(iou)

# --- 2. ПОДГОТОВКА ДАННЫХ ---

def preprocess_data(train, test):
    print("Start Preprocessing...")
    
    # 1. Даты
    for df in [train, test]:
        df['dt'] = pd.to_datetime(df['dt'])
        df['month_sin'] = np.sin(2 * np.pi * df['dt'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['dt'].dt.month / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['dt'].dt.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dt'].dt.dayofweek / 7)
    
    # Сортировка трейна
    train = train.sort_values('dt').reset_index(drop=True)
    
    # 2. Логарифмируем таргет для обучения
    train['log_p05'] = np.log1p(train['price_p05'])
    train['log_p95'] = np.log1p(train['price_p95'])
    
    # 3. Target Encoding (Global Stats)
    # Считаем средние по категориям только на трейне
    print("Generating Target Encodings...")
    for col in ['second_category_id', 'third_category_id']:
        mapper = train.groupby(col)[['log_p05', 'log_p95']].mean()
        mapper.columns = [f'{col}_mean_p05', f'{col}_mean_p95']
        
        train = train.merge(mapper, on=col, how='left')
        test = test.merge(mapper, on=col, how='left')
        
    # Статистики по самому товару
    product_mapper = train.groupby('product_id')[['log_p05', 'log_p95']].agg(['mean', 'std'])
    product_mapper.columns = ['product_mean_p05', 'product_std_p05', 'product_mean_p95', 'product_std_p95']
    
    train = train.merge(product_mapper, on='product_id', how='left')
    test = test.merge(product_mapper, on='product_id', how='left')
    
    # 4. Заполнение пропусков (для новых товаров в тесте)
    # Если нет статистики по товару, берем статистику по категории
    for df in [train, test]:
        df['product_mean_p05'] = df['product_mean_p05'].fillna(df['third_category_id_mean_p05'])
        df['product_mean_p05'] = df['product_mean_p05'].fillna(df['second_category_id_mean_p05'])
        
        df['product_mean_p95'] = df['product_mean_p95'].fillna(df['third_category_id_mean_p95'])
        df['product_mean_p95'] = df['product_mean_p95'].fillna(df['second_category_id_mean_p95'])
        
        # Std заполняем средним
        avg_std_p05 = train['log_p05'].std()
        avg_std_p95 = train['log_p95'].std()
        df['product_std_p05'] = df['product_std_p05'].fillna(avg_std_p05)
        df['product_std_p95'] = df['product_std_p95'].fillna(avg_std_p95)
        
        # Оставшиеся пропуски в категориях (если совсем редкие)
        cols_to_fill = [c for c in df.columns if 'mean' in c or 'std' in c]
        for c in cols_to_fill:
             df[c] = df[c].fillna(df[c].mean())

    # Погодные фичи - заполняем пропуски
    weather_cols = ['precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
    for df in [train, test]:
        df[weather_cols] = df[weather_cols].fillna(0)

    # 5. Отбор признаков
    num_features = [
        'n_stores', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'second_category_id_mean_p05', 'second_category_id_mean_p95',
        'third_category_id_mean_p05', 'third_category_id_mean_p95',
        'product_mean_p05', 'product_std_p05', 'product_mean_p95', 'product_std_p95'
    ]
    
    cat_features = ['activity_flag', 'holiday_flag'] # Простые флаги
    
    # 6. Scaling & OneHot
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])
    
    print("Fitting Scaler...")
    X_train = preprocessor.fit_transform(train)
    X_test = preprocessor.transform(test)
    
    # Подготовка таргетов (Midpoint, Width)
    y_p05 = train['log_p05'].values.astype(np.float32)
    y_p95 = train['log_p95'].values.astype(np.float32)
    
    y_mid = (y_p05 + y_p95) / 2
    y_width = (y_p95 - y_p05)
    y_target = np.stack([y_mid, y_width], axis=1)
    
    return X_train, y_target, X_test, test['row_id'].values

# --- 3. ОСНОВНАЯ ФУНКЦИЯ ---

def create_submission():
    print("Loading data...")
    try:
        train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    except FileNotFoundError:
        print("Error: Data files not found. Ensure 'data/' folder contains train.csv, test.csv")
        return

    # Препроцессинг
    X_train, y_train, X_test, test_ids = preprocess_data(train, test)
    
    print(f"Train shape: {X_train.shape}")
    
    # 5-Fold Training
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    test_preds_mid = np.zeros(len(X_test))
    test_preds_width = np.zeros(len(X_test))
    
    print(f"Starting {N_FOLDS}-Fold Training...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"--- Fold {fold+1}/{N_FOLDS} ---")
        
        # Данные фолда
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        
        train_ds = PricingDataset(X_tr, y_tr)
        val_ds = PricingDataset(X_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        
        # Инициализация модели
        model = IntervalNet(input_dim=X_train.shape[1]).to(DEVICE)
        criterion = SoftIoULoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Обучение
        best_loss = float('inf')
        best_weights = None
        
        for epoch in range(EPOCHS):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                mid, width = model(X_b)
                loss = criterion(mid, width, y_b[:, 0], y_b[:, 1])
                loss.backward()
                optimizer.step()
            
            # Валидация
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                    mid, width = model(X_b)
                    loss = criterion(mid, width, y_b[:, 0], y_b[:, 1])
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_weights = model.state_dict()
                
        # Предсказание на тесте лучшей моделью фолда
        model.load_state_dict(best_weights)
        model.eval()
        
        fold_mids = []
        fold_widths = []
        
        test_loader = DataLoader(PricingDataset(X_test), batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for X_b in test_loader:
                X_b = X_b.to(DEVICE)
                mid, width = model(X_b)
                fold_mids.append(mid.cpu().numpy())
                fold_widths.append(width.cpu().numpy())
                
        test_preds_mid += np.concatenate(fold_mids)
        test_preds_width += np.concatenate(fold_widths)
        
        print(f"Fold {fold+1} Best Val IoU: {1.0 - best_loss:.4f}")

    # Усреднение ансамбля
    avg_mid_log = test_preds_mid / N_FOLDS
    avg_width_log = test_preds_width / N_FOLDS
    
    # --- POST PROCESSING ---
    # Расширяем интервал на 5% (секрет успеха)
    avg_width_log = avg_width_log * WIDTH_MULTIPLIER
    
    # Восстанавливаем цены из логарифмов
    final_p05 = np.expm1(avg_mid_log - avg_width_log / 2)
    final_p95 = np.expm1(avg_mid_log + avg_width_log / 2)
    
    # Гарантия корректности
    final_p05 = np.maximum(0, final_p05)
    final_p95 = np.maximum(0, final_p95)
    
    # Сохранение
    sub = pd.DataFrame({
        'row_id': test_ids,
        'price_p05': final_p05,
        'price_p95': final_p95
    })
    
    # Важно сохранить порядок row_id как в sample_submission
    sub = sub.sort_values('row_id')
    
    output_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    sub.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(sub.head())

if __name__ == '__main__':
    create_submission()