import pandas as pd
import numpy as np
import random
import os
import gc
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# === 1. НАСТРОЙКИ ===
SEED = 993
N_FOLDS = 5
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
SUB_PATH = 'submission.csv'

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # Torch fix
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# Определяем устройство (GPU если есть, иначе CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# === 2. ЗАГРУЗКА ДАННЫХ ===
print("Loading data...")
try:
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
except FileNotFoundError:
    print("Error: train.csv or test.csv not found in current directory.")
    exit(1)

# Заполнение пропусков
text_cols = ['query', 'product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color']
for col in text_cols:
    if col in df_train.columns:
        df_train[col] = df_train[col].fillna("").astype(str)
        df_test[col] = df_test[col].fillna("").astype(str)

# === 3. FEATURE ENGINEERING ===

# 3.1 Текстовые длины и overlap
print("Generating basic features...")
def create_basic_features(df):
    df['query_len'] = df['query'].apply(len)
    df['title_len'] = df['product_title'].apply(len)
    df['desc_len'] = df['product_description'].apply(len)
    
    def word_overlap(row):
        q_words = set(row['query'].lower().split())
        t_words = set(row['product_title'].lower().split())
        return len(q_words.intersection(t_words))

    df['word_overlap'] = df.apply(word_overlap, axis=1)
    return df

df_train = create_basic_features(df_train)
df_test = create_basic_features(df_test)

# 3.2 BM25
print("Generating BM25 features...")
all_titles = pd.concat([df_train['product_title'], df_test['product_title']]).tolist()
tokenized_corpus = [doc.split(" ") for doc in all_titles]
bm25 = BM25Okapi(tokenized_corpus)
word_idf = bm25.idf

def fast_bm25_score(row):
    q_words = set(row['query'].lower().split())
    t_words = set(row['product_title'].lower().split())
    common = q_words.intersection(t_words)
    score = 0.0
    for w in common:
        try:
            score += word_idf.get(w, 0)
        except:
            pass
    return score

df_train['bm25_score'] = df_train.apply(fast_bm25_score, axis=1)
df_test['bm25_score'] = df_test.apply(fast_bm25_score, axis=1)

# 3.3 MPNet Embeddings (Bi-Encoder)
print("Generating MPNet Embeddings (Bi-Encoder)...")
st_model = SentenceTransformer('all-mpnet-base-v2', device=device)

# Формируем полный текст (Title + Description)
def make_full_text(row):
    return f"{row['product_title']} {row['product_description'][:200]}"

train_texts = df_train.apply(make_full_text, axis=1).tolist()
test_texts = df_test.apply(make_full_text, axis=1).tolist()

def get_cosine_sim(model, queries, docs, batch_size=32):
    q_emb = model.encode(queries, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    d_emb = model.encode(docs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    return torch.nn.functional.cosine_similarity(q_emb, d_emb).cpu().numpy()

df_train['bert_cosine'] = get_cosine_sim(st_model, df_train['query'].tolist(), train_texts)
df_test['bert_cosine'] = get_cosine_sim(st_model, df_test['query'].tolist(), test_texts)

del st_model, train_texts, test_texts
gc.collect()
torch.cuda.empty_cache()

# 3.4 Cross-Encoder (Самый мощный признак)
print("Generating Cross-Encoder Scores...")
cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

def prepare_pairs(df):
    pairs = []
    for idx, row in df.iterrows():
        # Берем Title + кусочек Description
        item_text = f"{row['product_title']} {str(row['product_description'])[:100]}"
        pairs.append([row['query'], item_text])
    return pairs

train_pairs = prepare_pairs(df_train)
test_pairs = prepare_pairs(df_test)

batch_size_ce = 64 if device == 'cuda' else 16
df_train['cross_score'] = cross_model.predict(train_pairs, batch_size=batch_size_ce, show_progress_bar=True)
df_test['cross_score'] = cross_model.predict(test_pairs, batch_size=batch_size_ce, show_progress_bar=True)

del cross_model, train_pairs, test_pairs
gc.collect()
torch.cuda.empty_cache()

# === 4. ОБУЧЕНИЕ МОДЕЛИ (CATBOOST CV) ===
print("Training CatBoost Ranker...")

feature_cols = ['query_len', 'title_len', 'desc_len', 'word_overlap', 'bert_cosine', 'bm25_score', 'product_brand', 'cross_score']
cat_features = ['product_brand']
target_col = 'relevance'
group_col = 'query_id'

# Подготовка брендов
df_train['product_brand'] = df_train['product_brand'].fillna("unknown").astype(str)
df_test['product_brand'] = df_test['product_brand'].fillna("unknown").astype(str)

# Сортировка для GroupKFold
df_train = df_train.sort_values(by=group_col).reset_index(drop=True)
df_test = df_test.sort_values(by=group_col).reset_index(drop=True)

final_predictions = np.zeros(len(df_test))
gkf = GroupKFold(n_splits=N_FOLDS)

test_pool = Pool(df_test[feature_cols], group_id=df_test[group_col], cat_features=cat_features)

for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train, groups=df_train[group_col])):
    print(f"--- Fold {fold+1}/{N_FOLDS} ---")
    
    train_pool = Pool(
        df_train.iloc[train_idx][feature_cols], 
        label=df_train.iloc[train_idx][target_col], 
        group_id=df_train.iloc[train_idx][group_col], 
        cat_features=cat_features
    )
    
    val_pool = Pool(
        df_train.iloc[val_idx][feature_cols], 
        label=df_train.iloc[val_idx][target_col], 
        group_id=df_train.iloc[val_idx][group_col], 
        cat_features=cat_features
    )
    
    # Параметры подобраны в ходе экспериментов
    model = CatBoostRanker(
        iterations=2000,
        learning_rate=0.08,
        depth=6,
        loss_function='YetiRank',
        eval_metric='NDCG:top=10',
        random_seed=SEED + fold,
        verbose=200,
        allow_writing_files=False # Чтобы не создавать лишние папки catboost_info
    )
    
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
    
    # Предсказание
    final_predictions += model.predict(test_pool)
    
    del model, train_pool, val_pool
    gc.collect()

# Усреднение
final_predictions /= N_FOLDS

# === 5. СОХРАНЕНИЕ РЕЗУЛЬТАТА ===
submission = df_test[['id']].copy()
submission['prediction'] = final_predictions
submission.to_csv(SUB_PATH, index=False)
print(f"Submission saved to {SUB_PATH}")