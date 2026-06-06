from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
tfidf = TfidfVectorizer(max_features=3000,ngram_range=(1,2))
X = tfidf.fit_transform(df['text']).toarray()
y = df['class'].values
from sklearn.model_selection import StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('movie_review.csv')

df_hf = df.copy()
df_hf['labels'] = df_hf['class']
df_hf = df_hf.drop(columns=['class'])

encoder = LabelEncoder()
df_hf['labels'] = encoder.fit_transform(df_hf['labels'])

hf_dataset = Dataset.from_pandas(df_hf)

dataset = hf_dataset.train_test_split(test_size=0.2,seed=42)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    # The 'text' column is used for tokenization
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Training setup
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
)
# function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)
# Train
trainer.train()
# Evaluate
results = trainer.evaluate()
print(results)