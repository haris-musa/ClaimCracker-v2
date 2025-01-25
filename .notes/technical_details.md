# ClaimCracker v2 - Technical Details

## Dataset Analysis Results

### Dataset Structure

- **Main Dataset**: `Dataset.csv` (17,424 samples)

  - Contains all articles with proper unique IDs
  - Clean, combined dataset ready for ML pipeline
  - This is the file we use in our implementation

- **Split Files** (Not Used):
  - `Dataset - Real.csv` (9,878 samples)
  - `Dataset - Fake.csv` (7,546 samples)
  - Note: These files reuse ID numbers (1 to N in each file)
  - Kept for reference but not used in pipeline

### Basic Statistics

- Total samples: 17,424
- Real news: 9,878 (56.7%)
- Fake news: 7,546 (43.3%)
- Class distribution is reasonably balanced

### Text Characteristics

1. **Title Length**

   - Mean: 65 characters
   - Median: 60 characters
   - 90% of titles between 50-71 characters
   - Some outliers up to 1,791 characters

2. **Article Length**
   - Mean: 208 words
   - Median: 201 words
   - Most articles between 147-256 words
   - Maximum length: 3,115 words
   - 32 articles missing text content

### Features Available

- ID: Unique identifier
- News_Title: Article headline
- News_Text: Main article content
- Published_Date: Publication date
- Source: News source
- Source_URL: Original URL
- Author: Article author
- Country: Publication country
- Language: Article language
- News_Type: Category/type
- Label: Real/Fake classification

## ML Pipeline Structure

### 1. Data Management (`src/ml/data/dataset.py`)

- **NewsDataset Class**
  - Lazy loading of data to optimize memory usage
  - Caching mechanism for dataset and statistics
  - Stratified train/val/test splitting
  - Built-in dataset statistics calculation

### 2. Text Preprocessing (`src/ml/data/preprocessor.py`)

- **TextPreprocessor Class**
  - scikit-learn compatible transformer
  - URL removal with regex pattern matching
  - Number handling (optional removal)
  - Unicode normalization
  - Minimum word length filtering
  - Case normalization

### 3. Configuration (`src/ml/config/config.py`)

- **MLConfig Class**
  - Type-safe configuration using dataclass
  - Path management for data and models
  - Default values for preprocessing
  - Training hyperparameters
  - Automatic directory creation

## Implementation Details

### Data Processing Pipeline

1. **Data Loading**

   ```python
   dataset = NewsDataset(data_dir="Dataset")
   df = dataset.load_data()
   ```

2. **Text Preprocessing**

   ```python
   preprocessor = TextPreprocessor(
       remove_urls=True,
       lowercase=True,
       min_word_length=2
   )
   ```

3. **Data Splitting**
   ```python
   train_df, val_df, test_df = dataset.prepare_splits(
       test_size=0.2,
       val_size=0.1
   )
   ```

## Design Decisions

1. **Lazy Loading**

   - Why: Memory efficiency for large datasets
   - Implementation: Properties and caching
   - Benefits: Faster initialization, lower memory footprint

2. **Type Hints**

   - Why: Code clarity and IDE support
   - Implementation: Full type annotation
   - Benefits: Better maintainability and error catching

3. **Scikit-learn Compatibility**
   - Why: Integration with ML ecosystem
   - Implementation: BaseEstimator and TransformerMixin
   - Benefits: Pipeline compatibility and standardization

## Performance Considerations

1. **Memory Management**

   - Lazy loading of dataset
   - Caching of processed data
   - Efficient string operations

2. **Processing Efficiency**
   - Compiled regex patterns
   - Vectorized operations where possible
   - Minimal data copying

## Future Improvements

1. **Planned Enhancements**

   - Add parallel processing support
   - Implement data augmentation
   - Add more text cleaning options

2. **Optimization Opportunities**
   - Batch processing for large files
   - Memory-mapped file reading
   - GPU acceleration for preprocessing

## Model Architecture

### Base Model

- DistilBERT (distilbert-base-uncased)
  - Lightweight but powerful transformer
  - 6 layers, 768 hidden size
  - 40% smaller than BERT-base
  - 97% of BERT's performance

### Classification Head

- CLS token pooling
- Dropout (p=0.1) for regularization
- Linear layer (768 -> 2)
- Cross-entropy loss

### Key Features

- Automatic mixed precision training
- Efficient tokenization
- Save/load functionality
- Type hints and documentation

## Training Pipeline

### Data Management

- Custom PyTorch Dataset
- Dynamic batching
- Efficient memory usage
- Automatic padding/truncation

### Training Loop

- AdamW optimizer
- Cosine annealing scheduler
- Gradient clipping
- Early stopping (F1-based)

### Evaluation Metrics

- Loss (cross-entropy)
- Accuracy
- F1 score (weighted)
- Precision
- Recall

### Configuration

- Fully configurable via dataclass
- JSON serialization
- Automatic directory creation
- Device management

### Performance Optimization

- Batch size: 16 (free-tier compatible)
- Learning rate: 2e-5
- Weight decay: 0.01
- Warmup steps: 500
