# Presidential Speech Analysis Dashboard

A comprehensive, containerized web application for analyzing U.S. presidential speeches with advanced natural language processing, built with Taipy and optimized for production deployment.

## 🚀 Quick Start

### Docker Deployment (Recommended)

```bash
# Start the application
docker-compose up --build -d

# Access the dashboard
open http://localhost:5000
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download speech data (if needed)
python sotu_downloader_script.py

# Run the dashboard
python dashboard.py
```

## 📊 Features

### Interactive Analytics Dashboard
- **Multi-dimensional Filtering**: President, decade, speech type, political party, sentiment, readability, lexical diversity
- **Real-time Statistics**: Total speeches, date ranges, average word counts, president counts
- **Responsive Design**: Optimized for desktop and mobile viewing

### Natural Language Processing
- **Sentiment Analysis**: Polarity and subjectivity scoring using TextBlob
- **Readability Metrics**: 
  - Flesch Reading Ease
  - Flesch-Kincaid Grade Level
  - Gunning Fog Index
  - Coleman-Liau Index
- **Lexical Diversity**:
  - Type-Token Ratio (TTR)
  - Moving Average TTR (MATTR)
  - Measure of Textual Lexical Diversity (MTLD)

### Data Visualizations
- **Time Series Analysis**: Word count, sentiment, and readability trends over time
- **Distribution Charts**: Speech types, sentiment categories, readability levels
- **Interactive Tables**: Sortable, paginated speech data with key metrics

## 📁 Project Structure

```
sotu-downloader/
├── dashboard.py              # Main Taipy dashboard application
├── app.py                   # WSGI wrapper (legacy)
├── sotu_downloader_script.py # Data collection script
├── requirements.txt         # Python dependencies
├── gunicorn.conf.py        # Production server configuration
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-container orchestration
├── presidential_speeches/   # JSON data files (1,055+ speeches)
└── README.md               # This file
```

## 🛠 Technical Stack

### Backend
- **Python 3.12**: Core runtime environment
- **Taipy**: Web framework for data applications
- **DuckDB**: High-performance analytical database
- **Pandas**: Data manipulation and analysis
- **TextBlob**: Natural language processing
- **TextStat**: Readability analysis

### Frontend
- **Taipy GUI**: Interactive web components
- **Plotly**: Data visualization charts
- **Responsive CSS**: Mobile-friendly design

### Infrastructure
- **Docker**: Containerization
- **Production-optimized Taipy server**: Multi-threaded, production-ready configuration
- **Health checks**: Automated container monitoring
- **Resource optimization**: Memory and CPU efficient processing

## 📊 Dataset

### Coverage
- **1,055+ Presidential Speeches** (1789-1933)
- **33 Presidents** from George Washington to Franklin D. Roosevelt
- **Multiple Speech Types**: Inaugural addresses, State of the Union, special messages, proclamations

### Data Sources
- **Miller Center API**: University of Virginia's presidential speech archive
- **Comprehensive Metadata**: Date, location, context, speech type, political party

### Data Processing
- **Real-time NLP Analysis**: Sentiment, readability, and lexical diversity calculated on startup
- **Efficient Storage**: JSON format optimized for DuckDB queries
- **Data Validation**: Input sanitization and error handling

## 🔧 Configuration

### Environment Variables
```bash
PYTHONUNBUFFERED=1    # Real-time logging
TAIPY_PORT=5000       # Application port
```

### Production Settings
- **Multi-threading enabled** for concurrent user handling
- **Memory optimization** with efficient data processing
- **Graceful error handling** with fallback values
- **Health monitoring** with Docker health checks

### Performance Tuning
```python
# Key constants (adjustable in dashboard.py)
MAX_TEXT_LENGTH = 5000           # Text processing limit
MATTR_WINDOW_SIZE = 100          # Lexical diversity window
SENTIMENT_THRESHOLDS = ±0.1      # Sentiment classification
```

## 📈 Usage Examples

### Data Querying with DuckDB
```sql
-- Query all speeches
SELECT unnest(COLUMNS(*)) FROM 'presidential_speeches/[0-9]*.json';

-- Filter by president and year
SELECT * FROM 'presidential_speeches/*.json' 
WHERE name = 'Abraham Lincoln' AND year > 1860;
```

### Programmatic Access
```python
import duckdb

# Load and analyze data
conn = duckdb.connect(':memory:')
query = "SELECT unnest(COLUMNS(*)) FROM 'presidential_speeches/[0-9]*.json'"
df = conn.execute(query).df()

# Calculate custom metrics
df['word_density'] = df['word_count'] / df['character_count']
```

## 🚢 Deployment

### Container Management
```bash
# View logs
docker-compose logs -f

# Scale horizontally
docker-compose up --scale sotu-dashboard=3

# Stop services
docker-compose down

# Clean rebuild
docker-compose down && docker-compose up --build -d
```

### Production Considerations
- **Resource Limits**: Configure memory/CPU limits in docker-compose.yml
- **Load Balancing**: Use nginx for multiple container instances
- **Monitoring**: Implement logging and metrics collection
- **Security**: Network isolation and resource constraints

## 📚 API Reference

### Core Functions

#### Data Loading
```python
load_speech_data() -> pd.DataFrame
# Loads and processes all presidential speech data
```

#### Analysis Functions
```python
calculate_sentiment(text: str) -> Tuple[float, float]
# Returns (polarity, subjectivity)

calculate_readability(text: str) -> Tuple[float, float, float, float]
# Returns (flesch_ease, flesch_kincaid, gunning_fog, coleman_liau)

calculate_lexical_diversity(text: str) -> Tuple[float, float, float]
# Returns (ttr, mattr, mtld)
```

#### Filtering
```python
apply_all_filters(df, president, decade, speech_type, party, sentiment, readability, lexical_diversity) -> pd.DataFrame
# Applies comprehensive filtering to dataset
```

## 🧪 Testing

### Data Validation
```bash
# Verify data integrity
python -c "from dashboard import load_speech_data; print(f'Loaded {len(load_speech_data())} speeches')"

# Test NLP functions
python -c "from dashboard import calculate_sentiment; print(calculate_sentiment('This is a positive message'))"
```

### Container Testing
```bash
# Health check
curl http://localhost:5000/

# Performance test
docker stats presidential-speech-dashboard
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install dependencies: `pip install -r requirements.txt`
4. Make changes with comprehensive testing
5. Submit a pull request

### Code Standards
- **Type hints** for all function parameters
- **Docstrings** for public functions
- **Error handling** with graceful degradation
- **Security validation** for all inputs

## 📄 License

This project analyzes public domain presidential speeches. The analysis code and dashboard are available for educational and research purposes.

## 🔗 Resources

- [Taipy Documentation](https://docs.taipy.io/)
- [Miller Center Presidential Speeches](https://millercenter.org/the-presidency/presidential-speeches)
- [TextBlob Documentation](https://textblob.readthedocs.io/)
- [DuckDB Documentation](https://duckdb.org/docs/)

## 🐛 Support

For issues, feature requests, or questions:
1. Check existing documentation
2. Review container logs: `docker-compose logs`
3. Test with local development setup
4. Submit detailed issue reports with reproduction steps

---

**Built with ❤️ for presidential speech analysis and democratic transparency.**