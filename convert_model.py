
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflowjs as tfjs

model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# Load model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save in TensorFlow format
model.save_pretrained("./tf_model")
tokenizer.save_pretrained("./tf_tokenizer")

# Convert to TensorFlow.js format
tfjs.converters.convert_tf_saved_model(
    "./tf_model/saved_model/1",
    "./tfjs_model"
)
