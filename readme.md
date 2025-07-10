# 🏥 AI SOAP Note Generator for Google Colab

> Transform unstructured medical notes into professional SOAP documentation using Google's Gemma 3N model - **Optimized for Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/your-notebook-link)

## 📋 Overview

The AI SOAP Note Generator is an intelligent medical documentation tool that converts informal doctor's notes, patient encounters, and clinical observations into structured SOAP (Subjective, Objective, Assessment, Plan) format. This tool leverages Google's Gemma 3N language model and runs seamlessly in Google Colab with GPU acceleration.

## ✨ Features

- **🚀 Google Colab Ready**: No local setup required - runs entirely in the cloud
- **⚡ GPU Acceleration**: Leverages Colab's free GPU/TPU for fast processing
- **🧠 Gemma 3N Integration**: Uses Google's latest medical-aware language model
- **📱 Multiple Interfaces**: 
  - Interactive Jupyter widgets
  - Modern Gradio web interface
  - Direct function calls
- **📁 File Support**: Upload .txt files directly in Colab
- **🎯 Pre-loaded Examples**: Built-in medical scenarios for immediate testing
- **🔗 Shareable Links**: Generate public links to share your interface

## 🚀 Quick Start in Google Colab

### 1. Open the Notebook
Click the "Open in Colab" badge above or create a new notebook in [Google Colab](https://colab.research.google.com/)

### 2. Set Runtime to GPU (Recommended)
```
Runtime → Change runtime type → Hardware accelerator → GPU
```

### 3. Install Dependencies
Run this cell first:
```python
# Install required packages
!pip install -q gradio torch transformers accelerate bitsandbytes
!pip install -q ipywidgets

# Import libraries
import gradio as gr
import torch
from transformers import pipeline
import ipywidgets as widgets
from IPython.display import display, HTML
```

### 4. Run All Cells
Execute the notebook cells in order to:
- Load the Gemma 3N model
- Set up the interface
- Start generating SOAP notes

### 5. Use the Interface
- **Gradio Interface**: Click the public URL to access the web interface
- **Colab Widgets**: Use the interactive widgets directly in the notebook

## 📱 Interface Options

### Option 1: Gradio Web Interface (Recommended)
```python
# Launches a web interface with public sharing
gradio_interface.launch(share=True)
```
**Benefits:**
- Modern, responsive design
- Public shareable links
- Mobile-friendly
- Copy-to-clipboard functionality

### Option 2: Jupyter Widgets
```python
# Interactive widgets within the notebook
display(main_interface)
```
**Benefits:**
- Runs directly in Colab
- No external links needed
- Integrated with notebook workflow

## 🔧 Colab-Specific Setup

### GPU Configuration
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Configure for Colab GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model_config = {
    "device_map": "auto",
    "torch_dtype": torch.float16 if device == "cuda" else torch.float32
}
```

### File Upload in Colab
```python
# Method 1: Direct file upload
from google.colab import files
uploaded = files.upload()

# Method 2: Google Drive integration
from google.colab import drive
drive.mount('/content/drive')
```

### Save Results to Drive
```python
# Save generated SOAP notes to Google Drive
def save_to_drive(soap_note, filename):
    with open(f'/content/drive/MyDrive/{filename}', 'w') as f:
        f.write(soap_note)
    print(f"✅ Saved to Google Drive: {filename}")
```

## 📝 Usage Examples

### Example 1: Quick Test
```python
# Test with example data
test_note = """
Patient: 45yo male with chest pain x2 hours
Sharp substernal pain 7/10, radiates to left arm
SOB, diaphoresis, no nausea
Vitals: BP 150/90, HR 110, O2 96%
Anxious, diaphoretic appearance
"""

soap_result = generate_soap_note(test_note)
print(soap_result)
```

### Example 2: File Processing
```python
# Upload and process medical files
from google.colab import files
uploaded_files = files.upload()

for filename, content in uploaded_files.items():
    medical_text = content.decode('utf-8')
    soap_note = generate_soap_note(medical_text)
    
    # Save result
    output_filename = f"SOAP_{filename}"
    with open(output_filename, 'w') as f:
        f.write(soap_note)
    
    # Download result
    files.download(output_filename)
```

## 🎯 Pre-loaded Medical Examples

The notebook includes three clinical scenarios:

1. **Chest Pain Case**: Acute coronary syndrome workup
2. **Diabetes Case**: New onset diabetes mellitus
3. **Pediatric Case**: Streptococcal pharyngitis

Click any example button to load and test immediately.

## 🔍 Model Information

### Gemma 3N Configuration
```python
model_name = "google/gemma-3n-7b"  # Adjust based on availability
tokenizer_config = {
    "max_length": 2048,
    "temperature": 0.7,
    "do_sample": True
}
```

### Memory Optimization for Colab
```python
# For Colab's memory constraints
torch.cuda.empty_cache()
model = model.half()  # Use 16-bit precision
```

## ⚠️ Colab-Specific Considerations

### Runtime Limitations
- **12-hour session limit**: Save work frequently
- **GPU quota**: Free tier has daily limits
- **Memory constraints**: ~12-15GB RAM available

### Best Practices
1. **Save frequently**: Download important results
2. **Use GPU wisely**: Enable only when needed
3. **Monitor resources**: Check RAM/GPU usage
4. **Backup notebooks**: Save to Drive regularly

## 🛠️ Troubleshooting in Colab

### Common Issues

**"Runtime disconnected"**
```python
# Prevent disconnection
import time
while True:
    time.sleep(60)  # Keep session alive
```

**"Out of GPU memory"**
```python
# Clear GPU memory
torch.cuda.empty_cache()
# Restart runtime if needed: Runtime → Restart runtime
```

**"Package not found"**
```python
# Reinstall packages
!pip install --upgrade gradio transformers torch
```

**Gradio interface not loading**
```python
# Try without share link
gradio_interface.launch(share=False, debug=True)
```

## 📊 Performance Tips

### Optimize for Colab
```python
# Batch processing for multiple notes
def batch_process_notes(note_list):
    results = []
    for i, note in enumerate(note_list):
        print(f"Processing {i+1}/{len(note_list)}")
        soap_note = generate_soap_note(note)
        results.append(soap_note)
    return results
```

### Monitor Resources
```python
# Check memory usage
!nvidia-smi
!cat /proc/meminfo | grep MemAvailable
```

## 🔗 Sharing Your Work

### Share Notebook
1. **File → Save a copy in Drive**
2. **Share → Get shareable link**
3. Set permissions to "Anyone with the link"

### Share Interface
```python
# Gradio creates public URLs automatically
gradio_interface.launch(share=True)
# Copy the public URL to share with others
```

## 📋 Colab Notebook Structure

```
📓 SOAP_Note_Generator.ipynb
├── 🔧 Setup & Installation
├── 🧠 Model Loading
├── 📝 SOAP Generation Function
├── 🎨 Interface Creation
│   ├── Gradio Web Interface
│   └── Jupyter Widgets
├── 📋 Example Cases
├── 🚀 Launch Interface
└── 💾 Save/Export Functions
```

## 📄 Medical Disclaimer

> **⚕️ IMPORTANT**: This tool is for **educational and research purposes only**
> - Not intended for actual clinical use
> - Always consult qualified healthcare professionals
> - Remove patient identifiers before processing
> - Comply with HIPAA and privacy regulations

## 🆘 Getting Help

### In Colab:
1. Use `!pip list` to check installed packages
2. Check GPU with `!nvidia-smi`
3. Restart runtime if needed: `Runtime → Restart runtime`

### Common Commands:
```python
# Debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Check versions
print(f"Torch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Gradio: {gr.__version__}")
```

## 🚀 Advanced Features

### Google Drive Integration
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save notebooks and results automatically
import shutil
shutil.copy('generated_soap_notes.txt', '/content/drive/MyDrive/')
```

### Scheduled Processing
```python
# Process notes at scheduled intervals
import schedule
import time

def scheduled_processing():
    # Your processing logic here
    pass

schedule.every(30).minutes.do(scheduled_processing)
```

---

**🔬 Ready to start generating professional SOAP notes in Google Colab!**

Click "Open in Colab" above and run all cells to get started immediately.