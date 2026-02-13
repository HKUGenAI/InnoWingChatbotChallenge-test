# InnoWing Chatbot Challenge
This is a testing repository.

## 📦 Requirements

### **General**

*   Git
*   A code editor (VS Code recommended)

### **Python Setup**

*   Python **3.9+**
*   `pip` for dependency installation

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/HKUGenAI/InnoWingChatbotChallenge-test.git
cd InnoWingChatbotChallenge-test
```

***

# 🐍 Python Setup

### 1. Create a virtual environment

```bash
# macOS & Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set environment variables

Copy the example template:

```bash
cp .env.example .env
```

Then edit `.env` and fill in your keys, e.g.:

    OPENAI_API_KEY=your_key_here
    MODEL=gpt-4o-mini

### 4. Run the chatbot

```bash
python main.py
```

If the project uses a web server:

```bash
python app.py
```

Open the printed local URL in your browser (commonly <http://localhost:8000>).

***
