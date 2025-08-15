Invisage: AI-Powered Visual Document Analysis
Invisage is a sophisticated Retrieval-Augmented Generation (RAG) application that intelligently extracts and analyzes information from complex visual documents, including PDFs and images containing text, tables, and charts.

✨ Features
Multi-Format Processing: Seamlessly handles PDFs, scanned documents, and various image formats (.png, .jpg, .webp, etc.).

Advanced OCR & Vision: Utilizes Tesseract and the GPT-4o vision model to accurately extract text and analyze visual elements like charts, graphs, and photos.

Structured Data Extraction: Intelligently identifies and parses tables and chart data into structured formats for clear presentation and analysis.

Interactive RAG Q&A: Allows users to ask questions in natural language and receive context-aware answers based on the document's content.

Cost-Saving Safeguards: Includes a built-in check to prevent the processing of overly long documents, helping to manage API costs effectively.

🛠️ Tech Stack
Backend: Python

Frontend: Streamlit

Core AI/RAG Framework: LangChain

LLM & Vision Model: OpenAI (GPT-4o-mini)

Vector Database: ChromaDB

PDF & Image Processing: PyMuPDF, Pillow

OCR Engine: Tesseract

Table Extraction: Camelot

🚀 Setup and Installation
Follow these steps to run the application locally.

1. Prerequisites
Python 3.9+

Tesseract OCR Engine installed and accessible in your system's PATH.

2. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


3. Create a Virtual Environment
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate


4. Install Dependencies
Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt


5. Set Up Environment Variables
Create a file named .env in the root of your project directory and add your OpenAI API key:

OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


6. Run the Streamlit App
streamlit run app.py


The application should now be running in your browser!

📖 How to Use
Upload a Document: Use the sidebar to upload a PDF or an image file.

Wait for Analysis: The app will process the document, performing OCR, vision analysis, and embedding.

View Extracted Content: Once processed, you can view the extracted text and structured data (like tables) page by page.

Ask a Question: Use the Q&A section to ask questions about the content of the document. The AI will retrieve relevant information and generate a concise answer.

👤 About the Creator
This project was created by Ayush Ranjan as a demonstration of advanced RAG capabilities on visual documents.

LinkedIn: Ayush Ranjan

GitHub: itsayushranjan

While the core architecture is original, the development process was accelerated by brainstorming and debugging with AI assistants like Gemini, ChatGPT, and others.

⚠️ Disclaimer
This is a public demo that utilizes a personal OpenAI API key. To ensure the service remains available and to manage costs, please adhere to the following:

Use Low-Resolution Files: For best results and to prevent high token usage, please use images with dimensions under 500x500 pixels.

Fair Use Policy: The API key is actively monitored. In the event of excessive or suspected malicious usage, the key will be revoked immediately.