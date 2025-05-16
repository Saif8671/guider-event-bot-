# EventMaster AI 🎪

EventMaster AI is an intelligent event assistant built with Streamlit and Google's Gemini AI. It helps event attendees navigate conferences, trade shows, and other events by providing real-time information, personalized recommendations, and interactive features.

## Features

### 🤖 AI Assistant
- Chat with an AI assistant about the event
- Get answers to common questions about schedules, locations, and event details
- Upload event agendas and get personalized recommendations

### 📅 Event Schedule
- View the complete event schedule in an organized timeline
- Filter events by day
- Highlight breaks and lunch times
- Get countdown timers to important sessions

### 📄 Document Analysis
- Upload and analyze event agendas (PDF format)
- Extract key information automatically
- Get insights about session topics and speakers

### 👥 Networking
- Upload your resume to find networking opportunities
- Get matched with attendees with similar interests
- Identify potential business connections

### 🗺️ Venue Navigation
- Find important locations within the venue
- Get directions to rooms, facilities, and services
- View interactive maps (when available)

## Installation

1. Clone this repository:
```bash
git clone 
cd EventMaster-AI
```

2. Create a virtual environment and activate it:
```bash
python -m venv myenv
# On Windows
myenv\Scripts\activate
# On macOS/Linux
source myenv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Upload an event agenda PDF to get started

4. Explore the different tabs to access all features

## Requirements

- Python 3.8+
- Streamlit
- Google Gemini API access
- Other dependencies listed in requirements.txt

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google Gemini AI](https://ai.google.dev/)
- Uses NLTK for natural language processing
- PDF processing with PyPDF2
