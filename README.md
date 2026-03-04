<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,100:f59e0b&height=200&section=header&text=EventMaster%20AI&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Your%20Intelligent%20Event%20Companion&descSize=18&descAlignY=58" width="100%"/>

[![Live Demo](https://img.shields.io/badge/Live_Demo-Visit_App-f59e0b?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Gemini_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> Navigate conferences, trade shows, and events with AI —  
> real-time schedules, smart networking matches, venue directions, and document analysis in one place.


</div>

---

## ✨ Features

### 🤖 AI Assistant
- Chat naturally about the event — schedules, locations, speakers, logistics
- Upload event agendas and get instant personalized recommendations
- Answers drawn directly from your uploaded documents

### 📅 Event Schedule
- Full timeline view with day-by-day filtering
- Highlighted breaks, meals, and key sessions
- Live countdown timers to upcoming sessions

### 📄 Document Analysis
- Upload event agendas in PDF format
- Auto-extract session topics, speaker names, and timings
- Get a quick digest of the most relevant sessions for you

### 👥 Smart Networking
- Upload your resume to find relevant networking opportunities
- Get matched with attendees sharing similar interests
- Identify potential business connections before you walk in the door

### 🗺️ Venue Navigation
- Find rooms, facilities, and services inside the venue
- Step-by-step directions to any location
- Interactive map support when available

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_AI-4285F4?style=flat&logo=google&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-f59e0b?style=flat&logo=python&logoColor=white)
![PyPDF2](https://img.shields.io/badge/PyPDF2-PDF_Processing-333?style=flat&logo=python&logoColor=white)

---

## 📂 Project Structure
```
EventMaster-AI/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .env                    # API keys (never commit this)
├── .env.example            # Template for environment setup
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python **3.8+**
- A [Google Gemini API key](https://ai.google.dev)

### 1. Clone the Repository
```bash
git clone https://github.com/Saif8671/EventMaster-AI.git
cd EventMaster-AI
```

### 2. Set Up Virtual Environment
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> ⚠️ Never commit your `.env` file. Make sure it's listed in `.gitignore`.

### 5. Run the App
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 Usage

1. **Upload an agenda PDF** to get started — the AI will parse it automatically
2. Use the **AI Assistant** tab to ask anything about the event
3. Browse the **Schedule** tab for a filtered timeline view
4. Go to **Networking** and upload your resume to find relevant connections
5. Use **Venue Navigation** to find your way around the event space

---

## 🗺️ Roadmap

- [ ] Multi-event support (switch between conferences)
- [ ] Calendar export (`.ics`) for session bookmarks
- [ ] Speaker profile cards with LinkedIn integration
- [ ] Real-time push notifications for session reminders
- [ ] QR code scanner for badge-based attendee matching
- [ ] Mobile-responsive PWA interface

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Commit: `git commit -m "feat: describe your change"`
4. Push and open a PR

---

## 📄 License

MIT © [Saif ur Rahman](https://github.com/Saif8671)

---

<div align="center">

Never miss a session. Never miss a connection. 🎪

[![GitHub](https://img.shields.io/badge/GitHub-Saif8671-100000?style=flat&logo=github)](https://github.com/Saif8671)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=flat&logo=netlify)](https://saif-portfolio8671.netlify.app)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:f59e0b,100:0d1117&height=100&section=footer" width="100%"/>

</div>
