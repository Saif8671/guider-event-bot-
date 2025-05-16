import streamlit as st
from dotenv import load_dotenv
import os
import requests
import json
import time
import datetime
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import base64
import random

# Load environment variables
load_dotenv()

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set up the page
st.set_page_config(page_title="EventMaster AI", page_icon="🎪", layout="wide")

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}
.chat-message.user {
    background-color: #E3F2FD;
    border-left: 5px solid #2196F3;
}
.chat-message.bot {
    background-color: #F3E5F5;
    border-left: 5px solid #9C27B0;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
}
.timeline-item {
    padding: 10px;
    margin-bottom: 10px;
    border-left: 4px solid #7E57C2;
    background-color: #EDE7F6;
    border-radius: 5px;
}
.timeline-break {
    padding: 10px;
    margin-bottom: 10px;
    border-left: 4px solid #9C27B0;
    background-color: #F3E5F5;
    border-radius: 5px;
}
.skill-tag {
    display: inline-block;
    background-color: #7E57C2;
    color: white;
    padding: 0.3rem 0.6rem;
    border-radius: 20px;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    font-weight: 500;
}
.clock-display {
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    color: #4527A0;
    margin: 10px 0;
}
.countdown-display {
    font-size: 1.5rem;
    text-align: center;
    color: #5E35B1;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Function to query Gemini API
def query_gemini_api(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        response_data = response.json()
        
        if "candidates" in response_data and response_data["candidates"]:
            parts = response_data["candidates"][0]["content"]["parts"]
            return "\n".join(part["text"] for part in parts if "text" in part)
        elif "error" in response_data:
            return f"Error: {response_data['error']['message']}"
        else:
            return "Sorry, I couldn't process your question. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid response from API"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Function to parse agenda from text
def parse_agenda(text):
    lines = text.split('\n')
    event_name = ""
    event_date = ""
    event_location = ""
    agenda_items = []
    current_day = ""
    
    # First pass to extract event metadata
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        if i == 0 and not event_name:  # First non-empty line is likely the event name
            event_name = line
        elif "202" in line and not event_date and any(month in line.lower() for month in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]):
            event_date = line
        elif any(loc in line.lower() for loc in ["center", "hall", "venue", "building", "hotel", "conference"]) and not event_location:
            event_location = line
    
    # Second pass to extract agenda items
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a day header
        if ("day" in line.lower() or "june" in line.lower() or "july" in line.lower() or 
            "august" in line.lower() or "september" in line.lower() or 
            "october" in line.lower() or "november" in line.lower() or 
            "december" in line.lower() or "january" in line.lower() or 
            "february" in line.lower() or "march" in line.lower() or 
            "april" in line.lower() or "may" in line.lower()):
            if "(" in line and ")" in line:  # Format like "Day 1 (June 15)"
                current_day = line
            elif ":" in line:  # Format like "Day 1:"
                current_day = line.rstrip(":")
            else:
                current_day = line
        
        # Check if this is an agenda item with time
        elif ("-" in line and ":" in line) or ("AM" in line.upper() and "-" in line) or ("PM" in line.upper() and "-" in line):
            # Split by first dash that separates time from description
            parts = line.split("-", 1)
            if len(parts) >= 2:
                time_slot = parts[0].strip()
                description = parts[1].strip()
                location = ""
                
                # Try to extract location in parentheses
                if "(" in description and ")" in description:
                    location = description[description.find("(")+1:description.find(")")]
                    description = description.replace(f"({location})", "").strip()
                
                # Only add if we have a valid time and description
                if time_slot and description:
                    agenda_items.append({
                        "day": current_day,
                        "time": time_slot,
                        "description": description,
                        "location": location
                    })
    
    # If we didn't find any agenda items but have lines with times, try a simpler approach
    if not agenda_items:
        for line in lines:
            if (":" in line and "-" in line) or ("AM" in line.upper()) or ("PM" in line.upper()):
                parts = line.split("-", 1)
                if len(parts) >= 2:
                    agenda_items.append({
                        "day": "Event Day",
                        "time": parts[0].strip(),
                        "description": parts[1].strip(),
                        "location": ""
                    })
    
    return {
        "event_name": event_name,
        "event_date": event_date,
        "event_location": event_location,
        "agenda": agenda_items
    }

# Function to extract skills from resume
def extract_skills(resume_text):
    # Normalize text - lowercase and remove extra whitespace
    normalized_text = ' '.join(resume_text.lower().split())
    
    # Comprehensive skill keywords list
    technical_skills = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "rust", "php", "swift", "kotlin",
        # Web Development
        "html", "css", "react", "angular", "vue", "node", "express", "flask", "django", "spring", "asp.net",
        "jquery", "bootstrap", "tailwind", "sass", "less", "webpack", "vite", "next.js", "gatsby", "graphql", "rest api",
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd", "jenkins", "github actions", 
        "gitlab ci", "ansible", "puppet", "chef", "prometheus", "grafana", "elk stack", "serverless",
        # Data & AI
        "machine learning", "data science", "ai", "nlp", "computer vision", "deep learning", "tensorflow", 
        "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "tableau", "power bi",
        "data visualization", "data analysis", "data mining", "data modeling", "data engineering", "etl",
        # Databases
        "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "sqlserver", "redis", "elasticsearch",
        "dynamodb", "cassandra", "neo4j", "firebase", "supabase",
        # Mobile Development
        "android", "ios", "react native", "flutter", "xamarin", "ionic", "swift", "objective-c", "kotlin",
        # Version Control
        "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
        # Testing
        "unit testing", "integration testing", "e2e testing", "test automation", "selenium", "cypress", "jest",
        "mocha", "pytest", "junit", "testng"
    ]
    
    soft_skills = [
        # Project Management
        "agile", "scrum", "kanban", "product management", "project management", "jira", "confluence", "trello",
        "asana", "monday", "waterfall", "prince2", "pmp", "lean", "six sigma",
        # Leadership & Communication
        "leadership", "team management", "communication", "presentation", "public speaking", "negotiation",
        "conflict resolution", "mentoring", "coaching", "team building", "stakeholder management",
        # Other Soft Skills
        "problem solving", "critical thinking", "creativity", "time management", "adaptability", "flexibility",
        "collaboration", "teamwork", "customer service", "client relations", "analytical thinking"
    ]
    
    # Combine all skills
    skill_keywords = technical_skills + soft_skills
    
    # Find skills in resume
    found_skills = []
    for skill in skill_keywords:
        # Check for exact matches with word boundaries
        if re.search(r'\b' + re.escape(skill) + r'\b', normalized_text):
            found_skills.append(skill)
    
    # Add some domain-specific skills based on context
    domains = {
        "finance": ["financial analysis", "accounting", "banking", "investment", "trading", "risk management"],
        "healthcare": ["healthcare", "medical", "clinical", "patient care", "pharmaceutical", "biotech"],
        "marketing": ["marketing", "digital marketing", "seo", "content marketing", "social media", "brand management"],
        "education": ["education", "teaching", "curriculum", "instructional design", "e-learning"],
        "retail": ["retail", "e-commerce", "sales", "inventory management", "supply chain"]
    }
    
    for domain, domain_skills in domains.items():
        if domain in normalized_text:
            for skill in domain_skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', normalized_text) and skill not in found_skills:
                    found_skills.append(skill)
    
    return found_skills

# Function to find lunch time
def find_lunch_time(agenda_data):
    if not agenda_data or "agenda" not in agenda_data:
        return None, None
        
    for item in agenda_data.get("agenda", []):
        if item.get("description") and "lunch" in item["description"].lower():
            return item.get("day"), item.get("time")
    return None, None

# Function to calculate time until lunch
def time_until_lunch(lunch_time_str):
    if not lunch_time_str:
        return "Lunch time not found in agenda"
    
    try:
        # Clean the lunch time string and handle special characters
        lunch_time_str = lunch_time_str.replace('\uf0b7', '').strip()
        
        # Parse the lunch time
        lunch_start = lunch_time_str.split("-")[0].strip() if "-" in lunch_time_str else lunch_time_str
        
        # Handle 24-hour format or AM/PM format
        if ":" in lunch_start:
            if "AM" in lunch_start.upper() or "PM" in lunch_start.upper():
                # Extract hour and minute, handling potential non-numeric characters
                time_parts = lunch_start.split(":")
                hour_str = ''.join(c for c in time_parts[0] if c.isdigit())
                lunch_hour = int(hour_str) if hour_str else 12
                
                # Extract minute, handling potential non-numeric characters
                minute_part = time_parts[1].split()[0] if len(time_parts) > 1 else "0"
                minute_str = ''.join(c for c in minute_part if c.isdigit())
                lunch_minute = int(minute_str) if minute_str else 0
                
                # Adjust for PM
                if "PM" in lunch_start.upper() and lunch_hour < 12:
                    lunch_hour += 12
            else:
                # For 24-hour format
                time_parts = lunch_start.split(":")
                hour_str = ''.join(c for c in time_parts[0] if c.isdigit())
                lunch_hour = int(hour_str) if hour_str else 12
                
                minute_str = ''.join(c for c in time_parts[1] if c.isdigit()) if len(time_parts) > 1 else "0"
                lunch_minute = int(minute_str) if minute_str else 0
        else:
            # If no colon, try to extract just the hour
            hour_str = ''.join(c for c in lunch_start if c.isdigit())
            lunch_hour = int(hour_str) if hour_str else 12
            lunch_minute = 0
        
        # Default to 12:30 PM if parsing fails
        if lunch_hour < 1 or lunch_hour > 23:
            lunch_hour = 12
            lunch_minute = 30
        
        # Get current time
        now = datetime.datetime.now()
        lunch_time = now.replace(hour=lunch_hour, minute=lunch_minute, second=0, microsecond=0)
        
        # If lunch time has passed for today, assume it's for tomorrow
        if now > lunch_time:
            lunch_time = lunch_time + datetime.timedelta(days=1)
        
        # Calculate time difference
        time_diff = lunch_time - now
        total_seconds = time_diff.total_seconds()
        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours >= 24:
            days = hours // 24
            hours = hours % 24
            return f"{days} days, {hours} hours and {minutes} minutes until lunch"
        else:
            return f"{hours} hours and {minutes} minutes until lunch"
    except Exception as e:
        # If any error occurs, return a default message with a fixed time
        return "Approximately 2 hours until lunch (12:30 PM)"

# Function to find similar participants based on skills
def find_similar_participants(user_skills, participants):
    if not user_skills or not participants:
        return []
    
    # Create a weighted skill dictionary for the user
    user_skill_weights = {}
    for i, skill in enumerate(user_skills):
        # Give higher weight to skills that appear earlier in the list (assuming they're more important)
        weight = 1.0 - (i * 0.02)  # Gradually decrease weight (max 50 skills)
        if weight < 0.5:  # Minimum weight threshold
            weight = 0.5
        user_skill_weights[skill.lower()] = weight
    
    similar_participants = []
    for participant in participants:
        # Calculate weighted similarity score
        total_weight = 0
        matched_weight = 0
        common_skills = []
        
        # Process participant skills
        for skill in participant["skills"]:
            skill_lower = skill.lower()
            if skill_lower in user_skill_weights:
                weight = user_skill_weights[skill_lower]
                matched_weight += weight
                common_skills.append(skill)
            total_weight += 1
        
        # Add participant's unique skills weight
        for skill in user_skills:
            if skill.lower() not in [s.lower() for s in participant["skills"]]:
                total_weight += 1
        
        # Calculate similarity score
        if total_weight > 0 and common_skills:
            similarity_score = matched_weight / total_weight
            
            # Boost score for participants with more common skills
            skill_count_boost = min(len(common_skills) / 10, 0.3)  # Max 30% boost
            similarity_score += skill_count_boost
            
            # Cap at 1.0
            similarity_score = min(similarity_score, 1.0)
            
            similar_participants.append({
                "name": participant["name"],
                "company": participant.get("company", ""),
                "title": participant.get("title", ""),
                "similarity_score": similarity_score,
                "common_skills": common_skills,
                "skills": participant["skills"]
            })
    
    # Sort by similarity score
    similar_participants.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similar_participants

# Function to recommend sessions based on skills
def recommend_sessions(user_skills, agenda_data):
    if not user_skills or not agenda_data or not agenda_data.get("agenda"):
        return []
    
    # Create a weighted skill dictionary for the user
    user_skill_weights = {}
    for i, skill in enumerate(user_skills):
        # Give higher weight to skills that appear earlier in the list (assuming they're more important)
        weight = 1.0 - (i * 0.02)  # Gradually decrease weight (max 50 skills)
        if weight < 0.5:  # Minimum weight threshold
            weight = 0.5
        user_skill_weights[skill.lower()] = weight
    
    recommendations = []
    for item in agenda_data["agenda"]:
        # Skip breaks and non-session items
        description = item.get("description", "").lower()
        if any(word in description for word in ["break", "lunch", "coffee", "registration", "reception"]):
            continue
            
        # Calculate relevance score
        score = 0
        matched_skills = []
        
        # Check for skill matches in description
        for skill, weight in user_skill_weights.items():
            if skill in description:
                score += weight
                matched_skills.append(skill)
        
        # Add session if it has any relevance
        if score > 0:
            # Normalize score to be between 1-10
            normalized_score = min(round(score * 3), 10)
            
            recommendations.append({
                "session": item.get("description", "Untitled Session"),
                "time": item.get("time", "TBD"),
                "day": item.get("day", "TBD"),
                "location": item.get("location", ""),
                "score": normalized_score,
                "matched_skills": matched_skills
            })
    
    # Sort by score
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_agenda" not in st.session_state:
    st.session_state.uploaded_agenda = False
if "event_data" not in st.session_state:
    st.session_state.event_data = {}
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}
if "participants" not in st.session_state:
    st.session_state.participants = []
if "user_resume" not in st.session_state:
    st.session_state.user_resume = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"
if "current_time" not in st.session_state:
    st.session_state.current_time = datetime.datetime.now()

# Main app header
st.title("🎪 EventMaster AI – Your Intelligent Event Assistant")

# Sidebar with current time and navigation
with st.sidebar:
    # Display current time
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
    
    st.markdown(f'<div class="clock-display">{current_time}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center;">{current_date}</div>', unsafe_allow_html=True)
    
    # Navigation
    st.subheader("Navigation")
    tabs = ["Chat", "Event Schedule", "Networking", "Feedback"]
    for tab in tabs:
        if st.button(tab, key=f"nav_{tab}"):
            st.session_state.active_tab = tab
            st.rerun()
    
    # Display time until lunch if agenda is uploaded
    if st.session_state.uploaded_agenda:
        lunch_day, lunch_time = find_lunch_time(st.session_state.event_data)
        if lunch_time:
            st.subheader("⏱ Time Until Lunch")
            st.markdown(f'<div class="countdown-display">{time_until_lunch(lunch_time)}</div>', unsafe_allow_html=True)
    
    # Upload section
    st.subheader("📤 Upload")
    
    # Agenda upload
    agenda_file = st.file_uploader("Upload Event Agenda", type=["pdf", "txt"], key="agenda_uploader")
    if agenda_file is not None:
        if agenda_file.type == "application/pdf":
            agenda_text = extract_text_from_pdf(agenda_file)
        else:
            agenda_text = agenda_file.getvalue().decode("utf-8")
        
        if agenda_text:
            st.session_state.event_data = parse_agenda(agenda_text)
            st.session_state.uploaded_agenda = True
            st.success("Agenda uploaded and processed successfully!")
    
    # Resume upload
    resume_file = st.file_uploader("Upload Your Resume", type=["pdf", "txt"], key="resume_uploader")
    if resume_file is not None:
        with st.spinner("Analyzing your resume..."):
            try:
                if resume_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(resume_file)
                else:
                    resume_text = resume_file.getvalue().decode("utf-8")
                
                if resume_text:
                    # Extract skills and categorize them
                    skills = extract_skills(resume_text)
                    
                    # Categorize skills
                    tech_skills = []
                    soft_skills = []
                    domain_skills = []
                    
                    for skill in skills:
                        skill_lower = skill.lower()
                        if skill_lower in ["python", "java", "javascript", "html", "css", "react", "angular", 
                                          "aws", "azure", "docker", "kubernetes", "sql", "nosql", "git", 
                                          "machine learning", "data science", "tensorflow", "pytorch"]:
                            tech_skills.append(skill)
                        elif skill_lower in ["leadership", "communication", "teamwork", "problem solving", 
                                            "critical thinking", "agile", "scrum", "project management"]:
                            soft_skills.append(skill)
                        else:
                            domain_skills.append(skill)
                    
                    st.session_state.user_resume = {
                        "text": resume_text,
                        "skills": skills,
                        "tech_skills": tech_skills,
                        "soft_skills": soft_skills,
                        "domain_skills": domain_skills
                    }
                    
                    # Show success message with skill breakdown
                    st.success(f"Resume analyzed successfully! Found {len(skills)} skills.")
                    
                    # Show a preview of skills
                    if skills:
                        with st.expander("Skills Preview"):
                            if tech_skills:
                                st.write("**Technical Skills:**")
                                st.markdown(" ".join([f'<span class="skill-tag">{skill}</span>' for skill in tech_skills[:5]]), unsafe_allow_html=True)
                                if len(tech_skills) > 5:
                                    st.write(f"...and {len(tech_skills) - 5} more")
                            
                            if soft_skills:
                                st.write("**Soft Skills:**")
                                st.markdown(" ".join([f'<span class="skill-tag">{skill}</span>' for skill in soft_skills[:3]]), unsafe_allow_html=True)
                                if len(soft_skills) > 3:
                                    st.write(f"...and {len(soft_skills) - 3} more")
                    else:
                        st.warning("No skills were detected. Try uploading a more detailed resume.")
            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")
                st.info("Try uploading a different file or check if the PDF is text-based rather than scanned images.")
    
    # Sample participant data for demo
    if st.button("Load Sample Participants"):
        st.session_state.participants = [
            {
                "name": "John Smith",
                "skills": ["python", "machine learning", "data science", "tensorflow", "nlp"],
                "company": "AI Solutions Inc.",
                "title": "Senior Data Scientist"
            },
            {
                "name": "Emily Chen",
                "skills": ["java", "spring", "kubernetes", "docker", "aws"],
                "company": "Cloud Systems Ltd.",
                "title": "DevOps Engineer"
            },
            {
                "name": "Michael Wong",
                "skills": ["python", "django", "javascript", "react", "postgresql"],
                "company": "WebTech Solutions",
                "title": "Full Stack Developer"
            },
            {
                "name": "Sarah Johnson",
                "skills": ["machine learning", "python", "data science", "computer vision", "pytorch"],
                "company": "Vision AI",
                "title": "AI Researcher"
            },
            {
                "name": "David Kim",
                "skills": ["javascript", "typescript", "react", "node", "mongodb"],
                "company": "Frontend Masters",
                "title": "Frontend Lead"
            }
        ]
        st.success("Sample participants loaded!")

# Main content area based on active tab
if st.session_state.active_tab == "Chat":
    # Chat interface
    st.subheader("💬 Chat with EventMaster AI")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the event..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Process different types of queries
            if "washroom" in prompt.lower() or "restroom" in prompt.lower() or "bathroom" in prompt.lower() or "toilet" in prompt.lower():
                # Handle washroom question
                response = """
                Here's information about washrooms:
                
                • Main Floor: Across from the registration desk
                • Second Floor: At the top of the main staircase, turn left
                • Near Main Hall: Exit the hall and turn right
                • Near Workshop Rooms: In the corridor between Rooms A and B
                
                Is there anything else you'd like to know about the venue?
                """
            
            elif "lunch" in prompt.lower() or "time until" in prompt.lower() or "how long until" in prompt.lower() or "food" in prompt.lower():
                # Handle lunch time question
                if st.session_state.uploaded_agenda:
                    lunch_day, lunch_time = find_lunch_time(st.session_state.event_data)
                    if lunch_time:
                        response = f"Lunch is scheduled for {lunch_time} on {lunch_day}.\n\n{time_until_lunch(lunch_time)}"
                    else:
                        response = "I couldn't find lunch time in the agenda. Based on typical conference schedules, lunch is usually around 12:30 PM.\n\n" + time_until_lunch("12:30 PM")
                else:
                    response = "Based on the sample event data, lunch is typically scheduled from 12:30 PM to 1:30 PM each day.\n\n" + time_until_lunch("12:30 PM")
            
            elif "agenda" in prompt.lower() or "schedule" in prompt.lower() or "program" in prompt.lower() or "timetable" in prompt.lower():
                # Handle agenda question
                if st.session_state.uploaded_agenda and st.session_state.event_data.get("agenda"):
                    response = f"Here's the agenda for {st.session_state.event_data.get('event_name', 'the event')}:\n\n"
                    
                    # Group by day
                    days = {}
                    for item in st.session_state.event_data.get("agenda", []):
                        day = item.get("day", "Event Day")
                        if day not in days:
                            days[day] = []
                        days[day].append(item)
                    
                    # Sort items within each day by time
                    for day, items in days.items():
                        items.sort(key=lambda x: x.get("time", ""))
                    
                    # Build response
                    for day, items in days.items():
                        response += f"\n**{day}**\n\n"
                        for item in items:
                            location_info = f" ({item['location']})" if item.get('location') else ""
                            response += f"• {item.get('time', 'TBD')} - {item.get('description', 'Session')}{location_info}\n"
                else:
                    # Use sample agenda
                    response = """
                    Here's a sample agenda for the AI Innovation Summit:
                    
                    **Day 1 (June 15)**
                    • 09:00-10:00 - Registration and Welcome Coffee (Main Lobby)
                    • 10:00-11:30 - Keynote: The Future of AI in Events (Main Hall)
                    • 11:30-12:30 - Workshop Session 1: Building Intelligent Chatbots (Room A)
                    • 12:30-13:30 - Networking Lunch (Main Hall)
                    • 13:30-15:00 - Workshop Session 2: AI for Resume Analysis (Room B)
                    • 15:00-15:30 - Coffee Break (Lounge Area)
                    • 15:30-17:00 - Panel Discussion: Ethics in AI (Main Hall)
                    
                    **Day 2 (June 16)**
                    • 09:00-09:30 - Morning Coffee (Main Lobby)
                    • 09:30-11:00 - Keynote: Generative AI Revolution (Main Hall)
                    • 11:00-12:30 - Workshop Session 3: Building Real-time Recommendation Systems (Room A)
                    • 12:30-13:30 - Networking Lunch (Main Hall)
                    • 13:30-15:00 - Workshop Session 4: Advanced NLP Techniques (Room B)
                    
                    Please upload the full agenda for more detailed information.
                    """
            
            elif any(word in prompt.lower() for word in ["similar", "participants", "people", "attendees", "networking", "connect"]):
                # Handle similar participants question
                if st.session_state.user_resume and st.session_state.participants:
                    similar = find_similar_participants(st.session_state.user_resume["skills"], st.session_state.participants)
                    if similar:
                        response = "### Participants with Similar Skills\n\n"
                        response += "I've analyzed the participant list and found these professionals who share skills with you:\n\n"
                        
                        for i, person in enumerate(similar[:3], 1):
                            match_percentage = int(person['similarity_score'] * 100)
                            response += f"#### {i}. {person['name']} ({match_percentage}% match)\n\n"
                            
                            if person.get('company') and person.get('title'):
                                response += f"**{person.get('title')}** at **{person.get('company')}**\n\n"
                            elif person.get('title'):
                                response += f"**{person.get('title')}**\n\n"
                            elif person.get('company'):
                                response += f"Works at **{person.get('company')}**\n\n"
                            
                            # Highlight top common skills
                            if len(person['common_skills']) > 0:
                                response += "**Top shared skills:**\n"
                                top_skills = person['common_skills'][:5]
                                for skill in top_skills:
                                    response += f"- {skill}\n"
                                
                                if len(person['common_skills']) > 5:
                                    response += f"- Plus {len(person['common_skills']) - 5} more shared skills\n"
                            
                            response += "\n"
                        
                        # Add networking tips
                        response += "### Networking Tips\n\n"
                        response += "- Visit the Networking tab to see more details about these participants\n"
                        response += "- Check the event schedule for networking sessions\n"
                        response += "- Consider attending sessions that these participants might be interested in\n"
                    else:
                        response = """
                        ### No Similar Participants Found
                        
                        I couldn't find participants with skills that closely match yours. This could be because:
                        
                        1. Your skills are unique compared to the current participant data
                        2. The participant data may be limited
                        
                        Try:
                        - Uploading a more detailed resume
                        - Checking back later as more participants register
                        - Visiting the Networking tab to browse all participants
                        """
                else:
                    response = """
                    ### Find Similar Participants
                    
                    To help you connect with like-minded professionals, I need:
                    
                    1. **Your resume** - Upload it in the sidebar to extract your skills
                    2. **Participant data** - Click "Load Sample Participants" in the sidebar
                    
                    Once you've done that, I'll analyze skill overlaps and recommend people you might want to connect with during the event.
                    
                    Networking is one of the most valuable aspects of this event!
                    """
            
            elif any(word in prompt.lower() for word in ["recommend", "session", "workshop", "talk", "presentation", "suggestion"]):
                # Handle session recommendations
                if st.session_state.user_resume and st.session_state.uploaded_agenda:
                    recommendations = recommend_sessions(st.session_state.user_resume["skills"], st.session_state.event_data)
                    if recommendations:
                        response = "### Personalized Session Recommendations\n\n"
                        response += "Based on the skills in your resume, I recommend these sessions:\n\n"
                        
                        for i, rec in enumerate(recommendations[:5], 1):
                            response += f"#### {i}. {rec['session']}\n\n"
                            
                            # Session details
                            time_info = f"{rec['day']} at {rec['time']}"
                            location_info = f" in {rec['location']}" if rec['location'] else ""
                            response += f"**When:** {time_info}{location_info}\n\n"
                            
                            # Relevance score with visual indicator
                            stars = "★" * rec['score'] + "☆" * (10 - rec['score'])
                            response += f"**Relevance:** {stars} ({rec['score']}/10)\n\n"
                            
                            # Matched skills
                            if rec.get('matched_skills'):
                                response += "**Matched skills:** " + ", ".join(rec['matched_skills'][:5])
                                if len(rec['matched_skills']) > 5:
                                    response += f" and {len(rec['matched_skills']) - 5} more"
                                response += "\n\n"
                        
                        # Add recommendation explanation
                        response += "### Why These Recommendations?\n\n"
                        response += "These sessions were selected based on how closely they match your skills and expertise. "
                        response += "The relevance score (1-10) indicates how strongly a session aligns with your profile.\n\n"
                        response += "Visit the Event Schedule tab to see the full agenda and plan your day!"
                    else:
                        response = """
                        ### No Matching Sessions Found
                        
                        I couldn't find sessions that closely match your skills. This could be because:
                        
                        1. Your skills are in areas not specifically covered by the current sessions
                        2. The session descriptions may not contain explicit skill keywords
                        
                        **Suggestions:**
                        - Try uploading a more detailed resume
                        - Check the Event Schedule tab to browse all sessions
                        - Consider attending general interest sessions like keynotes and panels
                        """
                else:
                    response = """
                    ### Get Personalized Session Recommendations
                    
                    To provide tailored session recommendations, I need:
                    
                    1. **Your resume** - Upload it in the sidebar so I can identify your skills and interests
                    2. **The event agenda** - Upload it in the sidebar or use the sample data
                    
                    Once you've done that, I'll analyze the sessions and recommend those that best match your professional profile.
                    
                    This will help you maximize the value of your time at the event!
                    """
            
            elif "feedback" in prompt.lower() or "review" in prompt.lower() or "rate" in prompt.lower():
                # Handle feedback collection
                response = """
                I'd love to collect your feedback! You can provide feedback in two ways:
                
                1. Go to the Feedback tab using the navigation in the sidebar
                2. Click on any session in the Event Schedule tab to provide specific feedback
                
                Your feedback helps improve future events and sessions. Thank you!
                """
            
            elif "wifi" in prompt.lower() or "internet" in prompt.lower() or "network" in prompt.lower() or "connection" in prompt.lower():
                # Handle WiFi information
                response = """
                **Wi-Fi Information:**
                
                Network: AI_Summit_2025
                Password: InnovateAI2025
                
                If you have trouble connecting, please visit the registration desk for assistance.
                """
            
            elif "contact" in prompt.lower() or "emergency" in prompt.lower() or "help" in prompt.lower() or "assistance" in prompt.lower():
                # Handle contact information
                response = """
                **Important Contacts:**
                
                - Event Manager: Sarah Johnson (555-123-4567)
                - Technical Support: Michael Wong (555-987-6543)
                - Venue Security: 555-111-2222
                
                For immediate assistance, please visit the registration desk in the Main Lobby.
                """
            
            else:
                # Use Gemini API for general questions
                api_key = os.getenv("GEMINI_API_KEY", "")
                if api_key:
                    # Create context for the bot
                    context = ""
                    if st.session_state.uploaded_agenda:
                        context += f"Event Name: {st.session_state.event_data.get('event_name', '')}\n"
                        context += f"Event Date: {st.session_state.event_data.get('event_date', '')}\n"
                        context += f"Event Location: {st.session_state.event_data.get('event_location', '')}\n\n"
                        context += "Agenda:\n"
                        for item in st.session_state.event_data.get("agenda", []):
                            location_info = f" at {item['location']}" if item.get('location') else ""
                            context += f"- {item.get('day', 'Event Day')} {item.get('time', 'TBD')} - {item.get('description', 'Session')}{location_info}\n"
                    else:
                        # Use sample event info if no agenda is uploaded
                        try:
                            with open("e:/CODE/project/data/sample_event_info.txt", "r") as f:
                                context = f.read()
                        except FileNotFoundError:
                            context = """
                            AI Innovation Summit 2025
                            June 15-17, 2025
                            Tech Conference Center, Silicon Valley

                            Day 1 (June 15):
                            - 09:00-10:00 - Registration and Welcome Coffee (Main Lobby)
                            - 10:00-11:30 - Keynote: The Future of AI in Events (Main Hall)
                            - 11:30-12:30 - Workshop Session 1: Building Intelligent Chatbots (Room A)
                            - 12:30-13:30 - Networking Lunch (Main Hall)
                            - 13:30-15:00 - Workshop Session 2: AI for Resume Analysis (Room B)
                            - 15:00-15:30 - Coffee Break (Lounge Area)
                            - 15:30-17:00 - Panel Discussion: Ethics in AI (Main Hall)
                            
                            Washroom Locations:
                            - Main Floor: Across from the registration desk
                            - Second Floor: At the top of the main staircase, turn left
                            
                            Wi-Fi Information:
                            Network: AI_Summit_2025
                            Password: InnovateAI2025
                            """
                    
                    full_prompt = f"""
                    You are EventMaster AI, an intelligent assistant for the event.
                    Use the following event information to answer questions:
                    
                    {context}
                    
                    Question: {prompt}
                    
                    If you don't have enough information to answer the question accurately, politely say so and suggest what information might be needed.
                    Keep your answers concise, helpful, and focused on the event information.
                    """
                    
                    try:
                        response = query_gemini_api(full_prompt, api_key)
                    except Exception as e:
                        response = f"""
                        I'm sorry, I couldn't process your question at the moment. 
                        
                        Here are some things you can ask me about:
                        - Event schedule and agenda
                        - Lunch times and locations
                        - Washroom locations
                        - Wi-Fi information
                        - Session recommendations
                        - Finding similar participants
                        """
                else:
                    response = """
                    I need a valid API key to answer general questions. 
                    
                    In the meantime, here are some things you can ask me about:
                    - Event schedule and agenda
                    - Lunch times and locations
                    - Washroom locations
                    - Wi-Fi information
                    - Session recommendations
                    - Finding similar participants
                    """
            
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick question buttons
    st.subheader("🔍 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What's on the agenda?"):
            st.session_state.messages.append({"role": "user", "content": "What's on the agenda?"})
            st.rerun()
    
    with col2:
        if st.button("Where are the washrooms?"):
            st.session_state.messages.append({"role": "user", "content": "Where are the washrooms?"})
            st.rerun()
    
    with col3:
        if st.button("How long until lunch?"):
            st.session_state.messages.append({"role": "user", "content": "How long until lunch?"})
            st.rerun()

elif st.session_state.active_tab == "Event Schedule":
    st.subheader("📅 Event Schedule")
    
    if st.session_state.uploaded_agenda:
        # Display event details
        st.markdown(f"### {st.session_state.event_data.get('event_name', 'Event')}")
        st.markdown(f"**Date:** {st.session_state.event_data.get('event_date', 'TBD')}")
        st.markdown(f"**Location:** {st.session_state.event_data.get('event_location', 'TBD')}")
        
        # Create tabs for each day
        days = set(item["day"] for item in st.session_state.event_data.get("agenda", []) if item["day"])
        if days:
            tabs = st.tabs([day.split("(")[1].split(")")[0] if "(" in day and ")" in day else day for day in days])
            
            for i, day in enumerate(days):
                with tabs[i]:
                    # Create timeline for this day
                    day_items = [item for item in st.session_state.event_data.get("agenda", []) if item["day"] == day]
                    day_items.sort(key=lambda x: x["time"])
                    
                    for item in day_items:
                        if "lunch" in item["description"].lower() or "break" in item["description"].lower():
                            st.markdown(f"""
                            <div class="timeline-break">
                                <strong>{item['time']}</strong> - {item['description']}
                                {f"<br><em>Location: {item['location']}</em>" if item['location'] else ""}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="timeline-item">
                                <strong>{item['time']}</strong> - {item['description']}
                                {f"<br><em>Location: {item['location']}</em>" if item['location'] else ""}
                            </div>
                            """, unsafe_allow_html=True)
    else:
        # Display sample agenda
        st.info("Please upload the event agenda to view the full schedule. Here's a sample schedule:")
        
        sample_days = ["Day 1 (June 15)", "Day 2 (June 16)", "Day 3 (June 17)"]
        tabs = st.tabs(["June 15", "June 16", "June 17"])
        
        with tabs[0]:
            st.markdown("""
            <div class="timeline-item">
                <strong>09:00-10:00</strong> - Registration and Welcome Coffee
                <br><em>Location: Main Lobby</em>
            </div>
            <div class="timeline-item">
                <strong>10:00-11:30</strong> - Keynote: The Future of AI in Events
                <br><em>Location: Main Hall</em>
            </div>
            <div class="timeline-item">
                <strong>11:30-12:30</strong> - Workshop Session 1: Building Intelligent Chatbots
                <br><em>Location: Room A</em>
            </div>
            <div class="timeline-break">
                <strong>12:30-13:30</strong> - Networking Lunch
                <br><em>Location: Main Hall</em>
            </div>
            <div class="timeline-item">
                <strong>13:30-15:00</strong> - Workshop Session 2: AI for Resume Analysis
                <br><em>Location: Room B</em>
            </div>
            <div class="timeline-break">
                <strong>15:00-15:30</strong> - Coffee Break
                <br><em>Location: Lounge Area</em>
            </div>
            <div class="timeline-item">
                <strong>15:30-17:00</strong> - Panel Discussion: Ethics in AI
                <br><em>Location: Main Hall</em>
            </div>
            """, unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown("""
            <div class="timeline-item">
                <strong>09:00-09:30</strong> - Morning Coffee
                <br><em>Location: Main Lobby</em>
            </div>
            <div class="timeline-item">
                <strong>09:30-11:00</strong> - Keynote: Generative AI Revolution
                <br><em>Location: Main Hall</em>
            </div>
            <div class="timeline-item">
                <strong>11:00-12:30</strong> - Workshop Session 3: Building Real-time Recommendation Systems
                <br><em>Location: Room A</em>
            </div>
            <div class="timeline-break">
                <strong>12:30-13:30</strong> - Networking Lunch
                <br><em>Location: Main Hall</em>
            </div>
            <div class="timeline-item">
                <strong>13:30-15:00</strong> - Workshop Session 4: Advanced NLP Techniques
                <br><em>Location: Room B</em>
            </div>
            <div class="timeline-break">
                <strong>15:00-15:30</strong> - Coffee Break
                <br><em>Location: Lounge Area</em>
            </div>
            <div class="timeline-item">
                <strong>15:30-17:00</strong> - Hands-on Lab: Implementing RAG Systems
                <br><em>Location: Room C</em>
            </div>
            """, unsafe_allow_html=True)
        
        with tabs[2]:
            st.markdown("""
            <div class="timeline-item">
                <strong>09:00-09:30</strong> - Morning Coffee
                <br><em>Location: Main Lobby</em>
            </div>
            <div class="timeline-item">
                <strong>09:30-11:00</strong> - Workshop Session 5: AI for Event Management
                <br><em>Location: Room A</em>
            </div>
            <div class="timeline-item">
                <strong>11:00-12:30</strong> - Hackathon Presentations
                <br><em>Location: Main Hall</em>
            </div>
            <div class="timeline-break">
                <strong>12:30-13:30</strong> - Networking Lunch
                <br><em>Location: Main Hall</em>
            </div>
            <div class="timeline-item">
                <strong>13:30-15:00</strong> - Closing Keynote: The Road Ahead
                <br><em>Location: Main Hall</em>
            </div>
            <div class="timeline-item">
                <strong>15:00-16:00</strong> - Awards Ceremony & Closing Remarks
                <br><em>Location: Main Hall</em>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.active_tab == "Networking":
    st.markdown('<h2 class="sub-header">👥 Networking & Recommendations</h2>', unsafe_allow_html=True)
    
    # Check if resume is uploaded
    if not st.session_state.user_resume:
        st.warning("⚠️ Please upload your resume in the sidebar to unlock personalized networking features.")
        
        # Show sample profile
        st.markdown("### Sample Profile Preview")
        st.markdown("""
        <div class="card">
            <h3>Your Professional Profile</h3>
            <p>Upload your resume to see your skills analyzed and get personalized recommendations.</p>
            <p>We'll extract your skills and help you:</p>
            <ul>
                <li>Find participants with similar interests</li>
                <li>Recommend relevant sessions</li>
                <li>Highlight networking opportunities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Resume analysis section
        tabs = st.tabs(["Your Profile", "Similar Participants", "Recommended Sessions"])
        
        with tabs[0]:
            st.markdown("### Your Professional Profile")
            
            # Create columns for profile layout
            profile_col1, profile_col2 = st.columns([3, 2])
            
            with profile_col1:
                # Display skills by category
                if st.session_state.user_resume.get("tech_skills"):
                    st.markdown("#### Technical Skills")
                    st.markdown('<div style="line-height: 2.5;">' + 
                                ' '.join([f'<span class="skill-tag">{skill}</span>' for skill in st.session_state.user_resume["tech_skills"]]) + 
                                '</div>', unsafe_allow_html=True)
                
                if st.session_state.user_resume.get("soft_skills"):
                    st.markdown("#### Soft Skills")
                    st.markdown('<div style="line-height: 2.5;">' + 
                                ' '.join([f'<span class="skill-tag" style="background-color: #3F51B5;">{skill}</span>' for skill in st.session_state.user_resume["soft_skills"]]) + 
                                '</div>', unsafe_allow_html=True)
                
                if st.session_state.user_resume.get("domain_skills"):
                    st.markdown("#### Domain Knowledge")
                    st.markdown('<div style="line-height: 2.5;">' + 
                                ' '.join([f'<span class="skill-tag" style="background-color: #4CAF50;">{skill}</span>' for skill in st.session_state.user_resume["domain_skills"]]) + 
                                '</div>', unsafe_allow_html=True)
            
            with profile_col2:
                # Display a simple visualization
                if st.button("Generate Skill Insights"):
                    st.markdown("### Skill Distribution")
                    
                    # Create data for pie chart
                    categories = {
                        "Technical": len(st.session_state.user_resume.get("tech_skills", [])),
                        "Soft": len(st.session_state.user_resume.get("soft_skills", [])),
                        "Domain": len(st.session_state.user_resume.get("domain_skills", []))
                    }
                    
                    # Filter out empty categories
                    categories = {k: v for k, v in categories.items() if v > 0}
                    
                    if categories:
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%',
                              colors=['#7E57C2', '#3F51B5', '#4CAF50'])
                        ax.axis('equal')
                        st.pyplot(fig)
                        
                        # Add skill insights
                        total_skills = sum(categories.values())
                        primary_category = max(categories.items(), key=lambda x: x[1])[0]
                        
                        st.markdown(f"""
                        #### Skill Insights
                        
                        - You have **{total_skills} skills** across {len(categories)} categories
                        - Your profile is strongest in **{primary_category} Skills**
                        - This profile will help you connect with like-minded professionals
                        """)
                    else:
                        st.warning("Not enough skills to generate insights.")
        
        with tabs[1]:
            st.markdown("### Connect with Similar Professionals")
            
            if not st.session_state.participants:
                st.info("👉 Click 'Load Sample Participants' in the sidebar to see potential connections.")
            else:
                similar = find_similar_participants(st.session_state.user_resume["skills"], st.session_state.participants)
                
                if similar:
                    # Create a grid layout for participants
                    cols = st.columns(2)
                    
                    for i, person in enumerate(similar[:6]):
                        col_idx = i % 2
                        match_percentage = int(person['similarity_score'] * 100)
                        
                        with cols[col_idx]:
                            st.markdown(f"""
                            <div class="card">
                                <h3>{person['name']} <span style="float:right; color:#7E57C2;">{match_percentage}%</span></h3>
                                <p><strong>{person.get('title', 'Professional')}</strong> at {person.get('company', 'Company')}</p>
                                <hr>
                                <p><strong>Common Interests:</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show common skills with nice formatting
                            st.markdown('<div style="line-height: 2.5;">' + 
                                        ' '.join([f'<span class="skill-tag">{skill}</span>' for skill in person["common_skills"][:8]]) + 
                                        '</div>', unsafe_allow_html=True)
                            
                            # Add a connect button (for demonstration)
                            if st.button(f"Connect with {person['name'].split()[0]}", key=f"connect_{i}"):
                                st.success(f"Connection request sent to {person['name']}!")
                                st.balloons()
                else:
                    st.warning("No similar participants found with the current data.")
                    st.info("Try uploading a more detailed resume or check back when more participants have registered.")
        
        with tabs[2]:
            st.markdown("### Personalized Session Recommendations")
            
            if not st.session_state.uploaded_agenda:
                st.info("👉 Upload the event agenda in the sidebar to get personalized session recommendations.")
            else:
                recommendations = recommend_sessions(st.session_state.user_resume["skills"], st.session_state.event_data)
                
                if recommendations:
                    # Create a more visual representation of recommendations
                    for i, rec in enumerate(recommendations[:5]):
                        # Calculate a color based on score (higher score = more intense color)
                        color_intensity = min(0.4 + (rec['score'] / 20), 0.95)  # Scale from 0.4 to 0.95
                        bg_color = f"rgba(126, 87, 194, {color_intensity})"
                        
                        # Create a visually appealing recommendation card
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                            <h3>{rec['session']}</h3>
                            <p><strong>When:</strong> {rec['day']} at {rec['time']}</p>
                            <p><strong>Where:</strong> {rec['location'] if rec['location'] else 'TBD'}</p>
                            <p><strong>Relevance:</strong> {'★' * rec['score']}{'☆' * (10 - rec['score'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show matched skills
                        if rec.get('matched_skills'):
                            st.markdown("**Why this is relevant to you:**")
                            st.markdown('<div style="line-height: 2.5; margin-bottom: 20px;">' + 
                                        ' '.join([f'<span class="skill-tag">{skill}</span>' for skill in rec['matched_skills']]) + 
                                        '</div>', unsafe_allow_html=True)
                        
                        # Add to calendar button (for demonstration)
                        if st.button(f"Add to My Schedule", key=f"add_session_{i}"):
                            st.success(f"Added to your personal schedule!")
                else:
                    st.warning("No session recommendations found based on your skills.")
                    st.info("Try uploading a more detailed resume or check the Event Schedule tab to browse all sessions.")

elif st.session_state.active_tab == "Feedback":
    st.subheader("📝 Session Feedback")
    
    # Create a feedback form
    st.markdown("### Provide Your Feedback")
    st.write("Your feedback helps us improve future events and sessions.")
    
    # Session selection
    if st.session_state.uploaded_agenda:
        sessions = [item["description"] for item in st.session_state.event_data.get("agenda", []) 
                   if "break" not in item["description"].lower() and "lunch" not in item["description"].lower()]
    else:
        sessions = [
            "Keynote: The Future of AI in Events",
            "Workshop Session 1: Building Intelligent Chatbots",
            "Workshop Session 2: AI for Resume Analysis",
            "Panel Discussion: Ethics in AI",
            "Keynote: Generative AI Revolution",
            "Workshop Session 3: Building Real-time Recommendation Systems",
            "Workshop Session 4: Advanced NLP Techniques",
            "Hands-on Lab: Implementing RAG Systems",
            "Workshop Session 5: AI for Event Management",
            "Closing Keynote: The Road Ahead"
        ]
    
    selected_session = st.selectbox("Select a session", sessions)
    
    # Rating
    rating = st.slider("How would you rate this session?", 1, 5, 4)
    
    # Feedback text
    feedback_text = st.text_area("Your feedback", height=150, 
                                placeholder="Please share your thoughts on the session content, speaker, and any suggestions for improvement...")
    
    # Submit button
    if st.button("Submit Feedback"):
        if feedback_text:
            # Store feedback in session state
            if "feedback" not in st.session_state:
                st.session_state.feedback = {}
            
            st.session_state.feedback[selected_session] = {
                "rating": rating,
                "feedback": feedback_text,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.feedback_given[selected_session] = True
            
            st.success("Thank you for your feedback! Your input helps us improve future events.")
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "user", 
                "content": f"I'd like to provide feedback for the session: {selected_session}"
            })
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Thank you for providing feedback for '{selected_session}'. Your input is valuable to us!"
            })
        else:
            st.warning("Please enter some feedback before submitting.")
    
    # Display submitted feedback
    if "feedback" in st.session_state and st.session_state.feedback:
        st.subheader("Your Submitted Feedback")
        
        for session, feedback in st.session_state.feedback.items():
            with st.expander(f"{session} - Rating: {'⭐' * feedback['rating']}"):
                st.write(f"**Submitted on:** {feedback['timestamp']}")
                st.write(f"**Your feedback:** {feedback['feedback']}")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #666;">
    <p>EventMaster AI © 2025 | Powered by Gemini AI</p>
</div>
""", unsafe_allow_html=True)