# 🎬 CineMatch-AI

### AI-Powered Movie Recommendation System with Sentiment Analysis, Behavioral Personalization & Natural Language Discovery

CineMatch-AI is an intelligent movie recommendation platform designed to go beyond traditional genre-based recommendations.

It combines **content-based filtering, behavioral personalization, sentiment analysis, natural-language intent understanding, TMDB metadata, and explainable recommendations** to create a more personalized movie discovery experience.

Instead of simply asking *"What genre do you like?"*, CineMatch-AI tries to understand **what you actually want to watch right now.**

---

## ✨ What Makes CineMatch-AI Different?

Most movie recommendation systems rely primarily on ratings, genres, or similarity between movies.

CineMatch-AI combines multiple signals:

> **Movie Content + User Behavior + Review Sentiment + Natural Language Intent + Context**

This allows the system to understand requests such as:

* 🎭 "Give me a dark psychological thriller"
* 😂 "I want something funny and light"
* 🧠 "Recommend movies similar to Inception"
* ❤️ "Something emotional but not too depressing"
* 🔥 "I want an intense action movie"
* 🌙 "Suggest something relaxing for tonight"

The system then converts the user's intent into meaningful recommendation signals.

---

## 🚀 Key Features

### 🎯 Hybrid Recommendation Engine

Combines multiple recommendation strategies rather than relying on a single algorithm.

* Content-based similarity
* Genre and metadata matching
* Behavioral personalization
* Sentiment signals
* Reference-title similarity
* Weighted recommendation scoring

---

### 🧠 AI Movie Discovery

CineMatch-AI can interpret natural-language movie requests and extract meaningful signals such as:

* Genres
* Mood
* Tone
* Intensity
* Themes
* Reference movies
* User intent

This transforms movie search from a traditional filter-based interface into an **AI-assisted discovery experience**.

---

### 💬 Sentiment Analysis

Movie reviews are analyzed to understand audience sentiment.

The system can derive:

* Overall sentiment
* Positive/negative sentiment
* Aspect-level sentiment
* Review-based signals

These signals are incorporated into the recommendation pipeline.

---

### 👤 Behavioral Personalization

The system tracks meaningful user interactions and uses them as recommendation signals.

Examples include:

* Movies viewed
* Movies selected
* User interactions
* Preference patterns

Over time, these signals can influence the recommendation ranking.

---

### ❓ "Why This Movie?"

Recommendations aren't just presented as unexplained results.

CineMatch-AI generates explanations describing **why a movie was recommended**, based on the signals contributing to its score.

For example:

> Recommended because it matches your preferred genre, shares characteristics with movies you've interacted with, and has strongly positive audience sentiment.

This makes the recommendation system more **transparent and explainable**.

---

### ⚡ Sentiment Caching

Sentiment analysis and review processing can be computationally expensive.

CineMatch-AI includes a process-local **TTL sentiment cache** to reduce repeated:

* Review API requests
* Sentiment model predictions
* Aspect analysis
* TMDB/API calls

This improves response efficiency while preserving existing recommendation behavior.

---

### 🎞️ TMDB Integration

CineMatch-AI uses **The Movie Database (TMDB)** to retrieve movie metadata and enrich the recommendation experience with real-world movie information.

---

## 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │      React UI        │
                         │   Movie Discovery    │
                         └──────────┬───────────┘
                                    │
                                    │ REST API
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │      Backend         │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
      ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
      │ Recommendation│     │   Sentiment   │     │  AI Discovery │
      │    Engine     │     │    Analysis   │     │    / Intent   │
      └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Movie Data / DB    │
                         │   + TMDB Metadata    │
                         └──────────────────────┘
```

---

## 🧩 Recommendation Pipeline

```text
User Request
     │
     ▼
Natural Language Understanding
     │
     ├── Genre
     ├── Mood
     ├── Tone
     ├── Themes
     └── Reference Movie
     │
     ▼
Candidate Movie Retrieval
     │
     ▼
Content Similarity
     │
     ▼
Behavioral Personalization
     │
     ▼
Sentiment Signals
     │
     ▼
Hybrid Ranking
     │
     ▼
Explainable Recommendations
```

---

## 🛠️ Tech Stack

### Frontend

* React 19
* JavaScript
* HTML5
* CSS3

### Backend

* Python
* FastAPI
* REST APIs

### Machine Learning / NLP

* Content-based recommendation
* Similarity-based ranking
* Sentiment analysis
* Aspect-level sentiment analysis
* Natural-language intent extraction
* Behavioral personalization

### Data & APIs

* TMDB API
* Movie datasets
* SQLite
* Cached sentiment analysis

### Development & Testing

* Git
* GitHub
* Pytest

---

## 📂 Project Structure

```text
CineMatch-AI/
│
├── backend/
│   ├── app/
│   │   ├── services/
│   │   ├── models/
│   │   ├── routes/
│   │   └── ...
│   │
│   ├── tests/
│   ├── requirements.txt
│   └── ...
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── data/
│   ├── movies_dataset.csv
│   └── ...
│
├── reports/
│
├── .gitignore
└── README.md
```

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Swaminathaan17/CineMatch-AI.git
cd CineMatch-AI
```

### 2. Backend Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Configure Environment Variables

Create:

```text
backend/.env
```

Add the required API configuration:

```env
TMDB_API_KEY=your_tmdb_api_key
```

Never commit your real API key to GitHub.

### 4. Start the Backend

```bash
cd backend
python -m uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 5. Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

---

## 🧪 Testing

The backend includes automated tests covering the major recommendation and personalization functionality.

Run:

```bash
pytest
```

Focused Phase 2 testing achieved:

```text
124 passed
```

---

## 🔬 Major Development Phases

### Phase 1 — Core Recommendation System

Established the foundation:

* Movie recommendation pipeline
* Content-based recommendations
* API architecture
* Frontend/backend integration

### Phase 2 — Intelligent Personalization

Expanded the system with:

* Behavioral signal tracking
* Personalized recommendation ranking
* Explainable recommendations
* AI movie discovery
* Natural-language intent understanding
* Mood/tone/genre/reference-title extraction
* Sentiment integration
* Sentiment caching
* Performance improvements
* Expanded automated testing

**Phase 2 is the final planned version of the project.**

---

## 📈 Engineering Focus

CineMatch-AI was developed with more than just recommendation accuracy in mind.

The project focuses on:

* Modular backend architecture
* Separation of recommendation services
* API-driven design
* Caching expensive operations
* Explainable AI
* Behavioral personalization
* Automated testing
* External API integration
* Maintainability and scalability

---

## 🔐 Security

Sensitive configuration such as API keys should be stored in environment variables.

```text
.env
```

should never contain credentials that are committed to the repository.

---

## 🎯 Future Possibilities

Although Phase 2 is considered the final project scope, the architecture leaves room for future experimentation such as:

* Collaborative filtering
* Deep-learning recommendation models
* Vector databases
* Embedding-based semantic search
* Advanced user profiles
* Large-scale recommendation evaluation
* Real-time recommendation adaptation

These are **possible extensions**, not part of the current project scope.

---

## 👨‍💻 Author

**Swaminathaan M**

Computer Science & Engineering Student
Chennai Institute of Technology

Interested in:

* Machine Learning
* Artificial Intelligence
* Full-Stack Development
* Data Structures & Algorithms
* Building real-world software

---

## ⭐ Project

If you find CineMatch-AI interesting, consider giving the repository a ⭐ on GitHub.

**CineMatch-AI — Movie recommendations that understand what you actually want to watch.**
