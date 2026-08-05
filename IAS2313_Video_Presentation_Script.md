# 🎥 IAS2313 Video Presentation Script
## AI Restaurant & Cafe Recommendation System (5-Minute Video Presentation Script)

---

### [0:00 - 0:45] INTRODUCTION & PROJECT OVERVIEW (SLIDES 1 & 2)
*"Hello Dr. and everyone. My name is Zaim, presenting my IAS2313 Artificial Intelligence Individual Project: the AI-Based Restaurant & Cafe Recommendation System."*

*"In urban dining, consumers face choice paralysis when picking a restaurant or cafe. Browsing through hundreds of places based on cuisine, Ringgit Malaysia (RM) budget, location area, and ratings takes significant time. My project solves this problem by building an intelligent multi-model recommendation engine that delivers personalized dining suggestions in milliseconds."*

*"The project is deployed live on cloud production infrastructure at `https://reccommendation-system.onrender.com/`."*

---

### [0:45 - 2:00] SYSTEM ARCHITECTURE & AI ALGORITHMS (SLIDES 3 & 4)
*"Let me explain the core AI algorithms powering the system in `app.py`:"*

1. *"First, **Content-Based Filtering** using TF-IDF vectorization and Cosine Similarity. It converts cuisine, category, location area, and amenity tags into TF-IDF sparse feature vectors to compute pairwise similarity angles between venues."*
2. *"Second, **Multi-Attribute Smart Search**. If a diner searches for free-text keywords like 'Burger', 'Italian', or 'Chicago', the system vectorizes the query in real-time with `tfv.transform()`, guaranteeing accurate similarity matches without zero-result search errors."*
3. *"Third, **Collaborative Filtering** using Item-Item k-Nearest Neighbors. It constructs a sparse User-Restaurant rating matrix across 10,000 ratings and fits a `NearestNeighbors` model with Cosine metric to identify venues visited by diners with matching taste."*
4. *"Fourth, **Hybrid AI Ensemble Model**, which fuses Content-Based TF-IDF vectors with Collaborative User rating patterns to eliminate cold-start challenges."*
5. *"Fifth, **Ringgit Malaysia (RM) Pricing Engine** (`compute_rm_price`) and **50+ HD Unsplash Photo Catalog** (`get_unique_venue_image`), ensuring deterministic RM pricing and unique food photos for every single venue."*

---

### [2:00 - 3:30] LIVE SYSTEM DEMONSTRATION (SLIDE 5)
*"Now, let's look at the live web application demo running on Flask and Sarab UI at `https://reccommendation-system.onrender.com/`:"*

- *"Notice the **Recommendation Grid Cards**, displaying HD food photos, rating badges, RM prices (e.g. RM 35.50), and explicit **`💡 Why Recommended`** explanation tags on every card."*
- *"When I click **`Inspect Details`** on any card, a **0ms Instant Inspection Modal** opens immediately, presenting venue details, aspect rating breakdowns (Food, Service, Ambience, Value), and verified customer reviews."*
- *"Inside the popup modal, notice Section 2: **`🗺️ 4-in-1 Real-World Live Action Hub`**. Diners can click direct live action buttons for **📍 Google Maps Navigation**, **🌐 Web Search Info**, **⭐ TripAdvisor Reviews**, and **🛵 GrabFood Delivery**."*
- *"Diners can also click the ❤️ heart icon to bookmark favorite spots, persisted in browser `localStorage` under **Saved Spots**, or try the **🪄 3-Step Foodie Craving Quiz**."*

---

### [3:30 - 4:30] EXPERIMENTAL RESULTS & UNIT TESTING (SLIDE 6)
*"Evaluating system performance and reliability:"*
- *"We constructed an automated unit test suite (`test_app.py`) evaluating 23 test cases across pricing, unique photo mapping, recommendation models, search fallbacks, and Flask API endpoints."*
- *"The entire test suite passes cleanly: `Ran 23 tests in 0.067s - OK (100% Pass Rate)`."*
- *"The system is hosted live on Render cloud infrastructure with Gunicorn WSGI server, delivering sub-10ms response times."*

---

### [4:30 - 5:00] CONCLUSION & FUTURE WORK (SLIDE 7)
*"In conclusion, the AI Restaurant & Cafe Recommendation System fulfills all requirements of the IAS2313 assessment. Future enhancements include real-time GPS Geolocation distance sorting and Deep Learning BERT NLP sentiment analysis on customer reviews."*

*"Thank you for watching! Try the live app at https://reccommendation-system.onrender.com/"*
