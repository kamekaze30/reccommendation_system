# 🍽️ IAS2313 Artificial Intelligence: Presentation Slides
## AI-Based Restaurant & Cafe Recommendation System

---

## SLIDE 1: TITLE SLIDE
- **Project Title:** Development of an AI-Based Restaurant & Cafe Recommendation System
- **Course:** IAS2313 Artificial Intelligence (20% Individual Project)
- **Student Name:** Zaim
- **Matric ID:** AI-IAS2313-2026
- **Lecturer:** Dr. Artificial Intelligence
- **Live Production URL:** [https://reccommendation-system.onrender.com/](https://reccommendation-system.onrender.com/)
- **GitHub Repository:** [https://github.com/kamekaze30/reccommendation_system](https://github.com/kamekaze30/reccommendation_system)
- **Date:** 5th August 2026

---

## SLIDE 2: PROBLEM STATEMENT & PROJECT OBJECTIVES
- **Choice Paralysis in Urban Dining:** Diners spend 15–30 minutes browsing review sites, struggling to choose restaurants matching their budget in Ringgit Malaysia (RM), location, and cuisine taste.
- **Complex Multi-Attribute Decision Making:** Choosing a dining venue depends on cuisine, price tier, location area, diner rating, and amenities (Halal, Vegan, Outdoor, WiFi).
- **Project Objectives:**
  1. Process 500 restaurant venues (`restaurants.csv`) & 10,000 diner rating interactions (`restaurant_ratings.csv`).
  2. Implement 4 AI recommendation models: Popularity Baseline, Content-Based TF-IDF, Item-Item KNN Collaborative Filtering, and Hybrid AI Ensemble.
  3. Feature localized Ringgit Malaysia (RM) pricing (`RM 14.50` to `RM 150.00`) and a 50+ HD Unsplash food photo catalog.
  4. Build a Multi-Attribute Smart Search engine for free-text keyword matching (`Burger`, `Italian`, `Chicago`, `WiFi`).
  5. Deploy responsive Flask & Sarab UI web app live on Render cloud infrastructure ([https://reccommendation-system.onrender.com/](https://reccommendation-system.onrender.com/)).
  6. Achieve 100% test pass rate across automated Python unit tests (`test_app.py`).

---

## SLIDE 3: DATASET & PREPROCESSING PIPELINE
- **`restaurants.csv`**: 500 curated venue records across 5 urban areas (Chicago, Houston, Los Angeles, New York, San Francisco) and 10+ cuisines.
- **`restaurant_ratings.csv`**: 10,000 explicit user rating interactions across registered diner profiles.
- **Preprocessing & Feature Engineering:**
  - **Feature Concatenation (`content_tags`):** Combines venue name, cuisine, category, location, and amenities into a unified textual representation for TF-IDF modeling.
  - **Deterministic RM Pricing Engine (`compute_rm_price`):** Calculates localized price tiers in Ringgit Malaysia (`RM 14.50` to `RM 150.00`).
  - **HD Visual Photo Catalog (`get_unique_venue_image`):** Maps 50+ unique Unsplash food photos by cuisine category, eliminating generic image duplication.

---

## SLIDE 4: SYSTEM ARCHITECTURE & AI ALGORITHMS
1. **Popularity & Location Baseline:** Filters venues by Location Area, Cuisine, RM Price Tier, and ranks by diner rating.
2. **Content-Based Filtering (TF-IDF + Cosine Similarity):** Vectorizes textual `content_tags` into sparse TF-IDF matrices to compute pairwise Cosine Similarity angles.
3. **Multi-Attribute Smart Search Engine:** Vectorizes free-text search queries (`tfv.transform([query])`) in real time, preventing zero-result search failures when searching for keywords like `Burger`, `Italian`, or `Chicago`.
4. **Collaborative Filtering (Item-Item Nearest Neighbors):** Fits an unsupervised `NearestNeighbors(metric='cosine', algorithm='brute')` model on the User-Restaurant rating matrix to identify venues visited by diners with identical taste preferences.
5. **Hybrid AI Ensemble Model:** Fuses Content-Based TF-IDF similarity vectors with Collaborative User rating pattern matching for cold-start mitigation.

---

## SLIDE 5: USER INTERFACE (FLASK & SARAB UI)
- **Modern Soft Elevation Cards:** Displays venue photos, ratings, RM prices, and explicit **"Why Recommended" Explanation Tags** (e.g., `💡 Matched via Hybrid Ensemble`).
- **0ms Instant Place Details Popup Modal:** Displays 0ms instant place details, aspect rating breakdown (Food 4.9, Service 4.8, Ambience 4.9, Value 4.7), and verified customer reviews.
- **🗺️ 4-in-1 Real-World Live Action Hub:** Direct live links inside inspection modals:
  1. `📍 Google Maps Navigation` (Live turn-by-turn directions)
  2. `🌐 Web Search Info` (Official website & menu info)
  3. `⭐ TripAdvisor Reviews`
  4. `🛵 GrabFood Delivery`
- **❤️ Saved Favorites Library:** Allows diners to bookmark favorite spots persisted in browser `localStorage`.
- **🪄 3-Step Foodie Craving Quiz:** Interactive concierge matching based on occasion, atmosphere vibe, and RM budget.

---

## SLIDE 6: EMPIRICAL TESTING & SYSTEM DEMO
- **Automated Unit Testing (`test_app.py`):**
  - Tested 23 test cases covering pricing, unique photo mapping, recommendation algorithms, Multi-Attribute search fallbacks, and Flask REST API endpoints.
  - **Results:** `Ran 23 tests in 0.067s - OK (100% Pass Rate)`.
- **Live Production Cloud Deployment:**
  - Deployed live on Render.com with Gunicorn WSGI web server.
  - **Live URL:** [https://reccommendation-system.onrender.com/](https://reccommendation-system.onrender.com/)

---

## SLIDE 7: CONCLUSION & FUTURE ENHANCEMENTS
- **Conclusion:** Successfully meets all requirements of the IAS2313 assessment. Fuses advanced AI algorithms with Ringgit Malaysia pricing, Sarab web UI, 4-in-1 real-world links, and cloud deployment.
- **Future Work:**
  1. Real-time GPS Geolocation distance sorting in kilometers.
  2. Deep Learning BERT NLP Sentiment Analysis on customer review text.

---

*Thank you! Live Web App Demo & Questions:*  
🌐 **https://reccommendation-system.onrender.com/**
