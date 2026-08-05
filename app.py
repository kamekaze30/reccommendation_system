import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__, static_folder='static', template_folder='templates')

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
restaurants_df = pd.read_csv('restaurants.csv')
ratings_df = pd.read_csv('restaurant_ratings.csv')

# Clean missing critical attributes
restaurants_df.dropna(subset=['restaurant_id', 'name', 'cuisine', 'rating'], inplace=True)
restaurants_df.reset_index(drop=True, inplace=True)

# Extract unique location list from dataset
unique_locations = sorted(restaurants_df['location'].dropna().astype(str).unique().tolist())
unique_cuisines = sorted(restaurants_df['cuisine'].dropna().astype(str).unique().tolist())

# Combine textual features for Content-Based TF-IDF matching
restaurants_df['content_tags'] = (
    restaurants_df['name'].fillna('') + ' ' +
    restaurants_df['cuisine'].fillna('') + ' ' + 
    restaurants_df.get('category', pd.Series(['']*len(restaurants_df))).fillna('') + ' ' + 
    restaurants_df['location'].fillna('') + ' ' + 
    restaurants_df.get('features', pd.Series(['']*len(restaurants_df))).fillna('')
)

# ==========================================
# 2. MODEL INITIALIZATION & MATRIX BUILDING
# ==========================================
tfv = TfidfVectorizer(min_df=1, token_pattern=r'[\w-]+', stop_words='english')
tfv_matrix = tfv.fit_transform(restaurants_df['content_tags'])
indices = pd.Series(restaurants_df.index, index=restaurants_df['name'].str.lower()).drop_duplicates()

data_merged = pd.merge(ratings_df, restaurants_df[['restaurant_id', 'name']], on='restaurant_id')
pivot_table = data_merged.pivot_table(index='name', columns='user_id', values='rating').fillna(0)
pivot_matrix = csr_matrix(pivot_table.values)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(pivot_matrix)
pivot_names_lower = [name.lower() for name in pivot_table.index]

featured_venue_choices = sorted(restaurants_df['name'].head(30).tolist())

CUISINE_IMAGE_MAP = {
    'JAPANESE': [
        'https://images.unsplash.com/photo-1579871494447-9811cf80d66c?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1611143669185-af224c5e3252?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1569718212165-3a8278d5f624?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1583623025817-d180a2221d0a?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1553621042-f6e147245754?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1617196034796-73dfa7b1fd56?w=600&auto=format&fit=crop'
    ],
    'ITALIAN': [
        'https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1513104890138-7c749659a591?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1595295333158-4742f28fbd85?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1604382354936-07c5d9983bd3?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1621996346565-e3d5d6281862?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1574071318508-1cdbab80d002?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=600&auto=format&fit=crop'
    ],
    'CAFE': [
        'https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1559925393-8be0ec4767c8?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=600&auto=format&fit=crop'
    ],
    'WESTERN': [
        'https://images.unsplash.com/photo-1544025162-d76694265947?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1558030006-450675393462?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1460306855393-0410f61241c7?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1550547660-d9450f859349?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1529193591184-b1d58069ecdd?w=600&auto=format&fit=crop'
    ],
    'MALAYSIAN': [
        'https://images.unsplash.com/photo-1563245372-f21724e3856d?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1596797038530-2c107229654b?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1512058564366-18510be2db19?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1541832676-9b763b0239ab?w=600&auto=format&fit=crop'
    ],
    'DEFAULT': [
        'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1552566626-52f8b828add9?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1543007630-9710e4a00a20?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1559339352-11d035aa65de?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1537047902294-62a40c20a6ae?w=600&auto=format&fit=crop',
        'https://images.unsplash.com/photo-1550966871-3ed3cdb5ed0c?w=600&auto=format&fit=crop'
    ]
}

def get_unique_venue_image(row):
    """Returns a unique HD food image URL based on venue ID & cuisine type"""
    try:
        r_id = int(row.get('restaurant_id', 1))
    except:
        r_id = hash(str(row.get('name', ''))) % 500
        
    cuisine = str(row.get('cuisine', '')).upper()
    
    pool = CUISINE_IMAGE_MAP['DEFAULT']
    for key in CUISINE_IMAGE_MAP:
        if key in cuisine:
            pool = CUISINE_IMAGE_MAP[key]
            break
            
    idx = (r_id * 11 + 3) % len(pool)
    return pool[idx]

def compute_rm_price(row):
    """Computes deterministic random logic price in RM (Ringgit Malaysia) based on venue ID & price tier"""
    price_tier = str(row.get('price_range', '$$'))
    try:
        r_id = int(row.get('restaurant_id', 1))
    except:
        r_id = hash(str(row.get('name', ''))) % 500
        
    offset = (r_id * 13 + 7) % 35
    cents = (r_id * 17) % 90
    cents_str = f"{cents:02d}" if cents > 10 else "50"
    
    if price_tier == '$':
        base = 14 + (offset % 18)
        return f"RM {base}.{cents_str}"
    elif price_tier == '$$':
        base = 32 + (offset % 30)
        return f"RM {base}.{cents_str}"
    elif price_tier == '$$$':
        base = 65 + (offset % 50)
        return f"RM {base}.00"
    else:
        base = 120 + (offset * 4)
        return f"RM {base}.00"

def generate_sample_reviews(cuisine, rating):
    """Generates realistic customer feedback reviews based on cuisine & rating"""
    reviews = [
        {
            "author": "Sarah M.",
            "avatar": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=100&auto=format&fit=crop",
            "rating": 5.0,
            "date": "2 days ago",
            "comment": f"Outstanding experience! The {cuisine} dishes were fresh, flavorful, and served promptly."
        },
        {
            "author": "Alex K.",
            "avatar": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&auto=format&fit=crop",
            "rating": 4.5,
            "date": "1 week ago",
            "comment": f"Warm atmosphere and excellent value for money. Highly recommended for date night!"
        },
        {
            "author": "Daniel T.",
            "avatar": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=100&auto=format&fit=crop",
            "rating": 5.0,
            "date": "2 weeks ago",
            "comment": "Fast service, clean venue, and amazing authentic taste. Will definitely revisit."
        }
    ]
    return reviews

def compute_match_reason(row, method, target_title=""):
    """Computes explicit 'Why Recommended' explanation tag for every venue card"""
    cuisine = str(row.get('cuisine', 'Dining'))
    location = str(row.get('location', 'Area'))
    rating = row.get('rating', 4.5)
    
    if "Content" in method:
        return f"Matched via TF-IDF feature tags ({cuisine} & {location})"
    elif "Collaborative" in method:
        return f"Matched via Diner Rating Matrix (Item-Item KNN)"
    elif "Popularity" in method:
        return f"Top Rated Popular Venue ({rating}★ Avg Diner Score)"
    else:
        return f"Matched via Hybrid Ensemble ({cuisine} & {location})"

# ==========================================
# 3. CORE RECOMMENDATION ENGINES
# ==========================================
def get_popular_recommendations(top_n=9, cuisine_filter="All", location_filter="All", price_filter="All", min_rating=0.0):
    filtered = restaurants_df[restaurants_df['rating'] >= min_rating].copy()
    if cuisine_filter != "All":
        filtered = filtered[filtered['cuisine'].astype(str).str.upper().str.contains(cuisine_filter.upper())]
    if location_filter != "All":
        filtered = filtered[filtered['location'].astype(str).str.upper().str.contains(location_filter.upper())]
    if price_filter != "All":
        filtered = filtered[filtered['price_range'] == price_filter]
    return filtered.sort_values(by=['rating'], ascending=[False]).head(top_n)

def get_content_recommendations(title, top_n=9, cuisine_filter="All", location_filter="All", price_filter="All", min_rating=0.0):
    title_lower = title.lower().strip()
    idx = None
    
    # 1. Exact match
    if title_lower in indices:
        idx = indices[title_lower]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
    else:
        # 2. Substring match in restaurant name
        matches = [i for i, t in enumerate(restaurants_df['name'].str.lower()) if title_lower in t]
        if matches:
            idx = matches[0]
            
    # 3. Multi-Attribute Smart Search: If query is not a venue name, transform query string into TF-IDF vector!
    if idx is not None:
        query_vector = tfv_matrix[idx]
    else:
        query_vector = tfv.transform([title])
        
    sim_scores = list(enumerate(cosine_similarity(query_vector, tfv_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    venue_indices = []
    scores_list = []
    
    for item_idx, score in sim_scores:
        if idx is not None and item_idx == idx:
            continue
        row = restaurants_df.iloc[item_idx]
        if min_rating > 0 and row['rating'] < min_rating:
            continue
        if cuisine_filter != "All" and cuisine_filter.upper() not in str(row['cuisine']).upper():
            continue
        if location_filter != "All" and location_filter.upper() not in str(row['location']).upper():
            continue
        if price_filter != "All" and str(row['price_range']) != price_filter:
            continue
            
        venue_indices.append(item_idx)
        scores_list.append(round(float(score), 4))
        if len(venue_indices) >= top_n:
            break
            
    if not venue_indices:
        return get_popular_recommendations(top_n, cuisine_filter, location_filter, price_filter, min_rating)
        
    result = restaurants_df.iloc[venue_indices].copy()
    result['Match Score'] = [f"{max(60, int(s * 100))}% Match" for s in scores_list]
    return result

def get_collaborative_recommendations(title, top_n=9, cuisine_filter="All", location_filter="All", price_filter="All", min_rating=0.0):
    title_lower = title.lower().strip()
    idx = None
    
    if title_lower in pivot_names_lower:
        idx = pivot_names_lower.index(title_lower)
    else:
        matches = [i for i, name in enumerate(pivot_names_lower) if title_lower in name]
        if matches:
            idx = matches[0]
            
    if idx is None:
        return get_content_recommendations(title, top_n, cuisine_filter, location_filter, price_filter, min_rating)
        
    distances, indices_knn = knn_model.kneighbors(pivot_table.iloc[idx, :].values.reshape(1, -1), n_neighbors=min(50, len(pivot_table)))
    
    rec_list = []
    for i in range(1, len(distances.flatten())):
        item_name = pivot_table.index[indices_knn.flatten()[i]]
        sim = round(1 - float(distances.flatten()[i]), 4)
        
        matches = restaurants_df[restaurants_df['name'] == item_name]
        if not matches.empty:
            row = matches.iloc[0].to_dict()
            if min_rating > 0 and row['rating'] < min_rating:
                continue
            if cuisine_filter != "All" and cuisine_filter.upper() not in str(row['cuisine']).upper():
                continue
            if location_filter != "All" and location_filter.upper() not in str(row['location']).upper():
                continue
            if price_filter != "All" and str(row['price_range']) != price_filter:
                continue
                
            row['Match Score'] = f"{int(sim * 100)}% Diner Sim"
            rec_list.append(row)
            if len(rec_list) >= top_n:
                break
                
    if not rec_list:
        return get_content_recommendations(title, top_n, cuisine_filter, location_filter, price_filter, min_rating)
        
    return pd.DataFrame(rec_list)

def get_hybrid_recommendations(title, top_n=9, cuisine_filter="All", location_filter="All", price_filter="All", min_rating=0.0):
    content_df = get_content_recommendations(title, top_n=top_n*2, cuisine_filter=cuisine_filter, location_filter=location_filter, price_filter=price_filter, min_rating=min_rating)
    collab_df = get_collaborative_recommendations(title, top_n=top_n*2, cuisine_filter=cuisine_filter, location_filter=location_filter, price_filter=price_filter, min_rating=min_rating)
    
    has_content = 'Message' not in content_df.columns and not content_df.empty
    has_collab = 'Message' not in collab_df.columns and not collab_df.empty
    
    if not has_content and not has_collab:
        return get_popular_recommendations(top_n, cuisine_filter, location_filter, price_filter, min_rating)
    if not has_content:
        return collab_df.head(top_n)
    if not has_collab:
        return content_df.head(top_n)
        
    combined = pd.concat([content_df, collab_df]).drop_duplicates(subset=['name']).head(top_n)
    return combined

def process_diner_quiz(occasion, vibe, budget):
    """Smart AI Foodie Concierge Quiz Matching Engine"""
    quiz_query = f"{occasion} {vibe} {budget}"
    sim_scores = list(enumerate(cosine_similarity(tfv.transform([quiz_query]), tfv_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    venue_indices = []
    scores_list = []
    
    target_price = "$" if "($)" in budget else ("$$" if "($$)" in budget else ("$$$" if "($$$)" in budget else None))
    
    for item_idx, score in sim_scores:
        row = restaurants_df.iloc[item_idx]
        if target_price and str(row['price_range']) != target_price:
            continue
        venue_indices.append(item_idx)
        scores_list.append(round(float(score), 4))
        if len(venue_indices) >= 6:
            break
            
    if not venue_indices:
        df = restaurants_df.head(6).copy()
    else:
        df = restaurants_df.iloc[venue_indices].copy()
        
    df['Match Score'] = [f"{max(75, int(s * 100))}% Vibe Sim" for s in scores_list] if scores_list else "Recommended"
    return df

# ==========================================
# 4. FLASK ROUTES & API ENDPOINTS
# ==========================================
@app.route('/')
def home_page():
    return render_template('index.html', 
                           featured_venues=featured_venue_choices,
                           locations=unique_locations,
                           cuisines=unique_cuisines,
                           total_venues=len(restaurants_df),
                           total_ratings=len(ratings_df),
                           total_cuisines=restaurants_df['cuisine'].nunique())

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json or {}
    title = data.get('title', featured_venue_choices[0])
    method = data.get('method', 'Hybrid AI Ensemble')
    cuisine = data.get('cuisine', 'All')
    location = data.get('location', 'All')
    price = data.get('price', 'All')
    min_rating = float(data.get('min_rating', 0.0))
    top_n = int(data.get('top_n', 9))
    
    if method == "Popularity Baseline":
        df = get_popular_recommendations(top_n, cuisine, location, price, min_rating)
    elif "Content" in method:
        df = get_content_recommendations(title, top_n, cuisine, location, price, min_rating)
    elif "Collaborative" in method:
        df = get_collaborative_recommendations(title, top_n, cuisine, location, price, min_rating)
    else:
        df = get_hybrid_recommendations(title, top_n, cuisine, location, price, min_rating)
        
    if df.empty or 'Message' in df.columns:
        df = get_popular_recommendations(top_n, cuisine, location, price, min_rating)
        
    results = df.to_dict(orient='records')
    for item in results:
        item['rm_price'] = compute_rm_price(item)
        item['image_url'] = get_unique_venue_image(item)
        item['match_reason'] = compute_match_reason(item, method, title)
    return jsonify({'status': 'success', 'count': len(results), 'results': results})

@app.route('/api/quiz', methods=['POST'])
def api_quiz():
    data = request.json or {}
    occasion = data.get('occasion', 'Casual Dining & Hangout')
    vibe = data.get('vibe', 'Cozy & Aesthetic Cafe')
    budget = data.get('budget', 'Any Budget')
    
    df = process_diner_quiz(occasion, vibe, budget)
    results = df.to_dict(orient='records')
    for item in results:
        item['rm_price'] = compute_rm_price(item)
        item['image_url'] = get_unique_venue_image(item)
        item['match_reason'] = f"Vibe matched: {occasion} & {vibe}"
    return jsonify({'status': 'success', 'results': results})

@app.route('/api/restaurant/<name>', methods=['GET'])
def api_restaurant_detail(name):
    matches = restaurants_df[restaurants_df['name'].str.lower() == name.lower()]
    if matches.empty:
        matches = restaurants_df[restaurants_df['name'].str.lower().str.contains(name.lower())]
    if matches.empty:
        row = restaurants_df.iloc[0].to_dict()
    else:
        row = matches.iloc[0].to_dict()
    row['rating_food'] = min(5.0, round(float(row.get('rating', 4.5)) + 0.1, 1))
    row['rating_service'] = min(5.0, round(float(row.get('rating', 4.5)) - 0.1, 1))
    row['rating_ambience'] = float(row.get('rating', 4.5))
    row['rating_value'] = min(5.0, round(float(row.get('rating', 4.5)), 1))
    row['rm_price'] = compute_rm_price(row)
    row['image_url'] = get_unique_venue_image(row)
    row['reviews'] = generate_sample_reviews(str(row.get('cuisine', 'Dining')), row.get('rating', 4.5))
    return jsonify({'status': 'success', 'restaurant': row})

if __name__ == '__main__':
    print("Starting Restaurant Recommendation System on http://127.0.0.1:5000...")
    app.run(host='127.0.0.1', port=5000, debug=False)
