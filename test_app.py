import unittest
import json
import pandas as pd
from app import (
    app,
    restaurants_df,
    ratings_df,
    unique_locations,
    unique_cuisines,
    compute_rm_price,
    get_unique_venue_image,
    get_popular_recommendations,
    get_content_recommendations,
    get_collaborative_recommendations,
    get_hybrid_recommendations,
    process_diner_quiz
)

class TestPriceAndImageFunctions(unittest.TestCase):
    def test_compute_rm_price_tiers(self):
        """Test RM price computation across all price tiers"""
        row_single = {'restaurant_id': 1, 'price_range': '$', 'name': 'Test Cafe'}
        row_double = {'restaurant_id': 2, 'price_range': '$$', 'name': 'Test Bistro'}
        row_triple = {'restaurant_id': 3, 'price_range': '$$$', 'name': 'Test Fine Dining'}
        row_quad   = {'restaurant_id': 4, 'price_range': '$$$$', 'name': 'Luxury Steakhouse'}

        price_s = compute_rm_price(row_single)
        price_d = compute_rm_price(row_double)
        price_t = compute_rm_price(row_triple)
        price_q = compute_rm_price(row_quad)

        self.assertTrue(price_s.startswith('RM '))
        self.assertTrue(price_d.startswith('RM '))
        self.assertTrue(price_t.startswith('RM '))
        self.assertTrue(price_q.startswith('RM '))

    def test_compute_rm_price_invalid_row(self):
        """Test fallback when restaurant_id is invalid or missing"""
        row_empty = {}
        price = compute_rm_price(row_empty)
        self.assertTrue(price.startswith('RM '))

    def test_get_unique_venue_image_cuisines(self):
        """Test unique venue image assignment for various cuisines"""
        cuisines = ['Japanese', 'Italian', 'Cafe & Bakery', 'Western & Grill', 'Malaysian', 'Unknown']
        for idx, cuis in enumerate(cuisines):
            row = {'restaurant_id': idx + 1, 'cuisine': cuis, 'name': f'Venue {idx}'}
            img_url = get_unique_venue_image(row)
            self.assertTrue(img_url.startswith('https://images.unsplash.com/'))

    def test_get_unique_venue_image_uniqueness(self):
        """Test that distinct venue IDs receive distinct image URLs"""
        row1 = {'restaurant_id': 1, 'cuisine': 'Japanese', 'name': 'Sushi Master'}
        row2 = {'restaurant_id': 2, 'cuisine': 'Japanese', 'name': 'Tokyo Ramen'}
        img1 = get_unique_venue_image(row1)
        img2 = get_unique_venue_image(row2)
        self.assertNotEqual(img1, img2)

class TestRecommendationEngines(unittest.TestCase):
    def test_popular_recommendations(self):
        """Test Popularity-Based recommendation engine"""
        df = get_popular_recommendations(top_n=5)
        self.assertEqual(len(df), 5)
        self.assertIn('rating', df.columns)

    def test_popular_recommendations_filtered(self):
        """Test Popularity engine with location & cuisine filtering"""
        loc = unique_locations[0]
        df = get_popular_recommendations(top_n=5, location_filter=loc)
        self.assertLessEqual(len(df), 5)
        if not df.empty:
            for val in df['location']:
                self.assertIn(loc.upper(), str(val).upper())

    def test_content_recommendations_valid_title(self):
        """Test Content-Based TF-IDF recommendation engine for valid title"""
        sample_title = restaurants_df.iloc[0]['name']
        df = get_content_recommendations(sample_title, top_n=5)
        self.assertNotIn('Message', df.columns)
        self.assertLessEqual(len(df), 5)
        self.assertIn('Match Score', df.columns)

    def test_content_recommendations_nonexistent_title(self):
        """Test Content-Based engine handling of nonexistent title via Multi-Attribute Smart Search"""
        df = get_content_recommendations("NonExistentRestaurant9999", top_n=5)
        self.assertGreater(len(df), 0)
        self.assertIn('name', df.columns)

    def test_collaborative_recommendations_valid_title(self):
        """Test Collaborative KNN recommendation engine for valid title"""
        sample_title = restaurants_df.iloc[0]['name']
        df = get_collaborative_recommendations(sample_title, top_n=5)
        self.assertNotIn('Message', df.columns)
        self.assertLessEqual(len(df), 5)
        self.assertIn('Match Score', df.columns)

    def test_collaborative_recommendations_nonexistent_title(self):
        """Test Collaborative engine handling of nonexistent title via Multi-Attribute Smart Search"""
        df = get_collaborative_recommendations("NonExistentRestaurant9999", top_n=5)
        self.assertGreater(len(df), 0)
        self.assertIn('name', df.columns)

    def test_hybrid_recommendations(self):
        """Test Hybrid AI Ensemble recommendation engine"""
        sample_title = restaurants_df.iloc[0]['name']
        df = get_hybrid_recommendations(sample_title, top_n=5)
        self.assertNotIn('Message', df.columns)
        self.assertLessEqual(len(df), 5)

    def test_process_diner_quiz(self):
        """Test 3-Step Foodie Craving Quiz matching engine"""
        df = process_diner_quiz("Casual Dining & Hangout", "Cozy & Aesthetic Cafe", "Budget Friendly ($)")
        self.assertEqual(len(df), 6)
        self.assertIn('Match Score', df.columns)

class TestFlaskAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_home_page(self):
        """Test GET / returns 200 OK and renders main HTML template"""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Restaurant Recommendation System', res.data)

    def test_api_recommend_hybrid(self):
        """Test POST /api/recommend with Hybrid AI Ensemble method"""
        sample_title = restaurants_df.iloc[0]['name']
        payload = {
            'title': sample_title,
            'method': 'Hybrid AI Ensemble',
            'cuisine': 'All',
            'location': 'All',
            'price': 'All',
            'top_n': 5
        }
        res = self.client.post('/api/recommend', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['results']), 5)
        self.assertIn('rm_price', data['results'][0])
        self.assertIn('image_url', data['results'][0])

    def test_api_recommend_content(self):
        """Test POST /api/recommend with Content-Based Filtering"""
        sample_title = restaurants_df.iloc[0]['name']
        payload = {
            'title': sample_title,
            'method': 'Content-Based Filtering',
            'top_n': 3
        }
        res = self.client.post('/api/recommend', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')

    def test_api_recommend_collaborative(self):
        """Test POST /api/recommend with Collaborative Filtering"""
        sample_title = restaurants_df.iloc[0]['name']
        payload = {
            'title': sample_title,
            'method': 'Collaborative Filtering',
            'top_n': 3
        }
        res = self.client.post('/api/recommend', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')

    def test_api_recommend_popularity(self):
        """Test POST /api/recommend with Popularity Baseline"""
        payload = {
            'method': 'Popularity Baseline',
            'top_n': 4
        }
        res = self.client.post('/api/recommend', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['results']), 4)

    def test_api_recommend_location_filtering(self):
        """Test POST /api/recommend with location filtering"""
        loc = unique_locations[0]
        payload = {
            'method': 'Popularity Baseline',
            'location': loc,
            'top_n': 3
        }
        res = self.client.post('/api/recommend', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')

    def test_api_recommend_invalid_title(self):
        """Test POST /api/recommend graceful fallback handling for free-text search queries"""
        payload = {
            'title': 'NonExistentVenueXYZ9999',
            'method': 'Content-Based Filtering'
        }
        res = self.client.post('/api/recommend', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertGreater(len(data['results']), 0)

    def test_api_quiz_endpoint(self):
        """Test POST /api/quiz endpoint returning matches"""
        payload = {
            'occasion': 'Romantic Date Night',
            'vibe': 'Fine Dining & Upscale',
            'budget': 'Fine Dining ($$$)'
        }
        res = self.client.post('/api/quiz', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['results']), 6)
        self.assertIn('rm_price', data['results'][0])

    def test_api_restaurant_detail_exact(self):
        """Test GET /api/restaurant/<name> for exact venue match"""
        sample_name = restaurants_df.iloc[0]['name']
        res = self.client.get(f'/api/restaurant/{sample_name}')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['restaurant']['name'], sample_name)
        self.assertIn('rating_food', data['restaurant'])
        self.assertIn('rm_price', data['restaurant'])

    def test_api_restaurant_detail_substring(self):
        """Test GET /api/restaurant/<name> for partial substring match"""
        first_word = restaurants_df.iloc[0]['name'].split()[0]
        res = self.client.get(f'/api/restaurant/{first_word}')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('restaurant', data)

    def test_api_restaurant_detail_fallback(self):
        """Test GET /api/restaurant/<name> fallback for non-matching query"""
        res = self.client.get('/api/restaurant/UnknownNonexistentQuery1234')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('restaurant', data)

if __name__ == '__main__':
    unittest.main()
