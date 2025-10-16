# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from lib.core.ratings import RatingManager, wilson_score_lower_bound, get_z_score


class TestWilsonScore(unittest.TestCase):
    def test_wilson_score_zero_observations(self):
        """Test Wilson score with zero observations"""
        score = wilson_score_lower_bound(0, 0)
        self.assertEqual(score, 0.0)
    
    def test_wilson_score_sparse_data(self):
        """Test Wilson score is appropriately cautious with sparse data"""
        # 1 hit, 0 misses should give very low Wilson score
        score = wilson_score_lower_bound(1, 1, 0.95)
        # Should be much lower than the naive rate of 100%
        self.assertLess(score, 0.5)
        self.assertGreater(score, 0.0)
    
    def test_wilson_score_convergence(self):
        """Test Wilson score converges to true rate with many observations"""
        # 15% hit rate with 1000 observations
        score = wilson_score_lower_bound(150, 1000, 0.95)
        # Should be close to 15% but slightly lower (conservative)
        self.assertGreater(score, 0.12)
        self.assertLess(score, 0.15)
    
    def test_wilson_score_confidence_levels(self):
        """Test different confidence levels"""
        hits, total = 50, 100
        score_90 = wilson_score_lower_bound(hits, total, 0.90)
        score_95 = wilson_score_lower_bound(hits, total, 0.95)
        score_99 = wilson_score_lower_bound(hits, total, 0.99)
        
        # Higher confidence = more conservative (lower bound)
        self.assertGreater(score_90, score_95)
        self.assertGreater(score_95, score_99)


class TestZScore(unittest.TestCase):
    def test_exact_z_score_lookup(self):
        """Test exact z-score lookup"""
        self.assertEqual(get_z_score(0.95), 1.96)
        self.assertEqual(get_z_score(0.99), 2.576)
    
    def test_closest_z_score_lookup(self):
        """Test closest z-score for non-exact values"""
        # 0.94 should round to 0.95
        z_score = get_z_score(0.94)
        self.assertEqual(z_score, 1.96)


class TestRatingManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_wordlist = "test_wordlist.txt"
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock options
        self.mock_options = {
            "ratings_directory": self.temp_dir,
            "prior_alpha": 5,
            "prior_beta": 95,
            "confidence_level": 0.95,
            "time_decay_enabled": False,
            "suspicious_threshold": 0.8,
            "rate_limit_threshold": 5,
        }
        
    def tearDown(self):
        """Clean up test files"""
        # Remove test rating file if it exists
        rating_file = os.path.join(self.temp_dir, f"{self.test_wordlist}.elo.json")
        if os.path.exists(rating_file):
            os.remove(rating_file)
        os.rmdir(self.temp_dir)
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_initialization(self, mock_opts):
        """Test rating manager initialization"""
        mock_opts.update(self.mock_options)
        
        rating_manager = RatingManager(self.test_wordlist)
        
        self.assertIsNotNone(rating_manager)
        self.assertEqual(rating_manager.wordlist_name, self.test_wordlist)
        self.assertEqual(rating_manager.metadata["total_scans"], 0)
        self.assertEqual(rating_manager.metadata["version"], 2)
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_initialize_word(self, mock_opts):
        """Test initializing a new word"""
        mock_opts.update(self.mock_options)
        
        rating_manager = RatingManager(self.test_wordlist)
        rating_manager.initialize_word("admin")
        
        self.assertIn("admin", rating_manager.words)
        self.assertEqual(rating_manager.words["admin"]["hits"], 0)
        self.assertEqual(rating_manager.words["admin"]["misses"], 0)
        # Scores should NOT be stored - only calculated on-demand
        self.assertNotIn("score", rating_manager.words["admin"])
        self.assertNotIn("wilson_score", rating_manager.words["admin"])
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_calculate_scores(self, mock_opts):
        """Test on-demand score calculation"""
        mock_opts.update(self.mock_options)
        
        rating_manager = RatingManager(self.test_wordlist)
        
        # Test sparse data (1 hit, 0 misses)
        bayesian, wilson = rating_manager.calculate_scores(1, 0)
        
        # Bayesian should be pulled down by prior
        self.assertGreater(bayesian, 0.05)
        self.assertLess(bayesian, 0.1)
        
        # Wilson should be calculated on posterior and be conservative
        self.assertGreater(wilson, 0.0)
        self.assertLess(wilson, 0.5)
        
        # Wilson on posterior should be <= Bayesian mean
        self.assertLessEqual(wilson, bayesian * 1.1)  # Some tolerance for numerical differences
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_high_miss_rate_convergence(self, mock_opts):
        """Test system handles 99% miss rate correctly"""
        mock_opts.update(self.mock_options)
        
        rating_manager = RatingManager(self.test_wordlist)
        
        # Realistic scenario: 1% hit rate (10 hits, 990 misses)
        bayesian, wilson = rating_manager.calculate_scores(10, 990)
        
        # Should converge close to 1% (15 total / 1005 total ≈ 0.015)
        self.assertGreater(bayesian, 0.01)
        self.assertLess(bayesian, 0.02)
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_record_result(self, mock_opts):
        """Test recording scan results"""
        mock_opts.update(self.mock_options)
        
        rating_manager = RatingManager(self.test_wordlist)
        rating_manager.record_result("admin", 200, True)
        rating_manager.record_result("login", 404, False)
        
        self.assertEqual(rating_manager.total_requests, 2)
        self.assertEqual(rating_manager.status_code_counts[200], 1)
        self.assertEqual(rating_manager.status_code_counts[404], 1)
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_sort_wordlist(self, mock_opts):
        """Test wordlist sorting by Wilson scores"""
        mock_opts.update(self.mock_options)
        
        rating_manager = RatingManager(self.test_wordlist)
        
        # Initialize words with different hit/miss patterns
        # High confidence, high rate
        rating_manager.words["high"] = {"hits": 100, "misses": 400}  # 20% rate, 500 obs
        # Low confidence, high rate
        rating_manager.words["medium"] = {"hits": 2, "misses": 8}    # 20% rate, 10 obs
        # High confidence, low rate
        rating_manager.words["low"] = {"hits": 10, "misses": 490}    # 2% rate, 500 obs
        
        wordlist = ["low", "medium", "high"]
        sorted_list = rating_manager.sort_wordlist(wordlist)
        
        # Should be sorted by Wilson score (descending)
        # "high" should be first (high confidence, high rate)
        # "low" should be last (low rate despite high confidence)
        self.assertEqual(sorted_list[0], "high")
        self.assertEqual(sorted_list[-1], "low")
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_suspicious_scan_detection(self, mock_opts):
        """Test detection of suspicious scans"""
        mock_opts.update(self.mock_options)
        
        rating_manager = RatingManager(self.test_wordlist)
        
        # Record many 403s (suspicious)
        for i in range(80):
            rating_manager.record_result(f"word{i}", 403, False)
        
        # Record a few 200s (legitimate)
        for i in range(20):
            rating_manager.record_result(f"legit{i}", 200, True)
        
        self.assertTrue(rating_manager._is_scan_suspicious())
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_rate_limit_detection(self, mock_opts):
        """Test detection of rate limiting"""
        mock_opts.update(self.mock_options)
        
        rating_manager = RatingManager(self.test_wordlist)
        
        # Record multiple 429s
        for i in range(5):
            rating_manager.record_result(f"word{i}", 429, False)
        
        self.assertTrue(rating_manager._is_scan_suspicious())
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_save_and_load(self, mock_opts):
        """Test saving and loading rating data"""
        mock_opts.update(self.mock_options)
        
        # Create and save
        rating_manager = RatingManager(self.test_wordlist)
        rating_manager.initialize_word("admin")
        rating_manager.words["admin"]["hits"] = 10
        rating_manager.words["admin"]["misses"] = 90
        rating_manager.save()
        
        # Load in new instance
        rating_manager2 = RatingManager(self.test_wordlist)
        
        self.assertIn("admin", rating_manager2.words)
        self.assertEqual(rating_manager2.words["admin"]["hits"], 10)
        self.assertEqual(rating_manager2.words["admin"]["misses"], 90)
        # Scores should NOT be stored, only hits/misses
        self.assertNotIn("score", rating_manager2.words["admin"])
        self.assertNotIn("wilson_score", rating_manager2.words["admin"])
        
        # But scores should be calculable on-demand
        bayesian, wilson = rating_manager2.calculate_scores(10, 90)
        self.assertGreater(bayesian, 0.0)
        self.assertGreater(wilson, 0.0)
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_backward_compatibility_elo_migration(self, mock_opts):
        """Test loading old ELO format and migrating to Bayesian"""
        mock_opts.update(self.mock_options)
        
        # Create old ELO format file
        old_data = {
            "metadata": {"total_scans": 5, "last_scan": "2024-01-01T12:00:00"},
            "words": {
                "admin": {"elo": 1200, "hits": 10, "misses": 5},
                "login": {"elo": 900, "hits": 2, "misses": 20}
            }
        }
        
        rating_file = os.path.join(self.temp_dir, f"{self.test_wordlist}.elo.json")
        with open(rating_file, 'w') as f:
            json.dump(old_data, f)
        
        # Load with RatingManager - should auto-migrate
        rating_manager = RatingManager(self.test_wordlist)
        
        # Should have migrated to version 2
        self.assertEqual(rating_manager.metadata["version"], 2)
        
        # Should have preserved hits/misses
        self.assertEqual(rating_manager.words["admin"]["hits"], 10)
        self.assertEqual(rating_manager.words["admin"]["misses"], 5)
        
        # Should have calculated Bayesian scores (no ELO anymore)
        self.assertNotIn("elo", rating_manager.words["admin"])
        # Scores should NOT be stored
        self.assertNotIn("score", rating_manager.words["admin"])
        self.assertNotIn("wilson_score", rating_manager.words["admin"])
        
        # But scores should be calculable on-demand
        bayesian, wilson = rating_manager.calculate_scores(10, 5)
        self.assertGreater(bayesian, 0.0)
        self.assertGreater(wilson, 0.0)
    
    @patch('lib.core.ratings.options', new_callable=dict)
    def test_time_decay(self, mock_opts):
        """Test time decay functionality"""
        mock_opts.update(self.mock_options)
        mock_opts["time_decay_enabled"] = True
        mock_opts["time_decay_rate"] = 0.1  # 10% per month
        mock_opts["time_decay_threshold_days"] = 30
        
        rating_manager = RatingManager(self.test_wordlist)
        
        # Word with 100 hits, 900 misses
        word_data = {"hits": 100, "misses": 900}
        
        # Apply decay for 60 days (1 month past threshold)
        decayed = rating_manager.apply_time_decay_to_word(word_data, 60)
        
        # Should decay by 10% = 0.9x
        # 100 * 0.9 = 90, 900 * 0.9 = 810
        self.assertLess(decayed["hits"], 100)
        self.assertLess(decayed["misses"], 900)
        self.assertGreater(decayed["hits"], 85)
        self.assertGreater(decayed["misses"], 800)


if __name__ == "__main__":
    unittest.main()

