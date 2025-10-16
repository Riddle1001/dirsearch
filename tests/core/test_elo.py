# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from lib.core.elo import EloManager


class TestEloManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_wordlist = "test_wordlist.txt"
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock options
        self.mock_options = {
            "elo_directory": self.temp_dir,
            "elo_initial": 1000,
            "elo_hit_gain": 100,
            "elo_miss_penalty": 25,
            "elo_time_decay_per_scan": 0.01,
            "elo_suspicious_threshold": 0.8,
            "elo_rate_limit_threshold": 5,
        }
        
    def tearDown(self):
        """Clean up test files"""
        # Remove test ELO file if it exists
        elo_file = os.path.join(self.temp_dir, f"{self.test_wordlist}.elo.json")
        if os.path.exists(elo_file):
            os.remove(elo_file)
        os.rmdir(self.temp_dir)
    
    @patch('lib.core.elo.options', new_callable=dict)
    def test_initialization(self, mock_opts):
        """Test ELO manager initialization"""
        mock_opts.update(self.mock_options)
        
        elo_manager = EloManager(self.test_wordlist)
        
        self.assertIsNotNone(elo_manager)
        self.assertEqual(elo_manager.wordlist_name, self.test_wordlist)
        self.assertEqual(elo_manager.metadata["total_scans"], 0)
    
    @patch('lib.core.elo.options', new_callable=dict)
    def test_initialize_word(self, mock_opts):
        """Test initializing a new word"""
        mock_opts.update(self.mock_options)
        
        elo_manager = EloManager(self.test_wordlist)
        elo_manager.initialize_word("admin")
        
        self.assertIn("admin", elo_manager.words)
        self.assertEqual(elo_manager.words["admin"]["elo"], 1000)
        self.assertEqual(elo_manager.words["admin"]["hits"], 0)
        self.assertEqual(elo_manager.words["admin"]["misses"], 0)
    
    @patch('lib.core.elo.options', new_callable=dict)
    def test_record_result(self, mock_opts):
        """Test recording scan results"""
        mock_opts.update(self.mock_options)
        
        elo_manager = EloManager(self.test_wordlist)
        elo_manager.record_result("admin", 200, True)
        elo_manager.record_result("login", 404, False)
        
        self.assertEqual(elo_manager.total_requests, 2)
        self.assertEqual(elo_manager.status_code_counts[200], 1)
        self.assertEqual(elo_manager.status_code_counts[404], 1)
    
    @patch('lib.core.elo.options', new_callable=dict)
    def test_sort_wordlist(self, mock_opts):
        """Test wordlist sorting by ELO"""
        mock_opts.update(self.mock_options)
        
        elo_manager = EloManager(self.test_wordlist)
        
        # Initialize words with different ELO scores
        elo_manager.words["low"] = {"elo": 900, "hits": 0, "misses": 5}
        elo_manager.words["high"] = {"elo": 1200, "hits": 5, "misses": 0}
        elo_manager.words["medium"] = {"elo": 1000, "hits": 2, "misses": 2}
        
        wordlist = ["low", "medium", "high"]
        sorted_list = elo_manager.sort_wordlist(wordlist)
        
        self.assertEqual(sorted_list, ["high", "medium", "low"])
    
    @patch('lib.core.elo.options', new_callable=dict)
    def test_suspicious_scan_detection(self, mock_opts):
        """Test detection of suspicious scans"""
        mock_opts.update(self.mock_options)
        
        elo_manager = EloManager(self.test_wordlist)
        
        # Record many 403s (suspicious)
        for i in range(80):
            elo_manager.record_result(f"word{i}", 403, False)
        
        # Record a few 200s (legitimate)
        for i in range(20):
            elo_manager.record_result(f"legit{i}", 200, True)
        
        self.assertTrue(elo_manager._is_scan_suspicious())
    
    @patch('lib.core.elo.options', new_callable=dict)
    def test_rate_limit_detection(self, mock_opts):
        """Test detection of rate limiting"""
        mock_opts.update(self.mock_options)
        
        elo_manager = EloManager(self.test_wordlist)
        
        # Record multiple 429s
        for i in range(5):
            elo_manager.record_result(f"word{i}", 429, False)
        
        self.assertTrue(elo_manager._is_scan_suspicious())
    
    @patch('lib.core.elo.options', new_callable=dict)
    def test_save_and_load(self, mock_opts):
        """Test saving and loading ELO data"""
        mock_opts.update(self.mock_options)
        
        # Create and save
        elo_manager = EloManager(self.test_wordlist)
        elo_manager.initialize_word("admin")
        elo_manager.words["admin"]["elo"] = 1100
        elo_manager.save()
        
        # Load in new instance
        elo_manager2 = EloManager(self.test_wordlist)
        
        self.assertIn("admin", elo_manager2.words)
        self.assertEqual(elo_manager2.words["admin"]["elo"], 1100)


if __name__ == "__main__":
    unittest.main()


