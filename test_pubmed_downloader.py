"""
Unit Tests for PubMed Downloader
Tests all functions in isolation with mocked dependencies
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
from pymongo.errors import DuplicateKeyError

# Import functions to test
from pubmed_downloader import (
    connect_mongodb,
    search_pubmed,
    get_pmc_id,
    fetch_full_text_from_pmc,
    fetch_paper_details,
    download_to_mongodb
)


# ============================================================
# TEST: MongoDB Connection
# ============================================================

class TestMongoDBConnection:
    """Test MongoDB connection functionality"""
    
    @patch('pubmed_downloader.pymongo.MongoClient')
    def test_successful_connection(self, mock_mongo_client):
        """Test successful MongoDB connection"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_mongo_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.count_documents.return_value = 0
        
        result = connect_mongodb()
        
        mock_client.server_info.assert_called_once()
        mock_collection.create_index.assert_called_once_with("pmid", unique=True)
        assert result == mock_collection
    
    @patch('pubmed_downloader.pymongo.MongoClient')
    def test_connection_failure(self, mock_mongo_client):
        """Test MongoDB connection failure exits"""
        mock_client = MagicMock()
        mock_client.server_info.side_effect = Exception("Connection refused")
        mock_mongo_client.return_value = mock_client
        
        with pytest.raises(SystemExit) as exc_info:
            connect_mongodb()
        
        assert exc_info.value.code == 1


# ============================================================
# TEST: PubMed Search
# ============================================================

class TestPubMedSearch:
    """Test PubMed search functionality"""
    
    @patch('pubmed_downloader.Entrez.esearch')
    @patch('pubmed_downloader.Entrez.read')
    def test_successful_search(self, mock_read, mock_esearch):
        """Test successful PubMed search"""
        mock_handle = MagicMock()
        mock_esearch.return_value = mock_handle
        mock_read.return_value = {'IdList': ['12345', '67890', '11111']}
        
        result = search_pubmed("diabetes", max_results=10)
        
        mock_esearch.assert_called_once()
        mock_read.assert_called_once_with(mock_handle)
        mock_handle.close.assert_called_once()
        assert result == ['12345', '67890', '11111']
        assert len(result) == 3
    
    @patch('pubmed_downloader.Entrez.esearch')
    def test_search_with_exception(self, mock_esearch):
        """Test PubMed search handles exceptions"""
        mock_esearch.side_effect = Exception("API Error")
        
        result = search_pubmed("test query")
        
        assert result == []
    
    @patch('pubmed_downloader.Entrez.esearch')
    @patch('pubmed_downloader.Entrez.read')
    def test_empty_search_results(self, mock_read, mock_esearch):
        """Test search with no results"""
        mock_handle = MagicMock()
        mock_esearch.return_value = mock_handle
        mock_read.return_value = {'IdList': []}
        
        result = search_pubmed("nonexistent query")
        
        assert result == []


# ============================================================
# TEST: PMC ID Retrieval
# ============================================================

class TestPMCIDRetrieval:
    """Test PMC ID retrieval from PMID"""
    
    @patch('pubmed_downloader.Entrez.elink')
    @patch('pubmed_downloader.Entrez.read')
    def test_get_pmc_id_success(self, mock_read, mock_elink):
        """Test successful PMC ID retrieval"""
        mock_handle = MagicMock()
        mock_elink.return_value = mock_handle
        mock_read.return_value = [
            {'LinkSetDb': [{'Link': [{'Id': 'PMC12345'}]}]}
        ]
        
        result = get_pmc_id('12345')
        
        mock_elink.assert_called_once()
        mock_handle.close.assert_called_once()
        assert result == 'PMC12345'
    
    @patch('pubmed_downloader.Entrez.elink')
    @patch('pubmed_downloader.Entrez.read')
    def test_get_pmc_id_not_found(self, mock_read, mock_elink):
        """Test PMC ID not available"""
        mock_handle = MagicMock()
        mock_elink.return_value = mock_handle
        mock_read.return_value = [{'LinkSetDb': []}]
        
        result = get_pmc_id('12345')
        
        assert result is None
    
    @patch('pubmed_downloader.Entrez.elink')
    def test_get_pmc_id_exception(self, mock_elink):
        """Test PMC ID retrieval handles exceptions"""
        mock_elink.side_effect = Exception("API Error")
        
        result = get_pmc_id('12345')
        
        assert result is None


# ============================================================
# TEST: Full Text Retrieval
# ============================================================

class TestFullTextRetrieval:
    """Test full text retrieval from PMC"""
    
    @patch('pubmed_downloader.Entrez.efetch')
    def test_fetch_full_text_success(self, mock_efetch):
        """Test successful full text retrieval"""
        mock_handle = MagicMock()
        mock_efetch.return_value = mock_handle
        long_xml = b"<article>" + b"x" * 60000 + b"</article>"
        mock_handle.read.return_value = long_xml
        
        result = fetch_full_text_from_pmc('PMC12345')
        
        mock_efetch.assert_called_once()
        mock_handle.close.assert_called_once()
        assert result is not None
        assert len(result) == 50000  # Should be truncated
    
    @patch('pubmed_downloader.Entrez.efetch')
    def test_fetch_full_text_short_content(self, mock_efetch):
        """Test full text with short content returns None"""
        mock_handle = MagicMock()
        mock_efetch.return_value = mock_handle
        mock_handle.read.return_value = b"short"
        
        result = fetch_full_text_from_pmc('PMC12345')
        
        assert result is None
    
    @patch('pubmed_downloader.Entrez.efetch')
    def test_fetch_full_text_exception(self, mock_efetch):
        """Test full text retrieval handles exceptions"""
        mock_efetch.side_effect = Exception("API Error")
        
        result = fetch_full_text_from_pmc('PMC12345')
        
        assert result is None


# ============================================================
# TEST: Paper Details Fetching
# ============================================================

class TestPaperDetailsFetching:
    """Test fetching paper details"""
    
    @patch('pubmed_downloader.get_pmc_id')
    @patch('pubmed_downloader.fetch_full_text_from_pmc')
    @patch('pubmed_downloader.Entrez.efetch')
    @patch('pubmed_downloader.Entrez.read')
    def test_fetch_paper_details_success(self, mock_read, mock_efetch, 
                                         mock_fetch_full, mock_get_pmc):
        """Test successful paper details fetching"""
        # Setup mocks
        mock_handle = MagicMock()
        mock_efetch.return_value = mock_handle
        
        mock_read.return_value = {
            'PubmedArticle': [{
                'MedlineCitation': {
                    'PMID': '12345',
                    'Article': {
                        'ArticleTitle': 'Test Paper Title',
                        'Abstract': {'AbstractText': ['Test abstract text']},
                        'Journal': {
                            'Title': 'Test Journal',
                            'JournalIssue': {'PubDate': {'Year': '2024'}}
                        },
                        'AuthorList': [
                            {'LastName': 'Smith', 'Initials': 'J'}
                        ]
                    }
                }
            }]
        }
        
        mock_get_pmc.return_value = 'PMC12345'
        mock_fetch_full.return_value = 'Full text content'
        
        result = fetch_paper_details('12345')
        
        assert result is not None
        assert result['pmid'] == '12345'
        assert result['title'] == 'Test Paper Title'
        assert result['abstract'] == 'Test abstract text'
        assert result['journal'] == 'Test Journal'
        assert result['year'] == 2024
        assert result['authors'] == 'Smith J'
        assert result['full_text'] == 'Full text content'
        assert result['metadata']['has_full_text'] is True
    
    @patch('pubmed_downloader.Entrez.efetch')
    @patch('pubmed_downloader.Entrez.read')
    def test_fetch_paper_details_no_article(self, mock_read, mock_efetch):
        """Test paper details when article not found"""
        mock_handle = MagicMock()
        mock_efetch.return_value = mock_handle
        mock_read.return_value = {}
        
        result = fetch_paper_details('12345')
        
        assert result is None
    
    @patch('pubmed_downloader.Entrez.efetch')
    def test_fetch_paper_details_exception(self, mock_efetch):
        """Test paper details handles exceptions"""
        mock_efetch.side_effect = Exception("API Error")
        
        result = fetch_paper_details('12345')
        
        assert result is None


# ============================================================
# TEST: Download to MongoDB
# ============================================================

class TestDownloadToMongoDB:
    """Test downloading papers to MongoDB"""
    
    @patch('pubmed_downloader.search_pubmed')
    @patch('pubmed_downloader.fetch_paper_details')
    @patch('pubmed_downloader.time.sleep')
    def test_download_to_mongodb_success(self, mock_sleep, mock_fetch, mock_search):
        """Test successful download to MongoDB"""
        # Setup mocks
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None  # No duplicates
        
        mock_search.return_value = ['12345', '67890']
        mock_fetch.return_value = {
            'pmid': '12345',
            'title': 'Test Paper',
            'abstract': 'Test abstract',
            'full_text': None,
            'metadata': {}
        }
        
        result = download_to_mongodb(
            mock_collection, 
            'diabetes', 
            'endocrinology', 
            max_results=10
        )
        
        assert mock_search.called
        assert mock_fetch.call_count == 2
        assert mock_collection.insert_one.call_count == 2
        assert result == 2
    
    @patch('pubmed_downloader.search_pubmed')
    def test_download_to_mongodb_no_results(self, mock_search):
        """Test download with no search results"""
        mock_collection = MagicMock()
        mock_search.return_value = []
        
        result = download_to_mongodb(
            mock_collection,
            'nonexistent query',
            'general'
        )
        
        assert result == 0
        assert not mock_collection.insert_one.called
    
    @patch('pubmed_downloader.search_pubmed')
    @patch('pubmed_downloader.fetch_paper_details')
    @patch('pubmed_downloader.time.sleep')
    def test_download_handles_duplicates(self, mock_sleep, mock_fetch, mock_search):
        """Test download skips duplicate papers"""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = {'pmid': '12345'}  # Duplicate
        
        mock_search.return_value = ['12345', '67890']
        
        result = download_to_mongodb(
            mock_collection,
            'test query',
            'general'
        )
        
        # Should skip duplicates, no inserts
        assert mock_collection.insert_one.call_count == 0
    
    @patch('pubmed_downloader.search_pubmed')
    @patch('pubmed_downloader.fetch_paper_details')
    @patch('pubmed_downloader.time.sleep')
    def test_download_handles_fetch_errors(self, mock_sleep, mock_fetch, mock_search):
        """Test download handles paper fetch errors"""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        
        mock_search.return_value = ['12345', '67890']
        mock_fetch.return_value = None  # Fetch failed
        
        result = download_to_mongodb(
            mock_collection,
            'test query',
            'general'
        )
        
        assert result == 0
        assert mock_collection.insert_one.call_count == 0


# ============================================================
# INTEGRATION-LIKE TEST
# ============================================================

class TestFullDownloadFlow:
    """Test complete download flow with all components"""
    
    @patch('pubmed_downloader.pymongo.MongoClient')
    @patch('pubmed_downloader.search_pubmed')
    @patch('pubmed_downloader.fetch_paper_details')
    @patch('pubmed_downloader.time.sleep')
    def test_complete_flow(self, mock_sleep, mock_fetch, mock_search, mock_mongo):
        """Test complete download flow"""
        # Setup MongoDB mock
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_mongo.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.count_documents.return_value = 0
        mock_collection.find_one.return_value = None
        
        # Setup PubMed mocks
        mock_search.return_value = ['12345']
        mock_fetch.return_value = {
            'pmid': '12345',
            'title': 'Test Paper',
            'abstract': 'Test abstract',
            'journal': 'Test Journal',
            'year': 2024,
            'authors': 'Smith J',
            'full_text': 'Full text',
            'metadata': {'has_full_text': True}
        }
        
        # Execute
        collection = connect_mongodb()
        result = download_to_mongodb(collection, 'diabetes', 'endocrinology')
        
        # Verify
        assert mock_search.called
        assert mock_fetch.called
        assert mock_collection.insert_one.called
        assert result == 1


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])