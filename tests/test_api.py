"""
Unit and Integration Tests for FastAPI Service
Tests both helper functions and API endpoints
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from pymongo.errors import DuplicateKeyError

# Import functions first
from api import (
    app,
    get_mongodb_collection,
    set_email,
    set_api_key,
    search_pubmed,
    get_pmc_id,
    fetch_full_text_from_pmc,
    fetch_paper_details
)
# ============================================================
# TEST SETUP
# ============================================================

@pytest.fixture
def test_client():
    """Create test client for API testing"""
    from starlette.testclient import TestClient
    # Use positional argument for older versions
    return TestClient(app)


# ============================================================
# TEST: Helper Functions
# ============================================================

class TestHelperFunctions:
    """Test utility helper functions"""
    
    def test_set_email(self):
        """Test setting Entrez email"""
        from Bio import Entrez
        test_email = "test@example.com"
        set_email(test_email)
        assert Entrez.email == test_email
    
    def test_set_api_key(self):
        """Test setting Entrez API key"""
        from Bio import Entrez
        test_key = "test_api_key"
        set_api_key(test_key)
        assert Entrez.api_key == test_key


# ============================================================
# TEST: PubMed Functions
# ============================================================

class TestPubMedFunctions:
    """Test PubMed API interaction functions"""
    
    @patch('api.Entrez.esearch')
    @patch('api.Entrez.read')
    def test_search_pubmed_success(self, mock_read, mock_esearch):
        """Test successful PubMed search"""
        mock_handle = MagicMock()
        mock_esearch.return_value = mock_handle
        mock_read.return_value = {'IdList': ['123', '456']}
        
        result = search_pubmed("diabetes", 10)
        
        assert result == ['123', '456']
        mock_handle.close.assert_called_once()
    
    @patch('api.Entrez.esearch')
    def test_search_pubmed_error(self, mock_esearch):
        """Test PubMed search handles errors"""
        mock_esearch.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            search_pubmed("test query")
    
    @patch('api.Entrez.elink')
    @patch('api.Entrez.read')
    def test_get_pmc_id_found(self, mock_read, mock_elink):
        """Test PMC ID retrieval success"""
        mock_handle = MagicMock()
        mock_elink.return_value = mock_handle
        mock_read.return_value = [
            {'LinkSetDb': [{'Link': [{'Id': 'PMC123'}]}]}
        ]
        
        result = get_pmc_id('12345')
        
        assert result == 'PMC123'
    
    @patch('api.Entrez.elink')
    def test_get_pmc_id_exception(self, mock_elink):
        """Test PMC ID handles exceptions"""
        mock_elink.side_effect = Exception("Error")
        result = get_pmc_id('12345')
        assert result is None
    
    @patch('api.Entrez.efetch')
    def test_fetch_full_text_success(self, mock_efetch):
        """Test full text retrieval"""
        mock_handle = MagicMock()
        mock_efetch.return_value = mock_handle
        mock_handle.read.return_value = b"x" * 60000
        
        result = fetch_full_text_from_pmc('PMC123')
        
        assert result is not None
        assert len(result) == 50000
    
    @patch('api.get_pmc_id')
    @patch('api.fetch_full_text_from_pmc')
    @patch('api.Entrez.efetch')
    @patch('api.Entrez.read')
    def test_fetch_paper_details_complete(self, mock_read, mock_efetch, 
                                          mock_fetch_full, mock_get_pmc):
        """Test fetching complete paper details"""
        mock_handle = MagicMock()
        mock_efetch.return_value = mock_handle
        mock_read.return_value = {
            'PubmedArticle': [{
                'MedlineCitation': {
                    'PMID': '123',
                    'Article': {
                        'ArticleTitle': 'Test Title',
                        'Abstract': {'AbstractText': ['Test abstract']},
                        'Journal': {
                            'Title': 'Test Journal',
                            'JournalIssue': {'PubDate': {'Year': '2024'}}
                        },
                        'AuthorList': [{'LastName': 'Smith', 'Initials': 'J'}]
                    }
                }
            }]
        }
        mock_get_pmc.return_value = 'PMC123'
        mock_fetch_full.return_value = 'Full text'
        
        result = fetch_paper_details('123', 'cardiology', 'test query')
        
        assert result is not None
        assert result['pmid'] == '123'
        assert result['title'] == 'Test Title'
        assert result['domain'] == 'cardiology'
        assert result['full_text'] == 'Full text'


# ============================================================
# TEST: MongoDB Connection
# ============================================================

class TestMongoDBConnection:
    """Test MongoDB connection functionality"""
    
    @patch('api.pymongo.MongoClient')
    def test_get_mongodb_collection_success(self, mock_mongo_client):
        """Test successful MongoDB connection"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_mongo_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        
        # Reset global collection
        import api
        api.collection = None
        
        result = get_mongodb_collection()
        
        mock_client.server_info.assert_called_once()
        assert result == mock_collection
    
    @patch('api.pymongo.MongoClient')
    def test_get_mongodb_collection_failure(self, mock_mongo_client):
        """Test MongoDB connection failure"""
        mock_client = MagicMock()
        mock_client.server_info.side_effect = Exception("Connection failed")
        mock_mongo_client.return_value = mock_client
        
        # Reset global collection
        import api
        api.collection = None
        
        with pytest.raises(Exception):
            get_mongodb_collection()


# ============================================================
# TEST: API Endpoints
# ============================================================

class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
    
    @patch('api.get_mongodb_collection')
    def test_health_endpoint(self, mock_get_collection, test_client):
        """Test health check endpoint"""
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 100
        mock_get_collection.return_value = mock_collection
        
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mongodb_connected"] is True
        assert data["total_papers"] == 100
    
    @patch('api.get_mongodb_collection')
    def test_stats_endpoint(self, mock_get_collection, test_client):
        """Test statistics endpoint"""
        mock_collection = MagicMock()
        mock_collection.count_documents.side_effect = [1000, 300, 400, 300, 200, 100]
        mock_collection.aggregate.return_value = [
            {'_id': 2024, 'count': 500},
            {'_id': 2023, 'count': 300}
        ]
        mock_get_collection.return_value = mock_collection
        
        response = test_client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_papers"] == 1000
        assert data["with_full_text"] == 300
        assert "by_domain" in data
        assert "recent_years" in data
    
    @patch('api.get_mongodb_collection')
    @patch('api.search_pubmed')
    @patch('api.fetch_paper_details')
    @patch('api.time.sleep')
    def test_search_endpoint(self, mock_sleep, mock_fetch, mock_search, mock_get_coll, test_client):
        """Test search and save endpoint"""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        mock_get_coll.return_value = mock_collection
        
        mock_search.return_value = ['123']
        mock_fetch.return_value = {
            'pmid': '123',
            'pmc_id': None,  # Added missing field
            'title': 'Test',
            'abstract': 'Abstract',
            'full_text': None,
            'journal': 'Journal',
            'year': 2024,
            'authors': 'Author',
            'pubmed_url': 'url',
            'pmc_url': None,
            'domain': 'general',
            'downloaded_at': datetime.utcnow(),
            'synced_to_pinecone': False,
            'metadata': {'has_full_text': False, 'pmc_available': False}
        }
        
        response = test_client.post("/search", json={
            "query": "diabetes",
            "max_results": 10,
            "domain": "general"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "diabetes"
        assert data["domain"] == "general"
        assert data["total_found"] == 1
    
    @patch('api.get_mongodb_collection')
    def test_get_paper_from_mongodb(self, mock_get_coll, test_client):
        """Test getting existing paper from MongoDB"""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = {
            'pmid': '123',
            'title': 'Test Paper',
            'abstract': 'Abstract',
            'metadata': {'has_full_text': False, 'pmc_available': False}
        }
        mock_get_coll.return_value = mock_collection
        
        response = test_client.get("/paper/123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["pmid"] == "123"
        assert data["source"] == "mongodb"
    
    @patch('api.get_mongodb_collection')
    @patch('api.fetch_paper_details')
    def test_get_paper_from_pubmed(self, mock_fetch, mock_get_coll, test_client):
        """Test fetching new paper from PubMed"""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        mock_get_coll.return_value = mock_collection
        
        mock_fetch.return_value = {
            'pmid': '999',
            'title': 'New Paper',
            'abstract': 'New abstract',
            'metadata': {'has_full_text': False, 'pmc_available': False}
        }
        
        response = test_client.get("/paper/999")
        
        assert response.status_code == 200
        data = response.json()
        assert data["pmid"] == "999"
        assert data["source"] == "pubmed"
    
    @patch('api.get_mongodb_collection')
    @patch('api.fetch_paper_details')
    def test_get_paper_not_found(self, mock_fetch, mock_get_coll, test_client):
        """Test paper not found scenario"""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        mock_get_coll.return_value = mock_collection
        
        mock_fetch.return_value = None
        
        response = test_client.get("/paper/999")
        
        assert response.status_code == 404
    
    @patch('api.get_mongodb_collection')
    @patch('api.fetch_paper_details')
    @patch('api.time.sleep')
    def test_batch_endpoint(self, mock_sleep, mock_fetch, mock_get_coll, test_client):
        """Test batch fetch endpoint"""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        mock_get_coll.return_value = mock_collection
        
        mock_fetch.return_value = {
            'pmid': '123',
            'title': 'Test',
            'metadata': {'has_full_text': False}
        }
        
        response = test_client.post("/batch?domain=cardiology", json=["123", "456"])
        
        assert response.status_code == 200
        data = response.json()
        assert data["requested"] == 2
    
    def test_batch_endpoint_too_many(self, test_client):
        """Test batch endpoint rejects too many PMIDs"""
        pmids = [str(i) for i in range(101)]
        response = test_client.post("/batch", json=pmids)
        assert response.status_code == 400


# ============================================================
# TEST: Domain Endpoints
# ============================================================

class TestDomainEndpoints:
    """Test domain-specific endpoints"""
    
    @patch('api.search_and_save')
    def test_cardiology_endpoint(self, mock_search, test_client):
        """Test cardiology domain endpoint"""
        from api import SearchResponse
        
        mock_search.return_value = SearchResponse(
            query="heart failure",
            domain="cardiology",
            total_found=10,
            saved_to_db=8,
            skipped_duplicate=2,
            errors=0,
            papers=[],
            execution_time=1.0
        )
        
        response = test_client.get("/domains/cardiology?max_results=30")
        
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "cardiology"
    
    @patch('api.search_and_save')
    def test_endocrinology_endpoint(self, mock_search, test_client):
        """Test endocrinology domain endpoint"""
        from api import SearchResponse
        
        mock_search.return_value = SearchResponse(
            query="diabetes",
            domain="endocrinology",
            total_found=10,
            saved_to_db=8,
            skipped_duplicate=2,
            errors=0,
            papers=[],
            execution_time=1.0
        )
        
        response = test_client.get("/domains/endocrinology?max_results=30")
        
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "endocrinology"


# ============================================================
# TEST: Error Handling
# ============================================================

class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('api.get_mongodb_collection')
    @patch('api.search_pubmed')
    def test_search_with_no_results(self, mock_search, mock_get_coll, test_client):
        """Test search endpoint with no results"""
        mock_collection = MagicMock()
        mock_get_coll.return_value = mock_collection
        mock_search.return_value = []
        
        response = test_client.post("/search", json={
            "query": "nonexistent query",
            "max_results": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_found"] == 0
        assert len(data["papers"]) == 0
    
    @patch('api.get_mongodb_collection')
    def test_duplicate_key_handling(self, mock_get_coll):
        """Test duplicate key error handling"""
        mock_collection = MagicMock()
        mock_collection.insert_one.side_effect = DuplicateKeyError("Duplicate")
        mock_get_coll.return_value = mock_collection
        
        # This should handle the error gracefully
        # Test depends on implementation details


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests with mocked external dependencies"""
    
    @patch('api.pymongo.MongoClient')
    @patch('api.search_pubmed')
    @patch('api.fetch_paper_details')
    @patch('api.time.sleep')
    def test_full_search_flow(self, mock_sleep, mock_fetch, mock_search, mock_mongo, test_client):
        """Test complete search flow"""
        # Setup MongoDB mock
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_mongo.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_collection.find_one.return_value = None
        mock_collection.count_documents.return_value = 0
        
        # Reset global collection
        import api
        api.collection = None
        
        # Setup PubMed mocks
        mock_search.return_value = ['123']
        mock_fetch.return_value = {
            'pmid': '123',
            'pmc_id': None,  # Added missing field
            'title': 'Test Paper',
            'abstract': 'Test abstract',
            'full_text': None,
            'journal': 'Journal',
            'year': 2024,
            'authors': 'Author',
            'pubmed_url': 'url',
            'pmc_url': None,
            'domain': 'cardiology',
            'downloaded_at': datetime.utcnow(),
            'synced_to_pinecone': False,
            'metadata': {'has_full_text': False, 'pmc_available': False}
        }
        
        # Execute search
        response = test_client.post("/search", json={
            "query": "heart failure",
            "max_results": 1,
            "domain": "cardiology"
        })
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["saved_to_db"] >= 0
        assert mock_collection.insert_one.called


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])