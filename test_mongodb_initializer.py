"""
Unit Tests for MongoDB Initializer
Tests all functions in isolation with mocked MongoDB
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import sys
from datetime import datetime

# Import functions to test
from mongodb_initializer import (
    check_mongodb_connection,
    create_database,
    create_collection,
    create_indexes,
    create_validation_schema,
    print_database_stats
)


# ============================================================
# TEST: MongoDB Connection
# ============================================================

class TestMongoDBConnection:
    """Test MongoDB connection functionality"""
    
    def test_successful_connection(self):
        """Test successful MongoDB connection"""
        mock_client = MagicMock()
        
        with patch('mongodb_initializer.MongoClient', return_value=mock_client):
            client = check_mongodb_connection("mongodb://localhost:27017/")
            
            # Verify ping was called
            mock_client.admin.command.assert_called_once_with('ping')
            assert client == mock_client
    
    def test_failed_connection(self):
        """Test failed MongoDB connection exits"""
        mock_client = MagicMock()
        mock_client.admin.command.side_effect = Exception("Connection refused")
        
        with patch('mongodb_initializer.MongoClient', return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                client = check_mongodb_connection("mongodb://localhost:27017/")
            
            assert exc_info.value.code == 1


# ============================================================
# TEST: Database Creation
# ============================================================

class TestDatabaseCreation:
    """Test database creation"""
    
    def test_create_database(self):
        """Test database creation returns correct db object"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        
        result = create_database(mock_client, "test_db")
        
        mock_client.__getitem__.assert_called_once_with("test_db")
        assert result == mock_db


# ============================================================
# TEST: Collection Creation
# ============================================================

class TestCollectionCreation:
    """Test collection creation and dropping"""
    
    def test_create_new_collection(self):
        """Test creating a new collection"""
        mock_db = MagicMock()
        mock_db.list_collection_names.return_value = []
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        
        result = create_collection(mock_db, "test_collection")
        
        mock_db.list_collection_names.assert_called_once()
        mock_db.__getitem__.assert_called_once_with("test_collection")
        assert result == mock_collection
    
    def test_drop_existing_collection(self):
        """Test dropping existing collection before creating new one"""
        mock_db = MagicMock()
        mock_db.list_collection_names.return_value = ["test_collection"]
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        
        result = create_collection(mock_db, "test_collection")
        
        mock_collection.drop.assert_called_once()
        assert result == mock_collection


# ============================================================
# TEST: Index Creation
# ============================================================

class TestIndexCreation:
    """Test index creation"""
    
    def test_create_all_indexes(self):
        """Test all indexes are created correctly"""
        mock_collection = MagicMock()
        
        create_indexes(mock_collection)
        
        # Check that create_index was called 7 times
        assert mock_collection.create_index.call_count == 7
        
        # Verify specific indexes
        calls = mock_collection.create_index.call_args_list
        
        # Check PMID unique index
        assert calls[0][0][0] == [("pmid", 1)]
        assert calls[0][1]['unique'] == True
        assert calls[0][1]['name'] == "idx_pmid_unique"
        
        # Check domain index
        assert calls[1][0][0] == [("domain", 1)]
        assert calls[1][1]['name'] == "idx_domain"
        
        # Check year index
        assert calls[2][0][0] == [("year", -1)]
        assert calls[2][1]['name'] == "idx_year"
    
    def test_index_creation_with_error(self):
        """Test index creation handles errors gracefully"""
        mock_collection = MagicMock()
        mock_collection.create_index.side_effect = Exception("Index error")
        
        # Should raise exception (not handled in function)
        with pytest.raises(Exception):
            create_indexes(mock_collection)


# ============================================================
# TEST: Validation Schema
# ============================================================

class TestValidationSchema:
    """Test validation schema setup"""
    
    def test_create_validation_schema_success(self):
        """Test successful validation schema creation"""
        mock_db = MagicMock()
        mock_db.command.return_value = {"ok": 1}
        
        create_validation_schema(mock_db, "test_collection")
        
        # Verify command was called
        mock_db.command.assert_called_once()
        call_args = mock_db.command.call_args[0][0]
        
        assert call_args['collMod'] == "test_collection"
        assert 'validator' in call_args
        assert call_args['validationLevel'] == "moderate"
    
    def test_create_validation_schema_failure(self):
        """Test validation schema creation handles errors"""
        mock_db = MagicMock()
        mock_db.command.side_effect = Exception("Schema error")
        
        # Should not raise, just print warning
        create_validation_schema(mock_db, "test_collection")
        
        mock_db.command.assert_called_once()


# ============================================================
# TEST: Database Statistics
# ============================================================

class TestDatabaseStatistics:
    """Test database statistics display"""
    
    def test_print_database_stats_empty_collection(self, capsys):
        """Test stats for empty collection"""
        mock_db = MagicMock()
        mock_db.name = "test_db"
        mock_db.list_collection_names.return_value = ["test_collection"]
        mock_db.command.side_effect = [
            {"dataSize": 0, "collections": 1},  # dbStats
            {"count": 0, "avgObjSize": 0, "size": 0}  # collStats
        ]
        
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.list_indexes.return_value = []
        mock_collection.find_one.return_value = None
        
        print_database_stats(mock_db, mock_collection)
        
        captured = capsys.readouterr()
        assert "DATABASE STATISTICS" in captured.out
        assert "Documents: 0" in captured.out
    
    def test_print_database_stats_with_documents(self, capsys):
        """Test stats for collection with documents"""
        mock_db = MagicMock()
        mock_db.name = "test_db"
        mock_db.list_collection_names.return_value = ["test_collection"]
        mock_db.command.side_effect = [
            {"dataSize": 1024, "collections": 1},
            {"count": 10, "avgObjSize": 512, "size": 5120}
        ]
        
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.list_indexes.return_value = [
            {"name": "idx_pmid", "key": {"pmid": 1}}
        ]
        mock_collection.find_one.return_value = {
            "pmid": "12345",
            "title": "Test Paper",
            "abstract": "This is a test abstract"
        }
        
        print_database_stats(mock_db, mock_collection)
        
        captured = capsys.readouterr()
        assert "Documents: 10" in captured.out
        assert "Sample document:" in captured.out


# ============================================================
# INTEGRATION-LIKE TEST (with full mocking)
# ============================================================

class TestFullInitialization:
    """Test complete initialization flow"""
    
    @patch('mongodb_initializer.MongoClient')
    def test_full_initialization_flow(self, mock_mongo_client):
        """Test the complete initialization process"""
        # Setup mocks
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_mongo_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.return_value = mock_collection
        mock_db.list_collection_names.return_value = []
        mock_db.command.return_value = {"ok": 1, "dataSize": 0, "collections": 1}
        mock_collection.list_indexes.return_value = []
        mock_collection.find_one.return_value = None
        
        # Test connection
        client = check_mongodb_connection("mongodb://localhost:27017/")
        assert client == mock_client
        
        # Test database creation
        db = create_database(client, "test_db")
        assert db == mock_db
        
        # Test collection creation
        collection = create_collection(db, "test_collection")
        assert collection == mock_collection
        
        # Test index creation
        create_indexes(collection)
        assert mock_collection.create_index.called
        
        # Test validation schema
        create_validation_schema(db, "test_collection")
        assert mock_db.command.called


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])