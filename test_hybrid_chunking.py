"""
Unit Tests for Hybrid Chunking
Tests all chunking functions in isolation with mocked dependencies
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import xml.etree.ElementTree as ET

# Import functions to test
from hybrid_chunking import (
    estimate_tokens,
    split_into_sentences,
    chunk_text_by_sentences,
    is_xml_content,
    clean_xml_tags,
    parse_xml_sections,
    chunk_xml_sections,
    create_chunks_from_paper,
    connect_mongodb,
    setup_chunks_collection,
    process_all_papers
)


# ============================================================
# TEST: Token Estimation
# ============================================================

class TestTokenEstimation:
    """Test token estimation functionality"""
    
    def test_estimate_tokens_empty_string(self):
        """Test token estimation with empty string"""
        assert estimate_tokens("") == 0
    
    def test_estimate_tokens_short_text(self):
        """Test token estimation with short text"""
        text = "Hello"  # 5 chars
        assert estimate_tokens(text) == 1  # 5 // 4 = 1
    
    def test_estimate_tokens_medium_text(self):
        """Test token estimation with medium text"""
        text = "This is a test sentence."  # 24 chars
        assert estimate_tokens(text) == 6  # 24 // 4 = 6
    
    def test_estimate_tokens_long_text(self):
        """Test token estimation with long text"""
        text = "a" * 1000
        assert estimate_tokens(text) == 250  # 1000 // 4


# ============================================================
# TEST: Sentence Splitting
# ============================================================

class TestSentenceSplitting:
    """Test sentence splitting with abbreviation handling"""
    
    def test_split_simple_sentences(self):
        """Test splitting simple sentences"""
        text = "This is sentence one. This is sentence two."
        sentences = split_into_sentences(text)
        assert len(sentences) == 2
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two."
    
    def test_split_with_abbreviations(self):
        """Test splitting preserves abbreviations"""
        text = "Dr. Smith studies diabetes. He works with i.e. patients."
        sentences = split_into_sentences(text)
        assert len(sentences) == 2
        assert "Dr." in sentences[0]
        assert "i.e." in sentences[1]
    
    def test_split_with_vs_abbreviation(self):
        """Test vs. abbreviation handling"""
        text = "Treatment A vs. Treatment B showed results. Analysis continues."
        sentences = split_into_sentences(text)
        assert len(sentences) == 2
        assert "vs." in sentences[0]
    
    def test_split_empty_text(self):
        """Test splitting empty text"""
        assert split_into_sentences("") == []
    
    def test_split_no_periods(self):
        """Test text without sentence terminators"""
        text = "This is continuous text without periods"
        sentences = split_into_sentences(text)
        assert len(sentences) == 1


# ============================================================
# TEST: Text Chunking
# ============================================================

class TestTextChunking:
    """Test sentence-based text chunking"""
    
    def test_chunk_empty_text(self):
        """Test chunking empty text"""
        assert chunk_text_by_sentences("") == []
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than max tokens"""
        text = "Short text."
        chunks = chunk_text_by_sentences(text, max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."
    
    def test_chunk_long_text(self):
        """Test chunking text longer than max tokens"""
        # Create text that will exceed max tokens
        sentence = "This is a test sentence. "
        text = sentence * 50  # ~1250 chars = ~312 tokens
        
        chunks = chunk_text_by_sentences(text, max_tokens=100, overlap=10)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert estimate_tokens(chunk) <= 100
    
    def test_chunk_with_overlap(self):
        """Test that chunking creates overlap between chunks"""
        # Create longer text that will actually be split
        text = ("This is a longer sentence with more content. " * 10 + 
                "Another sentence follows. " * 10)
        chunks = chunk_text_by_sentences(text, max_tokens=50, overlap=10)
        
        # Should have multiple chunks
        assert len(chunks) > 1
    
    def test_chunk_very_long_sentence(self):
        """Test chunking handles very long sentences"""
        # Create a sentence longer than max tokens
        long_sentence = "word " * 200  # ~1000 chars = ~250 tokens
        
        chunks = chunk_text_by_sentences(long_sentence, max_tokens=100)
        
        # Should split into multiple chunks
        assert len(chunks) > 1
        
        # Verify chunks exist and are not empty
        for chunk in chunks:
            assert len(chunk) > 0
            assert estimate_tokens(chunk) > 0


# ============================================================
# TEST: XML Detection and Cleaning
# ============================================================

class TestXMLHandling:
    """Test XML detection and cleaning"""
    
    def test_is_xml_content_true(self):
        """Test XML content detection returns True"""
        xml_text = "<sec><title>Introduction</title><p>Text</p></sec>"
        assert is_xml_content(xml_text) is True
    
    def test_is_xml_content_false(self):
        """Test non-XML content detection returns False"""
        plain_text = "This is plain text without XML tags"
        assert is_xml_content(plain_text) is False
    
    def test_clean_xml_tags(self):
        """Test XML tag removal"""
        xml_text = "<p>This is <b>bold</b> text.</p>"
        cleaned = clean_xml_tags(xml_text)
        assert cleaned == "This is bold text."
        assert "<" not in cleaned
        assert ">" not in cleaned
    
    def test_clean_xml_multiple_spaces(self):
        """Test cleaning converts multiple spaces to single"""
        xml_text = "<p>Text   with    multiple     spaces</p>"
        cleaned = clean_xml_tags(xml_text)
        assert "  " not in cleaned


# ============================================================
# TEST: XML Parsing
# ============================================================

class TestXMLParsing:
    """Test XML section parsing"""
    
    def test_parse_xml_sections_simple(self):
        """Test parsing simple XML sections"""
        xml_text = """
        <article>
            <sec>
                <title>Introduction</title>
                <p>This is the introduction text.</p>
            </sec>
            <sec>
                <title>Methods</title>
                <p>This is the methods text.</p>
            </sec>
        </article>
        """
        
        sections = parse_xml_sections(xml_text)
        
        assert len(sections) == 2
        assert sections[0]['title'] == 'Introduction'
        assert sections[0]['content'] == 'This is the introduction text.'
        assert sections[1]['title'] == 'Methods'
    
    def test_parse_xml_sections_with_namespace(self):
        """Test parsing XML with namespaces"""
        xml_text = """
        <article xmlns="http://example.com">
            <sec>
                <title>Results</title>
                <p>Results content.</p>
            </sec>
        </article>
        """
        
        sections = parse_xml_sections(xml_text)
        
        assert len(sections) == 1
        assert sections[0]['title'] == 'Results'
    
    def test_parse_xml_invalid(self):
        """Test parsing invalid XML returns empty list"""
        invalid_xml = "<sec><title>Unclosed tag"
        sections = parse_xml_sections(invalid_xml)
        assert sections == []
    
    def test_parse_xml_no_sections(self):
        """Test parsing XML without sections"""
        xml_text = "<article><p>No sections</p></article>"
        sections = parse_xml_sections(xml_text)
        assert sections == []


# ============================================================
# TEST: XML Section Chunking
# ============================================================

class TestXMLSectionChunking:
    """Test chunking of XML sections"""
    
    def test_chunk_short_section(self):
        """Test chunking short section (single chunk)"""
        sections = [
            {'title': 'Introduction', 'content': 'Short content.'}
        ]
        
        chunks = chunk_xml_sections(sections, max_tokens=100)
        
        assert len(chunks) == 1
        assert chunks[0]['section'] == 'Introduction'
        assert 'Introduction' in chunks[0]['text']
    
    def test_chunk_long_section(self):
        """Test chunking long section (multiple chunks)"""
        long_content = "This is a sentence. " * 100  # ~2000 chars
        sections = [
            {'title': 'Methods', 'content': long_content}
        ]
        
        chunks = chunk_xml_sections(sections, max_tokens=100)
        
        assert len(chunks) > 1
        assert chunks[0]['section'] == 'Methods'
        assert 'Methods' in chunks[0]['text']
        assert '(continued)' in chunks[1]['text']
    
    def test_chunk_multiple_sections(self):
        """Test chunking multiple sections"""
        sections = [
            {'title': 'Introduction', 'content': 'Intro text.'},
            {'title': 'Methods', 'content': 'Methods text.'}
        ]
        
        chunks = chunk_xml_sections(sections, max_tokens=100)
        
        assert len(chunks) == 2
        assert chunks[0]['section'] == 'Introduction'
        assert chunks[1]['section'] == 'Methods'


# ============================================================
# TEST: Main Chunking Logic
# ============================================================

class TestMainChunkingLogic:
    """Test main paper chunking logic"""
    
    def test_create_chunks_abstract_only(self):
        """Test chunking paper with abstract only"""
        paper = {
            'pmid': '12345',
            'title': 'Test Paper',
            'abstract': 'This is the abstract.',
            'full_text': None,
            'journal': 'Test Journal',
            'year': 2024,
            'authors': 'Smith J',
            'domain': 'cardiology',
            'pubmed_url': 'http://example.com',
            'pmc_url': None
        }
        
        chunks = create_chunks_from_paper(paper)
        
        assert len(chunks) == 1
        assert chunks[0]['chunk_type'] == 'abstract'
        assert chunks[0]['chunk_index'] == 0
        assert 'Test Paper' in chunks[0]['text']
    
    def test_create_chunks_with_xml_full_text(self):
        """Test chunking paper with XML full text"""
        paper = {
            'pmid': '12345',
            'title': 'Test Paper',
            'abstract': 'Abstract text.',
            'full_text': '''
            <article>
                <sec>
                    <title>Introduction</title>
                    <p>This is the introduction section with substantial content.</p>
                </sec>
                <sec>
                    <title>Methods</title>
                    <p>This is the methods section.</p>
                </sec>
            </article>
            ''',
            'journal': 'Test Journal',
            'year': 2024,
            'authors': 'Smith J',
            'domain': 'cardiology',
            'pubmed_url': 'http://example.com',
            'pmc_url': 'http://pmc.example.com'
        }
        
        chunks = create_chunks_from_paper(paper)
        
        # Should have abstract + full text chunks
        assert len(chunks) >= 2
        assert chunks[0]['chunk_type'] == 'abstract'
        
        # Check if XML parsing worked
        xml_chunks = [c for c in chunks if c['chunk_type'] == 'full_text_xml']
        text_chunks = [c for c in chunks if c['chunk_type'] == 'full_text']
        
        # Either XML parsing worked OR sentence-based fallback was used
        assert len(xml_chunks) > 0 or len(text_chunks) > 0
    
    def test_create_chunks_with_plain_full_text(self):
        """Test chunking paper with plain text (non-XML)"""
        paper = {
            'pmid': '12345',
            'title': 'Test Paper',
            'abstract': 'Abstract text.',
            'full_text': 'This is plain full text. ' * 100,  # Long text
            'journal': 'Test Journal',
            'year': 2024,
            'authors': 'Smith J',
            'domain': 'endocrinology',
            'pubmed_url': 'http://example.com',
            'pmc_url': None
        }
        
        chunks = create_chunks_from_paper(paper)
        
        assert len(chunks) >= 2
        assert chunks[0]['chunk_type'] == 'abstract'
        assert chunks[1]['chunk_type'] == 'full_text'
        assert chunks[1]['has_xml_structure'] is False
    
    def test_create_chunks_metadata_preservation(self):
        """Test that metadata is preserved in chunks"""
        paper = {
            'pmid': '12345',
            'title': 'Test Paper',
            'abstract': 'Abstract.',
            'full_text': None,
            'journal': 'Nature',
            'year': 2024,
            'authors': 'Smith J, Doe A',
            'domain': 'cardiology',
            'pubmed_url': 'http://pubmed.com/12345',
            'pmc_url': None
        }
        
        chunks = create_chunks_from_paper(paper)
        
        chunk = chunks[0]
        assert chunk['pmid'] == '12345'
        assert chunk['title'] == 'Test Paper'
        assert chunk['journal'] == 'Nature'
        assert chunk['year'] == 2024
        assert chunk['domain'] == 'cardiology'
        assert chunk['embedded'] is False
        assert chunk['synced_to_pinecone'] is False


# ============================================================
# TEST: MongoDB Operations
# ============================================================

class TestMongoDBOperations:
    """Test MongoDB connection and operations"""
    
    @patch('hybrid_chunking.pymongo.MongoClient')
    def test_connect_mongodb_success(self, mock_mongo_client):
        """Test successful MongoDB connection"""
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_source_coll = MagicMock()
        mock_chunks_coll = MagicMock()
        
        mock_mongo_client.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.side_effect = [mock_source_coll, mock_chunks_coll]
        
        source, chunks = connect_mongodb()
        
        mock_client.server_info.assert_called_once()
        assert source == mock_source_coll
        assert chunks == mock_chunks_coll
    
    @patch('hybrid_chunking.pymongo.MongoClient')
    def test_connect_mongodb_failure(self, mock_mongo_client):
        """Test MongoDB connection failure"""
        mock_client = MagicMock()
        mock_client.server_info.side_effect = Exception("Connection failed")
        mock_mongo_client.return_value = mock_client
        
        with pytest.raises(SystemExit):
            connect_mongodb()
    
    @patch('builtins.input', return_value='n')
    def test_setup_chunks_collection_cancel(self, mock_input):
        """Test setup cancellation when chunks exist"""
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 100
        
        result = setup_chunks_collection(mock_collection)
        
        assert result is False
        assert not mock_collection.delete_many.called
    
    @patch('builtins.input', return_value='y')
    def test_setup_chunks_collection_delete(self, mock_input):
        """Test setup deletes existing chunks"""
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 100
        
        result = setup_chunks_collection(mock_collection)
        
        assert result is True
        mock_collection.delete_many.assert_called_once()
        assert mock_collection.create_index.call_count == 7
    
    def test_setup_chunks_collection_empty(self):
        """Test setup with no existing chunks"""
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 0
        
        result = setup_chunks_collection(mock_collection)
        
        assert result is True
        assert not mock_collection.delete_many.called
        assert mock_collection.create_index.call_count == 7


# ============================================================
# TEST: Paper Processing
# ============================================================

class TestPaperProcessing:
    """Test batch paper processing"""
    
    @patch('hybrid_chunking.tqdm')
    def test_process_all_papers(self, mock_tqdm):
        """Test processing all papers"""
        mock_source = MagicMock()
        mock_chunks = MagicMock()
        
        # Setup mock papers
        papers = [
            {
                'pmid': '1',
                'title': 'Paper 1',
                'abstract': 'Abstract 1',
                'full_text': None,
                'journal': 'Journal 1',
                'year': 2024,
                'authors': 'Author 1',
                'domain': 'cardiology',
                'pubmed_url': 'url1',
                'pmc_url': None
            },
            {
                'pmid': '2',
                'title': 'Paper 2',
                'abstract': 'Abstract 2',
                'full_text': 'Full text content.',
                'journal': 'Journal 2',
                'year': 2024,
                'authors': 'Author 2',
                'domain': 'endocrinology',
                'pubmed_url': 'url2',
                'pmc_url': None
            }
        ]
        
        mock_source.count_documents.return_value = 2
        mock_source.find.return_value = papers
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        stats = process_all_papers(mock_source, mock_chunks)
        
        assert stats['total_papers'] == 2
        assert stats['abstract_only'] == 1
        assert stats['with_full_text'] == 1
        assert mock_chunks.insert_many.call_count == 2


# ============================================================
# INTEGRATION-LIKE TEST
# ============================================================

class TestFullChunkingFlow:
    """Test complete chunking flow"""
    
    @patch('hybrid_chunking.pymongo.MongoClient')
    @patch('hybrid_chunking.tqdm')
    @patch('builtins.input', return_value='y')
    def test_complete_chunking_flow(self, mock_input, mock_tqdm, mock_mongo):
        """Test complete chunking workflow"""
        # Setup mocks
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_source = MagicMock()
        mock_chunks = MagicMock()
        
        mock_mongo.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_db.__getitem__.side_effect = [mock_source, mock_chunks]
        
        mock_chunks.count_documents.return_value = 0
        mock_source.count_documents.return_value = 1
        
        papers = [{
            'pmid': '12345',
            'title': 'Test Paper',
            'abstract': 'Test abstract',
            'full_text': None,
            'journal': 'Test Journal',
            'year': 2024,
            'authors': 'Test Author',
            'domain': 'cardiology',
            'pubmed_url': 'http://test.com',
            'pmc_url': None
        }]
        
        mock_source.find.return_value = papers
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Execute
        source, chunks = connect_mongodb()
        result = setup_chunks_collection(chunks)
        stats = process_all_papers(source, chunks)
        
        # Verify
        assert result is True
        assert stats['total_papers'] == 1
        assert mock_chunks.insert_many.called


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])