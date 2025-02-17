"""
Unit tests for the CodebaseImproverParser
"""

import pytest
import os
import tempfile
import xml.etree.ElementTree as ET
from deepdroid.parsers.codebase_improver import CodebaseImproverParser, MemoryEntry

@pytest.fixture
def parser():
    """Create a temporary parser instance with a temporary memory file"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        memory_file = f.name
    
    parser = CodebaseImproverParser(memory_file=memory_file)
    yield parser
    
    # Cleanup
    if os.path.exists(memory_file):
        os.unlink(memory_file)

@pytest.fixture
def test_file():
    """Create a temporary test file"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Test content\nLine 2\nLine 3")
        filename = f.name
    
    yield filename
    
    # Cleanup
    if os.path.exists(filename):
        os.unlink(filename)

@pytest.fixture
def sample_system_prompt():
    """Sample system prompt with memory section"""
    return """Some text before
<memory_contents>
1: blank - no memories stored yet
</memory_contents>

Memory buffer usage: 0/100 entries (0%)

Some text after"""

def create_element(tag: str, text: str = None, attrib: dict = None) -> ET.Element:
    """Helper function to create XML elements for testing"""
    element = ET.Element(tag)
    if text:
        element.text = text
    if attrib:
        element.attrib = attrib
    return element

class TestCodebaseImproverParser:
    def test_init(self, parser):
        """Test parser initialization"""
        assert parser.memories == []
        assert parser.max_memory == 100
        assert os.path.exists(parser.memory_file)

    def test_memory_append(self, parser):
        """Test memory append functionality"""
        element = create_element('memory_append', text='test memory entry')
        result = parser._handle_memory_append(element)
        
        assert len(parser.memories) == 1
        assert parser.memories[0].content == 'test memory entry'
        assert 'Memory appended' in result

    def test_memory_remove(self, parser):
        """Test memory remove functionality"""
        # First add a memory
        append_element = create_element('memory_append', text='test memory')
        parser._handle_memory_append(append_element)
        
        # Then remove it
        remove_element = create_element('memory_remove', attrib={'id': '1'})
        result = parser._handle_memory_remove(remove_element)
        
        assert len(parser.memories) == 0
        assert 'Memory 1 removed' in result

    def test_memory_persistence(self, parser):
        """Test that memories persist to file"""
        element = create_element('memory_append', text='persistent memory')
        parser._handle_memory_append(element)
        
        # Create new parser instance with same memory file
        new_parser = CodebaseImproverParser(memory_file=parser.memory_file)
        assert len(new_parser.memories) == 1
        assert new_parser.memories[0].content == 'persistent memory'

    def test_handle_thinking(self, parser):
        """Test thinking handler"""
        element = create_element('thinking', text='thinking about code')
        result = parser._handle_thinking(element)
        assert result is None

    def test_handle_bash(self, parser):
        """Test bash command execution"""
        element = create_element('bash', text='echo "test command"')
        result = parser._handle_bash(element)
        
        assert 'test command' in result
        assert 'Command output' in result

    def test_handle_read_file(self, parser, test_file):
        """Test file reading"""
        element = create_element('read_file', attrib={'filename': test_file})
        result = parser._handle_read_file(element)
        
        assert 'Test content' in result
        assert 'Line 2' in result
        assert 'Line 3' in result

    def test_handle_write_file(self, parser):
        """Test file writing"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        
        element = create_element('write_file', 
                               text='new content',
                               attrib={'filename': filename})
        result = parser._handle_write_file(element)
        
        assert 'written successfully' in result
        with open(filename, 'r') as f:
            content = f.read()
        assert content == 'new content'
        
        os.unlink(filename)

    def test_handle_backup(self, parser, test_file):
        """Test file backup creation"""
        element = create_element('backup_file', attrib={'filename': test_file})
        result = parser._handle_backup(element)
        
        backup_file = f"{test_file}.bak"
        assert os.path.exists(backup_file)
        assert 'Created backup' in result
        
        with open(backup_file, 'r') as f:
            backup_content = f.read()
        with open(test_file, 'r') as f:
            original_content = f.read()
        
        assert backup_content == original_content
        os.unlink(backup_file)

    def test_handle_patch(self, parser, test_file):
        """Test patch application"""
        patch_content = f"""
--- {test_file}
+++ {test_file}
@@ -1,3 +1,3 @@
 Test content
-Line 2
+Modified Line 2
 Line 3"""
        
        element = create_element('patch', 
                               text=patch_content,
                               attrib={'filename': test_file})
        result = parser._handle_patch(element)
        
        assert 'Successfully patched' in result
        with open(test_file, 'r') as f:
            content = f.read()
        assert 'Modified Line 2' in content

    def test_parse_invalid_xml(self, parser):
        """Test handling of invalid XML"""
        invalid_xml = '<invalid>'
        result = parser.parse(invalid_xml)
        assert 'Error parsing LLM output as XML' in result

    def test_memory_limit(self, parser):
        """Test that memory limit is enforced"""
        parser.max_memory = 2
        
        # Add three memories
        for i in range(3):
            element = create_element('memory_append', text=f'memory {i}')
            parser._handle_memory_append(element)
        
        assert len(parser.memories) == 2
        assert parser.memories[0].content == 'memory 1'
        assert parser.memories[1].content == 'memory 2'

    def test_handle_tests(self, parser, tmp_path):
        """Test test execution handler"""
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        
        # Test file with both passing and failing tests
        test_file = test_dir / "test_sample.py"
        test_file.write_text("""
def test_pass():
    assert True

def test_fail():
    assert False

def test_error():
    raise ValueError("Test error")
    """)
        
        # Test successful execution
        element = create_element('run_tests', attrib={'path': str(test_dir)})
        result = parser._handle_tests(element)
        
        # Verify test execution details
        assert 'collected 3 items' in result
        assert '2 failed, 1 passed' in result  # Changed order to match pytest output
        assert 'test_fail' in result  # Should show the failing test name
        assert 'ValueError: Test error' in result  # Should show the error
        
        # Test invalid path
        invalid_element = create_element('run_tests', attrib={'path': 'nonexistent/path'})
        invalid_result = parser._handle_tests(invalid_element)
        assert 'ERROR: file or directory not found: nonexistent/path' in invalid_result

    def test_execute_command_unknown(self, parser):
        """Test handling of unknown commands"""
        element = create_element('unknown_command')
        result = parser._execute_command(element)
        assert 'Unknown command tag' in result 

    def test_system_prompt_update_on_memory_append(self, parser, sample_system_prompt):
        """Test that system prompt is updated when memory is appended"""
        parser.set_system_prompt(sample_system_prompt)
        
        element = create_element('memory_append', text='test memory entry')
        parser._handle_memory_append(element)
        
        updated_prompt = parser.get_system_prompt()
        assert "1: test memory entry" in updated_prompt
        assert "Memory buffer usage: 1/100 entries (1%)" in updated_prompt
        assert "blank - no memories stored yet" not in updated_prompt

    def test_system_prompt_update_on_memory_remove(self, parser, sample_system_prompt):
        """Test that system prompt is updated when memory is removed"""
        parser.set_system_prompt(sample_system_prompt)
        
        # First add a memory
        append_element = create_element('memory_append', text='test memory')
        parser._handle_memory_append(append_element)
        
        # Then remove it
        remove_element = create_element('memory_remove', attrib={'id': '1'})
        parser._handle_memory_remove(remove_element)
        
        updated_prompt = parser.get_system_prompt()
        assert "1: blank - no memories stored yet" in updated_prompt
        assert "Memory buffer usage: 0/100 entries (0%)" in updated_prompt
        assert "test memory" not in updated_prompt

    def test_system_prompt_update_multiple_memories(self, parser, sample_system_prompt):
        """Test system prompt updates with multiple memories"""
        parser.set_system_prompt(sample_system_prompt)
        
        # Add three memories
        for i in range(3):
            element = create_element('memory_append', text=f'memory {i}')
            parser._handle_memory_append(element)
        
        updated_prompt = parser.get_system_prompt()
        assert "1: memory 0" in updated_prompt
        assert "2: memory 1" in updated_prompt
        assert "3: memory 2" in updated_prompt
        assert "Memory buffer usage: 3/100 entries (3%)" in updated_prompt

    def test_system_prompt_update_on_max_memories(self, parser, sample_system_prompt):
        """Test system prompt updates when max memories is reached"""
        parser.max_memory = 2  # Set small max for testing
        parser.set_system_prompt(sample_system_prompt)
        
        # Add three memories (exceeding max)
        for i in range(3):
            element = create_element('memory_append', text=f'memory {i}')
            parser._handle_memory_append(element)
        
        updated_prompt = parser.get_system_prompt()
        # After adding 3 memories with max_memory=2:
        # - memory 0 should be removed
        # - memory 1 should become id 1
        # - memory 2 should become id 2
        assert "memory 0" not in updated_prompt  # First memory should be removed
        assert "1: memory 1" in updated_prompt  # Second memory becomes first
        assert "2: memory 2" in updated_prompt  # Third memory becomes second
        assert "Memory buffer usage: 2/2 entries (100%)" in updated_prompt

    def test_system_prompt_not_set(self, parser):
        """Test that memory operations work even if system prompt is not set"""
        element = create_element('memory_append', text='test memory')
        result = parser._handle_memory_append(element)
        
        assert result == "Memory appended: test memory"
        assert parser.get_system_prompt() is None 