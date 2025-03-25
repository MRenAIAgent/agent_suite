import unittest
import asyncio
import os
import tempfile
from typing import Dict, Any, List

from agents.examples.fields_extraction.fields_extraction_agent import FieldsExtractionService, DriverLicenseExtractor
from llm.openai.openai_llm import OpenAILLM
from dotenv import load_dotenv


class TestFieldsExtractionService(unittest.TestCase):
    """Test cases for the FieldsExtractionService."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        load_dotenv()
        cls.llm = OpenAILLM.create_llm()
        cls.extraction_service = FieldsExtractionService(cls.llm)
    
    def setUp(self):
        """Set up before each test."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove all files in the temporary directory
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        # Remove the directory
        os.rmdir(self.test_dir)
    
    def create_test_file(self, content: str, filename: str, extension: str = ".txt") -> str:
        """Create a test file with the given content.
        
        Args:
            content: Content to write to the file
            filename: Base name of the file
            extension: File extension
            
        Returns:
            Path to the created file
        """
        file_path = os.path.join(self.test_dir, f"{filename}{extension}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path
    
    def test_init(self):
        """Test that the service initializes correctly."""
        service = FieldsExtractionService(self.llm)
        self.assertIsNotNone(service.agent)
        self.assertIsInstance(service.tools[0], DriverLicenseExtractor)
        self.assertEqual(len(service.default_fields), 8)
    
    def test_set_default_fields(self):
        """Test setting default fields."""
        service = FieldsExtractionService(self.llm)
        new_fields = ["field1", "field2", "field3"]
        service.set_default_fields(new_fields)
        self.assertEqual(service.default_fields, new_fields)
    
    def test_extract_from_text_with_missing_fields(self):
        """Test extracting from text with missing fields."""
        async def _test():
            text = """
            NAME: Smith, John
            """
            result = await self.extraction_service.extract_from_text(text)
            self.assertEqual(result["source"], "text")
            self.assertEqual(len(result["fields_requested"]), 8)
            return result
        
        # Run the async test
        result = asyncio.run(_test())
        print(f"Missing fields test result: {result}")
    
    def test_extract_from_text_with_custom_fields(self):
        """Test extracting specific fields from text."""
        async def _test():
            text = """
            DRIVER LICENSE
            NAME: DOE, JANE E
            DATE OF BIRTH: 02/15/1985
            EXPIRATION: 02/15/2026
            LICENSE #: F9876543
            GENDER: F
            STATE: NY
            """
            
            custom_fields = ["first_name", "last_name", "license_number"]
            result = await self.extraction_service.extract_from_text(
                text, 
                fields=custom_fields
            )
            
            self.assertEqual(result["source"], "text")
            self.assertEqual(result["fields_requested"], custom_fields)
            return result
        
        # Run the async test
        result = asyncio.run(_test())
        print(f"Custom fields test result: {result}")
    
    def test_extract_from_file_txt(self):
        """Test extracting from a .txt file."""
        async def _test():
            content = """
            DRIVER LICENSE
            NAME: DOE, JOHN M
            DOB: 01/01/1980
            EXP: 01/01/2025
            LICENSE #: D12345678
            GENDER: M
            STATE: CA
            """
            
            file_path = self.create_test_file(content, "license", ".txt")
            result = await self.extraction_service.extract_from_file(file_path)
            
            self.assertEqual(result["source"], "file")
            self.assertEqual(result["filename"], "license.txt")
            return result
        
        # Run the async test
        result = asyncio.run(_test())
        print(f"Text file test result: {result}")
    
    def test_extract_from_file_md(self):
        """Test extracting from a .md file."""
        async def _test():
            content = """
            # Driver License Information
            
            - **Name**: Wilson, Robert J
            - **Date of Birth**: 03/22/1972
            - **Expiration Date**: 03/22/2024
            - **License Number**: W7654321
            - **Gender**: M
            - **State**: TX
            """
            
            file_path = self.create_test_file(content, "license", ".md")
            result = await self.extraction_service.extract_from_file(file_path)
            
            self.assertEqual(result["source"], "file")
            self.assertEqual(result["filename"], "license.md")
            return result
        
        # Run the async test
        result = asyncio.run(_test())
        print(f"Markdown file test result: {result}")
    
    def test_extract_from_file_csv(self):
        """Test extracting from a .csv file."""
        async def _test():
            content = """field,value
            name,Brown Sandra L
            dob,11/05/1990
            expiration,11/05/2026
            license,B4567890
            gender,F
            state,FL
            """
            
            file_path = self.create_test_file(content, "license", ".csv")
            result = await self.extraction_service.extract_from_file(file_path)
            
            self.assertEqual(result["source"], "file")
            self.assertEqual(result["filename"], "license.csv")
            return result
        
        # Run the async test
        result = asyncio.run(_test())
        print(f"CSV file test result: {result}")
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        async def _test():
            with self.assertRaises(FileNotFoundError):
                await self.extraction_service.extract_from_file("nonexistent_file.txt")
        
        # Run the async test
        asyncio.run(_test())
    
    def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        async def _test():
            content = "Some binary content"
            file_path = self.create_test_file(content, "document", ".pdf")
            
            with self.assertRaises(ValueError) as context:
                await self.extraction_service.extract_from_file(file_path)
            
            self.assertIn("Unsupported file type", str(context.exception))
        
        # Run the async test
        asyncio.run(_test())
    
    def test_different_license_formats(self):
        """Test extraction from licenses with different formats."""
        async def _test():
            # Format 1: Standard format
            format1 = """
            DRIVER LICENSE
            NAME: DOE, JOHN M
            DOB: 01/01/1980
            EXP: 01/01/2025
            LICENSE #: D12345678
            GENDER: M
            STATE: CA
            """
            
            # Format 2: Different labels and order
            format2 = """
            OPERATOR LICENSE
            LAST NAME: SMITH
            FIRST NAME: JANE
            MIDDLE INITIAL: A
            DATE OF BIRTH: 02/15/1985
            GENDER: F
            ID NUMBER: S9876543
            ISSUED: STATE OF NY
            VALID UNTIL: 02/15/2026
            """
            
            # Format 3: Single line format
            format3 = "DL: W4321987 | NAME: WILSON, ROBERT J | DOB: 03/22/1972 | EXPIRES: 03/22/2024 | SEX: M | STATE: TX"
            
            results = []
            for i, content in enumerate([format1, format2, format3]):
                file_path = self.create_test_file(content, f"license_format{i+1}", ".txt")
                result = await self.extraction_service.extract_from_file(file_path)
                results.append(result)

            return results

        # Run the async test
        results = asyncio.run(_test())

        for i, result in enumerate(results, 1):
            print(f"\nLicense Format {i} test result:")
            print(result)


if __name__ == "__main__":
    unittest.main() 