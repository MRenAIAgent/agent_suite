import asyncio
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from tools.adapter.langchain_tool import LangChainToolAdapter, convert_langchain_tools

async def test_langchain_adapter():
    print("Testing LangChain Tool Adapter...")
    
    # Create LangChain tools
    print("Creating LangChain tools...")
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    search_tool = DuckDuckGoSearchRun()
    
    # Test individual tool adaptation
    print("\nTesting individual tool adaptation...")
    try:
        wiki_adapter = LangChainToolAdapter.from_langchain_tool(wikipedia_tool)
        print(f"✓ Successfully created adapter for {wikipedia_tool.__class__.__name__}")
        print(f"  Name: {wiki_adapter.name}")
        print(f"  Description: {wiki_adapter.description}")
        
        # Test running the tool
        result = await wiki_adapter.arun(query="Python programming language")
        print(f"✓ Successfully ran the tool")
        print(f"  Result snippet: {result[:100]}...")
    except Exception as e:
        print(f"✗ Error adapting individual tool: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test batch conversion
    print("\nTesting batch conversion...")
    try:
        tools = convert_langchain_tools([wikipedia_tool, search_tool])
        print(f"✓ Successfully converted {len(tools)} tools")
        for i, tool in enumerate(tools):
            print(f"  Tool {i+1}: {tool.__class__.__name__}")
    except Exception as e:
        print(f"✗ Error in batch conversion: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(test_langchain_adapter()) 