"""
RAG-Enhanced Chat Handler
Combines fine-tuned model with vector database retrieval and tool calling
"""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from .build_vector_db import VectorDatabase
    from .tools import get_tool_registry
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from build_vector_db import VectorDatabase
    from tools import get_tool_registry


class RAGChatHandler:
    """Handles chat with RAG and tool calling"""
    
    def __init__(self, vector_db_path: str = None):
        """
        Initialize RAG chat handler
        
        Args:
            vector_db_path: Path to vector database file
        """
        self.vector_db = None
        self.vector_db_path = vector_db_path
        self.tool_registry = get_tool_registry()
        
        # Load vector DB if path provided
        if vector_db_path and Path(vector_db_path).exists():
            self.load_vector_db(vector_db_path)
    
    def load_vector_db(self, path: str):
        """Load vector database"""
        try:
            self.vector_db = VectorDatabase.load(path)
            print(f"✓ RAG: Loaded vector database with {len(self.vector_db.chunks)} chunks")
        except Exception as e:
            print(f"⚠️  RAG: Could not load vector database: {e}")
            self.vector_db = None
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector database
        
        Args:
            query: User query
            top_k: Number of context chunks to retrieve
            
        Returns:
            List of relevant text chunks with metadata
        """
        if self.vector_db is None:
            return []
        
        try:
            results = self.vector_db.search(query, top_k=top_k)
            return results
        except Exception as e:
            print(f"⚠️  RAG: Context retrieval error: {e}")
            return []
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved context for prompt"""
        if not results:
            return ""
        
        context_parts = ["RELEVANT KNOWLEDGE FROM TEXTS:"]
        
        for i, result in enumerate(results, 1):
            source = result['metadata']['source']
            text = result['text']
            score = result['score']
            
            context_parts.append(f"\n[Source {i}: {source} (relevance: {score:.2f})]")
            context_parts.append(text)
        
        return "\n".join(context_parts)
    
    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response
        
        Format: <tool>tool_name{"arg1": "value1", "arg2": value2}</tool>
        
        Args:
            text: LLM response text
            
        Returns:
            List of tool calls with name and arguments
        """
        tool_pattern = r'<tool>(\w+)(\{[^}]*\})?</tool>'
        matches = re.findall(tool_pattern, text)
        
        tool_calls = []
        for tool_name, args_str in matches:
            try:
                if args_str:
                    arguments = json.loads(args_str)
                else:
                    arguments = {}
                
                tool_calls.append({
                    "name": tool_name,
                    "arguments": arguments
                })
            except json.JSONDecodeError:
                print(f"⚠️  Invalid tool arguments: {args_str}")
                continue
        
        return tool_calls
    
    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]
            
            result = self.tool_registry.execute_tool(tool_name, arguments, context)
            results.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            })
        
        return results
    
    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> str:
        """Format tool results for prompt"""
        if not tool_results:
            return ""
        
        parts = ["TOOL EXECUTION RESULTS:"]
        
        for result in tool_results:
            tool_name = result["tool"]
            tool_result = result["result"]
            
            parts.append(f"\n[Tool: {tool_name}]")
            
            if tool_result.get("success"):
                # Format result nicely
                result_data = tool_result["result"]
                if isinstance(result_data, dict):
                    parts.append(json.dumps(result_data, indent=2))
                else:
                    parts.append(str(result_data))
            else:
                parts.append(f"Error: {tool_result.get('error', 'Unknown error')}")
        
        return "\n".join(parts)
    
    def get_tool_definitions_prompt(self) -> str:
        """Get tool definitions for system prompt"""
        tools = self.tool_registry.get_tool_definitions()
        
        prompt_parts = ["AVAILABLE TOOLS:"]
        prompt_parts.append("You can use these tools to get chart data and perform calculations.")
        prompt_parts.append("To use a tool, format as: <tool>tool_name{\"arg\": \"value\"}</tool>")
        prompt_parts.append("\nTools:")
        
        for tool in tools:
            prompt_parts.append(f"\n- {tool['name']}: {tool['description']}")
            
            if tool['parameters'].get('properties'):
                prompt_parts.append("  Parameters:")
                for param_name, param_info in tool['parameters']['properties'].items():
                    required = param_name in tool['parameters'].get('required', [])
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', '')
                    req_marker = " (required)" if required else " (optional)"
                    prompt_parts.append(f"    - {param_name} ({param_type}){req_marker}: {param_desc}")
        
        prompt_parts.append("\nExample usage:")
        prompt_parts.append('  <tool>calculate_shadbala{"planet": "Jupiter"}</tool>')
        prompt_parts.append('  <tool>get_current_chart{}</tool>')
        
        return "\n".join(prompt_parts)
    
    def build_enhanced_prompt(
        self,
        user_message: str,
        chart_data: Optional[Dict[str, Any]] = None,
        history: List[Dict[str, Any]] = None,
        use_rag: bool = True,
        use_tools: bool = True
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build enhanced prompt with RAG context and tool information
        
        Args:
            user_message: User's message
            chart_data: Current chart data (if available)
            history: Conversation history
            use_rag: Whether to use RAG retrieval
            use_tools: Whether to include tool definitions
            
        Returns:
            (enhanced_prompt, context_dict)
        """
        prompt_parts = []
        
        # System prompt
        system_prompt = """You are an expert Vedic astrologer with deep knowledge of planetary positions, nakshatras, yogas, and birth chart analysis.

IMPORTANT: When analyzing a chart, you MUST use the EXACT planetary positions, dates, and locations provided in the CURRENT CHART DATA section below. DO NOT make up or hallucinate any birth details. Base your analysis ONLY on the provided data."""
        
        prompt_parts.append(system_prompt)
        
        # Add tool definitions if enabled
        if use_tools:
            prompt_parts.append("\n" + self.get_tool_definitions_prompt())
        
        # Retrieve and add RAG context
        rag_context = ""
        if use_rag and self.vector_db is not None:
            retrieved = self.retrieve_context(user_message, top_k=3)
            if retrieved:
                rag_context = self.format_context(retrieved)
                prompt_parts.append("\n" + rag_context)
        
        # Add chart data if available
        chart_context = ""
        if chart_data:
            chart_context = "\n\n" + "="*60 + "\n"
            chart_context += "CURRENT CHART DATA (USE THIS EXACT DATA FOR ANALYSIS):\n"
            chart_context += "="*60 + "\n"
            
            if 'ascendant' in chart_data:
                asc = chart_data['ascendant']
                chart_context += f"\n🔹 ASCENDANT: {asc.get('sign', 'Unknown')} at {asc.get('degree_in_sign', 0):.2f}°\n"
                chart_context += f"   Nakshatra: {asc.get('nakshatra', 'Unknown')}\n"
            
            if 'planets' in chart_data:
                chart_context += "\n🔹 PLANETARY POSITIONS:\n"
                for planet, pos in chart_data['planets'].items():
                    ret_status = " (R)" if pos.get('is_retrograde', False) else ""
                    comb_status = " [Combust]" if pos.get('is_combust', False) else ""
                    chart_context += f"\n   {planet}: {pos.get('sign', '')} {pos.get('degree_in_sign', 0):.2f}°{ret_status}{comb_status}\n"
                    chart_context += f"   └─ Nakshatra: {pos.get('nakshatra', '')}, Pada: {pos.get('nakshatra_pada', 0)}\n"
            
            chart_context += "\n" + "="*60 + "\n"
            chart_context += "ANALYZE ONLY THIS CHART DATA ABOVE. Do not make up birth details.\n"
            chart_context += "="*60 + "\n"
            
            prompt_parts.append(chart_context)
        
        # Add user message
        prompt_parts.append(f"\n\nUSER QUESTION:\n{user_message}")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Context for tool execution
        context = {
            "chart_data": chart_data,
            "user_message": user_message,
            "rag_context": rag_context,
            "chart_context": chart_context
        }
        
        return full_prompt, context
    
    def process_response_with_tools(
        self,
        response: str,
        context: Dict[str, Any],
        max_iterations: int = 3
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Process LLM response, executing tool calls if present
        
        Args:
            response: Initial LLM response
            context: Context for tool execution
            max_iterations: Max tool calling iterations
            
        Returns:
            (final_response, tool_results)
        """
        all_tool_results = []
        current_response = response
        
        for iteration in range(max_iterations):
            # Parse tool calls from response
            tool_calls = self.parse_tool_calls(current_response)
            
            if not tool_calls:
                # No more tool calls, return final response
                break
            
            # Execute tool calls
            tool_results = self.execute_tool_calls(tool_calls, context)
            all_tool_results.extend(tool_results)
            
            # Format results for next iteration
            tool_results_text = self.format_tool_results(tool_results)
            
            # Remove tool tags from response but preserve surrounding text
            current_response = re.sub(r'<tool>.*?</tool>', '[calculation performed]', current_response, flags=re.DOTALL)
            
            # Add tool results to response
            if tool_results_text:
                current_response = f"{current_response}\n\n{tool_results_text}"
        
        # Clean up response - remove any remaining empty lines
        lines = [line for line in current_response.split('\n') if line.strip()]
        final_response = '\n'.join(lines)
        
        return final_response, all_tool_results


# Singleton instance
_rag_handler = None

def get_rag_handler(vector_db_path: str = None) -> RAGChatHandler:
    """Get or create RAG handler singleton"""
    global _rag_handler
    if _rag_handler is None:
        _rag_handler = RAGChatHandler(vector_db_path)
    return _rag_handler